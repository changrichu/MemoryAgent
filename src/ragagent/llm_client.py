"""
统一的 LLM / Embedding 客户端封装
"""
from typing import List, Optional
from functools import lru_cache
from openai import OpenAI

from .config import settings
from .utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """OpenAI 兼容接口的 LLM 客户端"""

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
            timeout=60.0,
        )
        self.model = settings.llm_model

    def chat(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        response_format: Optional[dict] = None,
    ) -> str:
        """对话式调用"""
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_format:
            kwargs["response_format"] = response_format

        try:
            resp = self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise

    def chat_json(self, messages: List[dict], **kw) -> dict:
        """结构化 JSON 输出"""
        import json
        text = self.chat(
            messages,
            response_format={"type": "json_object"},
            **kw,
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 兜底提取 JSON
            import re
            m = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(m.group()) if m else {}

    def embed(self, text: str) -> List[float]:
        """单文本 Embedding(BGE 走 sentence-transformers)"""
        from sentence_transformers import SentenceTransformer
        model = _get_embedding_model()
        return model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量 Embedding"""
        from sentence_transformers import SentenceTransformer
        model = _get_embedding_model()
        return model.encode(texts, normalize_embeddings=True).tolist()


@lru_cache(maxsize=1)
def _get_embedding_model():
    """单例 Embedding 模型"""
    from sentence_transformers import SentenceTransformer
    logger.info(f"加载 Embedding 模型: {settings.embedding_model}")
    return SentenceTransformer(
        settings.embedding_model,
        device=settings.embedding_device,
    )


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """单例 LLM 客户端"""
    return LLMClient()
