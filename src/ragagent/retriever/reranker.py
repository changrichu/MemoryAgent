"""
Re-ranker 精排模块
- Cross-Encoder 精度高于 Bi-Encoder
- 适合对 Top 50-100 候选做最后排序
"""
from typing import List, Dict
from functools import lru_cache

from ..config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BGEReranker:
    """BGE-Reranker-v2-M3 二次精排"""

    def __init__(self, model=None):
        self.model = model or self._load_model()

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_model():
        from sentence_transformers import CrossEncoder
        logger.info(f"加载 Reranker 模型: {settings.reranker_model}")
        return CrossEncoder(
            settings.reranker_model,
            device=settings.reranker_device,
        )

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
        key: str = "content",
    ) -> List[Dict]:
        """对候选列表精排,返回 top_k"""
        if not candidates:
            return []

        pairs = [[query, c[key]] for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        # 加分排序
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        results = []
        for cand, score in ranked[:top_k]:
            item = dict(cand)
            item["rerank_score"] = float(score)
            item["final_score"] = float(score)
            results.append(item)

        return results
