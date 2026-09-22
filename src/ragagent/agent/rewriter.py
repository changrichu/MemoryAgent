"""
Query 改写层:指代消解 + 检索优化
"""
from typing import List, Dict, Optional

from ..llm_client import get_llm_client
from ..memory import L1WorkingMemory, L3LongTermMemory
from ..prompts import REWRITER_PROMPT
from ..config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Rewriter:
    """Query 改写 + 指代消解"""

    def __init__(
        self,
        l1: Optional[L1WorkingMemory] = None,
        l3: Optional[L3LongTermMemory] = None,
    ):
        self.llm = get_llm_client()
        self.l1 = l1 or L1WorkingMemory()
        self.l3 = l3 or L3LongTermMemory()

    def rewrite(
        self,
        user_id: str,
        query: str,
        history: List[Dict] = None,
    ) -> Dict:
        """
        1. 先从 L3 拉 3 条可能的上下文(指代消解)
        2. 让 LLM 基于上下文改写
        """
        # 1. 拉 L3 提示(轻量)
        hints = self.l3.recall(
            user_id=user_id,
            query=query,
            top_k=3,
            score_threshold=0.4,  # 阈值更低,只为提示
        )
        hints_text = "\n".join(
            f"- {h['content']}" for h in hints
        ) or "(无相关历史)"

        history_text = "\n".join(
            f"[{m.get('role','user')}] {m.get('content','')}"
            for m in (history or [])[-5:]
        ) or "(无对话历史)"

        prompt = REWRITER_PROMPT.format(
            history=history_text,
            retrieval_hints=hints_text,
            query=query,
        )

        result = self.llm.chat_json(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        rewritten = result.get("rewritten_query", query)
        sub_queries = result.get("sub_queries", [])
        logger.info(f"改写: {query[:30]} → {rewritten[:60]}")
        return {
            "original_query": query,
            "rewritten_query": rewritten,
            "sub_queries": sub_queries or [rewritten],
        }
