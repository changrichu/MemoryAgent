"""
路由层:决定要不要 RAG + 走哪条路径
"""
from typing import List, Dict

from ..llm_client import get_llm_client
from ..prompts import ROUTER_PROMPT
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Router:
    """Agent 路由决策"""

    def __init__(self):
        self.llm = get_llm_client()

    def decide(
        self,
        query: str,
        history: List[Dict] = None,
    ) -> Dict:
        """决策路由:direct / retrieve_short / retrieve_long / retrieve_all"""
        history_text = "\n".join(
            f"[{m.get('role','user')}] {m.get('content','')}"
            for m in (history or [])[-5:]
        ) or "(无历史)"

        prompt = ROUTER_PROMPT.format(
            history=history_text,
            query=query,
        )

        result = self.llm.chat_json(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        decision = result.get("decision", "retrieve_short")
        logger.info(f"Router 决策: {decision} | {result.get('reason','')}")
        return result

    @staticmethod
    def needs_retrieval(decision: str) -> bool:
        return decision in ("retrieve_short", "retrieve_long", "retrieve_all")
