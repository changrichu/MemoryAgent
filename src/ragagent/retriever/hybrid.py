"""
混合检索器: BM25 + 向量 + (可选)知识图谱 三路召回
"""
from typing import List, Dict, Optional
from dataclasses import dataclass

from ..memory import L1WorkingMemory, L2EpisodicMemory, L3LongTermMemory
from .reranker import BGEReranker
from ..config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalHit:
    content: str
    score: float
    source: str
    metadata: Optional[Dict] = None


class HybridRetriever:
    """同时从 L1 / L2 / L3 召回,合并去重,送 Reranker"""

    def __init__(
        self,
        l1: Optional[L1WorkingMemory] = None,
        l2: Optional[L2EpisodicMemory] = None,
        l3: Optional[L3LongTermMemory] = None,
        reranker: Optional[BGEReranker] = None,
        graph: Optional[object] = None,
    ):
        self.l1 = l1 or L1WorkingMemory()
        self.l2 = l2 or L2EpisodicMemory()
        self.l3 = l3 or L3LongTermMemory()
        self.graph = graph  # 可选:Neo4j 图遍历
        self.reranker = reranker or BGEReranker()

    def retrieve(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        l3_candidates: int = 20,
        use_graph: bool = False,
    ) -> List[Dict]:
        """
        完整检索流程:
          1. L1 直接拿最近对话(无 RAG,作为短期上下文)
          2. L2 / L3 向量召回
          3. (可选)图遍历
          4. 合并候选
          5. Reranker 精排到 top_k

        返回:
          {
            "l1": [...],         # 短期上下文
            "retrieved": [...],  # Rerank 后的最终候选
            "all_candidates": [...]  # 供调试
          }
        """
        # 1. L1 工作记忆(直接拿)
        l1_msgs = self.l1.get_recent(user_id, n=settings.l1_max_messages)
        l1_docs = [
            {
                "content": f"[{m['role']}] {m['content']}",
                "source": "L1",
                "score": 1.0,
            }
            for m in l1_msgs
        ]

        # 2. L2 情景记忆
        l2_docs = self.l2.recall(
            user_id=user_id,
            query=query,
            top_k=settings.l3_top_k // 2,
            days=30,
            score_threshold=settings.retrieval_score_threshold,
        )

        # 3. L3 长期记忆
        l3_docs = self.l3.recall(
            user_id=user_id,
            query=query,
            top_k=l3_candidates,
            score_threshold=settings.retrieval_score_threshold,
        )

        # 4. 用户画像(当作"软文档")
        profile = self.l3.get_profile(user_id)
        profile_docs = []
        if profile:
            profile_text = "用户画像(已知偏好): " + " | ".join(
                f"{k}={v}" for k, v in profile.items()
            )
            profile_docs.append({
                "content": profile_text,
                "source": "L3_profile",
                "score": 0.95,
            })

        # 5. (可选)图遍历
        graph_docs = []
        if use_graph and self.graph:
            graph_docs = self.graph.retrieve(query, user_id)

        # 6. 合并候选(去重)
        all_candidates = (
            profile_docs + l2_docs + l3_docs + graph_docs
        )
        all_candidates = self._dedupe(all_candidates)

        # 7. Reranker 精排
        retrieved = self.reranker.rerank(
            query=query,
            candidates=all_candidates,
            top_k=top_k,
        )

        return {
            "l1": l1_docs,
            "retrieved": retrieved,
            "all_candidates": all_candidates,
        }

    @staticmethod
    def _dedupe(candidates: List[Dict]) -> List[Dict]:
        """简单的基于内容去重"""
        seen = set()
        out = []
        for c in candidates:
            content = c.get("content", "")
            key = content[:100]
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    def health_check(self) -> Dict:
        """健康检查 - 用于 API"""
        status = {}
        try:
            self.l1.redis.ping()
            status["redis"] = "ok"
        except Exception as e:
            status["redis"] = f"err: {e}"
        try:
            self.l2.conn.cursor().execute("SELECT 1")
            status["postgres"] = "ok"
        except Exception as e:
            status["postgres"] = f"err: {e}"
        try:
            n = self.l3.coll.num_entities
            status["milvus"] = f"ok ({n} entities)"
        except Exception as e:
            status["milvus"] = f"err: {e}"
        return status
