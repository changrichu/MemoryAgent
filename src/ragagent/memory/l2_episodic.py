"""
L2 情景记忆 (Episodic Memory)
- 会话级摘要 + 关键事实
- 存储在 PostgreSQL (pgvector) 中
- 跨轮次但单用户级生命周期
"""
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np

from ..config import settings
from ..llm_client import get_llm_client
from ..utils.logger import get_logger

logger = get_logger(__name__)


class L2EpisodicMemory:
    """会话摘要 + 关键事实 + 向量检索"""

    def __init__(self):
        self.conn = psycopg2.connect(settings.postgres_url)
        self.conn.autocommit = True
        self.llm = get_llm_client()
        self._init_schema()

    def _init_schema(self):
        """初始化表结构"""
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS l2_summaries (
                    id BIGSERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    summary TEXT NOT NULL,
                    facts JSONB DEFAULT '[]',
                    embedding VECTOR(%s),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """, (settings.embedding_dim,))
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_l2_user_id
                ON l2_summaries (user_id, created_at DESC);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_l2_embedding
                ON l2_summaries USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
        logger.info("L2 表结构初始化完成")

    def add_summary(
        self,
        user_id: str,
        summary: str,
        facts: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ):
        """写入一条摘要 + 事实"""
        embedding = self.llm.embed(summary)
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO l2_summaries
                    (user_id, session_id, summary, facts, embedding)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                user_id,
                session_id,
                summary,
                json.dumps(facts or [], ensure_ascii=False),
                embedding,
            ))

    def maybe_summarize(self, user_id: str, messages: List[Dict]):
        """当消息数达阈值,触发一次摘要"""
        if len(messages) < settings.l2_summary_trigger:
            return None

        text = "\n".join(
            f"[{m['role']}] {m['content']}" for m in messages[-20:]
        )
        prompt = f"""请将以下对话总结为一段简洁的摘要(不超过 200 字),
并抽出 3-5 条关键事实(用户偏好、需求、提到的实体等)。

对话:
{text}

输出严格 JSON: {{"summary": "...", "facts": ["...", "..."]}}
"""
        result = self.llm.chat_json(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        summary = result.get("summary", "")
        facts = result.get("facts", [])
        if summary:
            self.add_summary(user_id, summary, facts)
            logger.info(f"L2 摘要生成:user={user_id}, facts={len(facts)}")

    def recall(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        days: int = 7,
        score_threshold: float = 0.5,
    ) -> List[Dict]:
        """向量检索最近 N 天的相关摘要"""
        embedding = self.llm.embed(query)
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    summary,
                    facts,
                    created_at,
                    1 - (embedding <=> %s::vector) AS score
                FROM l2_summaries
                WHERE user_id = %s
                  AND created_at > NOW() - INTERVAL '%s days'
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (embedding, user_id, days, embedding, top_k))
            rows = cur.fetchall()

        results = []
        for r in rows:
            if r["score"] < score_threshold:
                continue
            results.append({
                "summary": r["summary"],
                "facts": r["facts"],
                "score": float(r["score"]),
                "created_at": r["created_at"].isoformat(),
                "source": "L2",
            })
        return results
