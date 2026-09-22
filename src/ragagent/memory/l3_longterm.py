"""
L3 长期语义记忆 (Long-term Semantic Memory)
- 跨会话持久化的事实 / 用户画像 / 知识库
- 存储: Milvus 向量库 + Redis KV 标签
- 支持版本化、衰减、合并
"""
from typing import List, Dict, Optional
from datetime import datetime
import json
import time

import redis
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)

from ..config import settings
from ..llm_client import get_llm_client
from ..utils.logger import get_logger

logger = get_logger(__name__)


class L3LongTermMemory:
    """向量 + KV 双存储"""

    COLLECTION = "l3_longterm"
    PROFILE_PREFIX = "l3:profile:"

    def __init__(self):
        self._connect_milvus()
        self._ensure_collection()
        self.redis = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            decode_responses=True,
        )
        self.llm = get_llm_client()

    def _connect_milvus(self):
        try:
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port,
                user=settings.milvus_user,
                password=settings.milvus_password,
            )
            logger.info("Milvus 连接成功")
        except Exception as e:
            logger.warning(f"Milvus 连接失败: {e}")

    def _ensure_collection(self):
        if utility.has_collection(self.COLLECTION):
            self.coll = Collection(self.COLLECTION)
        else:
            fields = [
                FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema("user_id", DataType.VARCHAR, max_length=128),
                FieldSchema("content", DataType.VARCHAR, max_length=4000),
                FieldSchema("tags", DataType.VARCHAR, max_length=512),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=settings.embedding_dim),
                FieldSchema("ts", DataType.INT64),
                FieldSchema("version", DataType.INT64),
            ]
            schema = CollectionSchema(fields, description="L3 long-term memory")
            self.coll = Collection(self.COLLECTION, schema=schema)
            # 建索引
            self.coll.create_index(
                "embedding",
                {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128},
                },
            )
            logger.info(f"创建 Milvus collection: {self.COLLECTION}")

        self.coll.load()

    # ============ 事实写入 ============
    def add_fact(
        self,
        user_id: str,
        content: str,
        tags: Optional[List[str]] = None,
    ):
        """写入一条长期记忆"""
        embedding = self.llm.embed(content)
        data = [
            [user_id],
            [content],
            [",".join(tags or [])],
            [embedding],
            [int(time.time())],
            [1],
        ]
        self.coll.insert(data)
        self.coll.flush()
        logger.info(f"L3 写入:user={user_id}, len={len(content)}")

    def add_facts_batch(self, user_id: str, contents: List[str]):
        """批量写入"""
        if not contents:
            return
        embeddings = self.llm.embed_batch(contents)
        data = [
            [user_id] * len(contents),
            contents,
            [""] * len(contents),
            embeddings,
            [int(time.time())] * len(contents),
            [1] * len(contents),
        ]
        self.coll.insert(data)
        self.coll.flush()

    # ============ 事实检索 ============
    def recall(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.5,
    ) -> List[Dict]:
        """向量检索"""
        if self.coll.num_entities == 0:
            return []

        embedding = self.llm.embed(query)
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        results = self.coll.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=f'user_id == "{user_id}"',
            output_fields=["content", "tags", "ts", "version"],
        )

        out = []
        for hits in results:
            for h in hits:
                score = float(h.score)
                if score < score_threshold:
                    continue
                out.append({
                    "content": h.entity.get("content"),
                    "tags": h.entity.get("tags", ""),
                    "score": score,
                    "ts": h.entity.get("ts"),
                    "version": h.entity.get("version"),
                    "source": "L3",
                })
        return out

    # ============ 用户画像 (KV) ============
    def get_profile(self, user_id: str) -> Dict:
        """读取用户画像"""
        key = f"{self.PROFILE_PREFIX}{user_id}"
        return self.redis.hgetall(key) or {}

    def update_profile(self, user_id: str, fields: Dict[str, str]):
        """更新画像字段"""
        if not fields:
            return
        self.redis.hset(
            f"{self.PROFILE_PREFIX}{user_id}",
            mapping={k: str(v) for k, v in fields.items()},
        )

    def update_profile_field(self, user_id: str, field: str, value: str):
        self.redis.hset(f"{self.PROFILE_PREFIX}{user_id}", field, str(value))

    # ============ 事实抽取(LLM 辅助) ============
    def extract_and_store(self, user_id: str, text: str):
        """从对话里抽取事实,并去重写入"""
        prompt = f"""从以下文本中抽出可作为长期记忆的事实(用户偏好、习惯、提及的关键信息)。
严格输出 JSON: {{"facts": ["事实1", "事实2", ...], "profile_updates": {{"key": "value"}}}}
如果没有,输出空。

文本: {text}
"""
        result = self.llm.chat_json(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        facts = result.get("facts", [])
        profile = result.get("profile_updates", {})

        if facts:
            self.add_facts_batch(user_id, facts)
            logger.info(f"L3 抽取并写入 {len(facts)} 条事实")
        if profile:
            self.update_profile(user_id, profile)
