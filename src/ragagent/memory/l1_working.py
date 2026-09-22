"""
L1 工作记忆 (Working Memory)
- 最近 N 轮对话原文
- 存储在 Redis 中,LIST 结构,毫秒级读写
- 自动 LRU 滚动,会话级 TTL
"""
from typing import List, Dict, Optional
import json
import redis

from ..config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class L1WorkingMemory:
    """基于 Redis 的工作记忆,管当前会话上下文"""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client or redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            decode_responses=True,
        )
        self.max_messages = settings.l1_max_messages

    def get_recent(self, user_id: str, n: Optional[int] = None) -> List[Dict]:
        """获取最近 N 条消息"""
        n = n or self.max_messages
        key = f"l1:chat:{user_id}"
        try:
            raw_list = self.redis.lrange(key, -n, -1)
        except redis.ConnectionError as e:
            logger.warning(f"Redis 连接失败,返回空: {e}")
            return []

        messages = []
        for raw in raw_list:
            try:
                messages.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return messages

    def add(self, user_id: str, role: str, content: str):
        """追加一条消息(角色+内容),自动 LRU 滚动"""
        key = f"l1:chat:{user_id}"
        msg = json.dumps(
            {"role": role, "content": content, "ts": _now_ts()},
            ensure_ascii=False,
        )
        try:
            self.redis.rpush(key, msg)
            # 只保留 max_messages * 2,避免无限增长
            self.redis.ltrim(key, -(self.max_messages * 2), -1)
            # 24h TTL
            self.redis.expire(key, 86400)
        except redis.ConnectionError as e:
            logger.warning(f"Redis 写入失败: {e}")

    def clear(self, user_id: str):
        """清空某个用户的 L1"""
        self.redis.delete(f"l1:chat:{user_id}")

    def count(self, user_id: str) -> int:
        return self.redis.llen(f"l1:chat:{user_id}")


def _now_ts() -> int:
    import time
    return int(time.time())
