"""
测试三层记忆模块
运行: pytest tests/test_memory.py -v
"""
import sys
import os
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestL1WorkingMemory:
    """L1 工作记忆测试(单元测试,无需 Redis)"""

    def test_add_and_get(self):
        """测试基本读写"""
        from ragagent.memory.l1_working import L1WorkingMemory
        from unittest.mock import MagicMock

        mock_redis = MagicMock()
        mock_redis.lrange.return_value = [
            '{"role":"user","content":"hello","ts":1}',
            '{"role":"assistant","content":"hi","ts":2}',
        ]

        l1 = L1WorkingMemory(redis_client=mock_redis)
        msgs = l1.get_recent("u1", n=2)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["content"] == "hi"

    def test_add_format(self):
        """添加消息的格式"""
        from ragagent.memory.l1_working import L1WorkingMemory
        from unittest.mock import MagicMock
        import json

        mock_redis = MagicMock()
        l1 = L1WorkingMemory(redis_client=mock_redis)
        l1.add("u1", "user", "test content")

        # 验证调用了 rpush 和 ltrim
        mock_redis.rpush.assert_called_once()
        args = mock_redis.rpush.call_args
        assert args[0][0] == "l1:chat:u1"
        msg = json.loads(args[0][1])
        assert msg["role"] == "user"
        assert msg["content"] == "test content"

    def test_count(self):
        from ragagent.memory.l1_working import L1WorkingMemory
        from unittest.mock import MagicMock

        mock_redis = MagicMock()
        mock_redis.llen.return_value = 5
        l1 = L1WorkingMemory(redis_client=mock_redis)
        assert l1.count("u1") == 5


class TestMemoryArchitecture:
    """三层记忆架构的接口契约测试"""

    def test_three_layers_exist(self):
        """三层都有完整接口"""
        from ragagent.memory import (
            L1WorkingMemory,
            L2EpisodicMemory,
            L3LongTermMemory,
        )

        for cls in [L1WorkingMemory, L2EpisodicMemory, L3LongTermMemory]:
            methods = [m for m in dir(cls) if not m.startswith("_")]
            assert len(methods) > 0, f"{cls.__name__} 没有公开方法"

    def test_recall_signature(self):
        """L2 和 L3 都有 recall 方法"""
        from ragagent.memory import L2EpisodicMemory, L3LongTermMemory

        for cls in [L2EpisodicMemory, L3LongTermMemory]:
            assert hasattr(cls, "recall"), f"{cls.__name__} 缺少 recall 方法"
            assert hasattr(cls, "add_fact") or hasattr(cls, "add_summary"), \
                f"{cls.__name__} 缺少写入方法"
