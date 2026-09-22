"""
测试混合检索器
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestHybridRetriever:
    """混合检索测试"""

    @patch("ragagent.retriever.hybrid.L1WorkingMemory")
    @patch("ragagent.retriever.hybrid.L2EpisodicMemory")
    @patch("ragagent.retriever.hybrid.L3LongTermMemory")
    @patch("ragagent.retriever.hybrid.BGEReranker")
    def test_basic_retrieve(
        self, mock_reranker, mock_l3, mock_l2, mock_l1_cls
    ):
        """基础检索流程"""
        from ragagent.retriever import HybridRetriever

        # Mock L1 返回
        mock_l1_instance = MagicMock()
        mock_l1_instance.get_recent.return_value = [
            {"role": "user", "content": "我喜欢美式咖啡"},
            {"role": "assistant", "content": "好的,已记录"},
        ]
        mock_l1_cls.return_value = mock_l1_instance

        # Mock L2
        mock_l2_instance = MagicMock()
        mock_l2_instance.recall.return_value = [
            {
                "summary": "咖啡偏好:美式不加糖",
                "facts": ["美式不加糖"],
                "score": 0.85,
                "source": "L2",
            }
        ]
        mock_l2.return_value = mock_l2_instance

        # Mock L3
        mock_l3_instance = MagicMock()
        mock_l3_instance.recall.return_value = [
            {
                "content": "美式咖啡不加糖是用户偏好",
                "score": 0.92,
                "source": "L3",
            }
        ]
        mock_l3_instance.get_profile.return_value = {"name": "test_user"}
        mock_l3.return_value = mock_l3_instance

        # Mock Reranker
        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank.return_value = [
            {"content": "test", "score": 0.9, "source": "L3"}
        ]
        mock_reranker.return_value = mock_reranker_instance

        # 执行
        retriever = HybridRetriever()
        result = retriever.retrieve(user_id="u1", query="咖啡", top_k=3)

        assert "l1" in result
        assert "retrieved" in result
        assert "all_candidates" in result
        assert len(result["l1"]) == 2

    def test_dedupe(self):
        """去重逻辑"""
        from ragagent.retriever.hybrid import HybridRetriever
        cands = [
            {"content": "我是喜欢美式"},
            {"content": "我是喜欢美式"},  # 重复
            {"content": "另一段内容"},
        ]
        out = HybridRetriever._dedupe(cands)
        assert len(out) == 2


class TestBGEReranker:
    """Reranker 测试(用 mock 替换模型加载)"""

    def test_rerank_returns_sorted(self):
        from unittest.mock import MagicMock
        from ragagent.retriever.reranker import BGEReranker

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.9, 0.7]

        reranker = BGEReranker(model=mock_model)
        result = reranker.rerank(
            query="q",
            candidates=[
                {"content": "a"},
                {"content": "b"},
                {"content": "c"},
            ],
            top_k=2,
        )

        # 应该有 2 个结果,按分数倒序
        assert len(result) == 2
        assert result[0]["final_score"] >= result[1]["final_score"]
