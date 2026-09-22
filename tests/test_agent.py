"""
Agentic RAG 主流程测试
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestAgenticRAGAgent:
    """Agent 端到端测试(大量 mock)"""

    @patch("ragagent.agent.agent_graph.HybridRetriever")
    @patch("ragagent.agent.agent_graph.L3LongTermMemory")
    @patch("ragagent.agent.agent_graph.L2EpisodicMemory")
    @patch("ragagent.agent.agent_graph.L1WorkingMemory")
    @patch("ragagent.agent.router.get_llm_client")
    @patch("ragagent.agent.rewriter.get_llm_client")
    def test_full_flow(
        self,
        mock_rewriter_llm,
        mock_router_llm,
        mock_l1_cls,
        mock_l2_cls,
        mock_l3_cls,
        mock_retriever_cls,
    ):
        """完整对话流程"""
        from ragagent.agent import AgenticRAGAgent

        # Mock LLM
        def make_mock_llm(decision_resp):
            m = MagicMock()
            m.chat_json.side_effect = [
                decision_resp,  # router 决策
                {  # rewriter 改写
                    "rewritten_query": "改写后的问题",
                    "sub_queries": ["改写后的问题"],
                },
                {  # verifier
                    "is_acceptable": True,
                    "faithfulness_score": 0.92,
                },
            ]
            m.chat.return_value = "这是答案"
            return m

        mock_llm = make_mock_llm({
            "decision": "retrieve_long",
            "reason": "测试",
            "sub_questions": [],
        })
        mock_router_llm.return_value = mock_llm
        mock_rewriter_llm.return_value = mock_llm

        # Mock memories
        for cls in [mock_l1_cls, mock_l2_cls, mock_l3_cls]:
            instance = MagicMock()
            instance.get_recent.return_value = []
            instance.recall.return_value = []
            instance.get_profile.return_value = {}
            cls.return_value = instance

        # Mock retriever
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.retrieve.return_value = {
            "l1": [],
            "retrieved": [
                {"content": "证据1", "score": 0.9, "source": "L3"}
            ],
            "all_candidates": [],
        }
        mock_retriever_instance.health_check.return_value = {
            "redis": "ok", "postgres": "ok", "milvus": "ok"
        }
        mock_retriever_cls.return_value = mock_retriever_instance

        # 运行
        agent = AgenticRAGAgent()
        result = agent.chat(
            user_id="test_user",
            query="我之前提到过什么?",
        )

        assert "final_answer" in result
        # 因为 mock 答了 "这是答案"
        assert result["final_answer"] is not None

    def test_router_decides(self):
        """Router 决策逻辑"""
        from ragagent.agent.router import Router

        # 直接测试 needs_retrieval 静态方法
        assert Router.needs_retrieval("retrieve_short") is True
        assert Router.needs_retrieval("retrieve_long") is True
        assert Router.needs_retrieval("retrieve_all") is True
        assert Router.needs_retrieval("direct") is False

    def test_rewriter_recursive_check(self):
        """Rewriter 会调用 L3 hints"""
        from unittest.mock import MagicMock, patch
        from ragagent.agent.rewriter import Rewriter

        with patch("ragagent.agent.rewriter.get_llm_client") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.chat_json.return_value = {
                "rewritten_query": "改写的查询",
                "sub_queries": ["改写的查询"],
            }
            mock_get_llm.return_value = mock_llm

            with patch("ragagent.agent.rewriter.L1WorkingMemory") as mock_l1:
                with patch("ragagent.agent.rewriter.L3LongTermMemory") as mock_l3:
                    mock_l3.return_value.recall.return_value = []

                    r = Rewriter()
                    result = r.rewrite("u1", "那我呢?", [])
                    assert "rewritten_query" in result
                    # L3 被用来拉取 hints
                    mock_l3.return_value.recall.assert_called()
