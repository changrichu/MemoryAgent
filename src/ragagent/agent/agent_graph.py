"""
Agentic RAG 主控图
- LangGraph 编排 Router → Rewriter → Retriever → Verifier → Generator
- 支持检索质量自检 + 不合格时循环重检
"""
from typing import Dict, List, Optional, TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END

from .router import Router
from .rewriter import Rewriter
from ..retriever import HybridRetriever, BGEReranker
from ..memory import L1WorkingMemory, L2EpisodicMemory, L3LongTermMemory
from ..llm_client import get_llm_client
from ..prompts import ANSWER_PROMPT, VERIFY_PROMPT
from ..config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AgentState(TypedDict):
    """Agent 状态(在 LangGraph 节点间传递)"""
    user_id: str
    query: str
    history: List[Dict]
    routing: Dict
    rewritten: Dict
    retrieved: Dict
    attempts: int
    evidence: List[Dict]
    answer: str
    final_answer: str
    sources: List[str]
    # 处理 accumulated
    logs: Annotated[List[str], operator.add]


class AgenticRAGAgent:
    """端到端 Agentic RAG 入口"""

    def __init__(self):
        self.router = Router()
        self.rewriter = Rewriter()
        self.retriever = HybridRetriever(
            l1=L1WorkingMemory(),
            l2=L2EpisodicMemory(),
            l3=L3LongTermMemory(),
            reranker=BGEReranker(),
        )
        self.llm = get_llm_client()
        self.l1 = self.retriever.l1
        self.l2 = self.retriever.l2
        self.l3 = self.retriever.l3
        self.graph = self._build_graph()

    # ========== 节点实现 ==========

    def _node_router(self, state: AgentState) -> AgentState:
        state["routing"] = self.router.decide(state["query"], state.get("history"))
        state["logs"] = [f"📍 Router: {state['routing'].get('decision')}"]
        return state

    def _node_direct_answer(self, state: AgentState) -> AgentState:
        """不走 RAG,直接 LLM 答"""
        history_text = self._format_history(state.get("history"))
        profile = self.l3.get_profile(state["user_id"])

        prompt = ANSWER_PROMPT.format(
            profile=self._format_profile(profile),
            retrieved_context="(本次未触发检索)",
            recent_dialog=history_text,
            query=state["query"],
        )
        answer = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.7)
        state["answer"] = answer
        state["final_answer"] = answer
        state["evidence"] = []
        state["sources"] = []
        state["logs"] = ["📍 直接回答"]
        return state

    def _node_rewriter(self, state: AgentState) -> AgentState:
        rewritten = self.rewriter.rewrite(
            user_id=state["user_id"],
            query=state["query"],
            history=state.get("history"),
        )
        state["rewritten"] = rewritten
        state["logs"] = [f"📍 Rewriter: {rewritten['rewritten_query'][:60]}"]
        return state

    def _node_retriever(self, state: AgentState) -> AgentState:
        rewritten_query = state["rewritten"]["rewritten_query"]
        result = self.retriever.retrieve(
            user_id=state["user_id"],
            query=rewritten_query,
            top_k=settings.final_top_k,
            l3_candidates=settings.l3_top_k,
        )
        # 把 L1 直接拼到 evidence
        evidence = list(result.get("retrieved", []))
        state["evidence"] = evidence
        state["retrieved"] = result
        state["attempts"] = state.get("attempts", 0) + 1
        state["logs"] = [f"📍 Retriever: 召回到 {len(evidence)} 条"]
        return state

    def _node_verifier(self, state: AgentState) -> AgentState:
        """Self-RAG 自检,不通过就回到 rewriter"""
        if state["attempts"] >= 3:
            state["logs"] = ["📍 达到最大重试次数,跳过验证"]
            return state

        evidence_text = "\n".join(
            f"[{e.get('source','?')}] {e.get('content','')[:200]}"
            for e in state.get("evidence", [])
        )

        prompt = VERIFY_PROMPT.format(
            query=state["query"],
            evidence=evidence_text or "(空)",
            draft="(待生成)",
        )
        result = self.llm.chat_json(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        acceptable = result.get("is_acceptable", True)
        score = float(result.get("faithfulness_score", 1.0))

        if acceptable and score >= 0.7:
            state["logs"] = [f"📍 Verifier: 通过 (faith={score:.2f})"]
        else:
            state["rewritten"]["rewritten_query"] = (
                state["query"] + " (更详细、更具体的事实)"
            )
            state["logs"] = [f"📍 Verifier: 重检 (faith={score:.2f})"]

        return state

    def _node_generate(self, state: AgentState) -> AgentState:
        profile = self.l3.get_profile(state["user_id"])
        retrieved_context = "\n".join(
            f"[{e.get('source','?')}|score={e.get('final_score',0):.2f}] "
            f"{e.get('content','')[:300]}"
            for e in state.get("evidence", [])
        )

        history_text = self._format_history(state.get("history"))

        prompt = ANSWER_PROMPT.format(
            profile=self._format_profile(profile),
            retrieved_context=retrieved_context or "(未检索到相关记忆)",
            recent_dialog=history_text,
            query=state["query"],
        )
        answer = self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        state["answer"] = answer
        state["final_answer"] = answer
        state["sources"] = [
            f"{e.get('source')}:{e.get('content','')[:80]}"
            for e in state.get("evidence", [])
        ]
        state["logs"] = ["📍 Generate: 完成生成"]
        return state

    def _node_memorize(self, state: AgentState) -> AgentState:
        """对话后写入 L1(短期)+ 触发 L2 摘要 + 写 L3 事实"""
        user_id = state["user_id"]
        user_msg = state["query"]
        ai_msg = state["final_answer"]

        # L1 工作记忆
        self.l1.add(user_id, "user", user_msg)
        self.l1.add(user_id, "assistant", ai_msg)

        # L2 摘要(消息数达阈值时)
        all_msgs = self.l1.get_recent(user_id, n=50)
        if len(all_msgs) >= settings.l2_summary_trigger:
            self.l2.maybe_summarize(user_id, all_msgs)

        # L3 事实抽取
        try:
            self.l3.extract_and_store(
                user_id,
                f"用户: {user_msg}\n助手: {ai_msg}",
            )
        except Exception as e:
            logger.warning(f"L3 写入失败: {e}")

        state["logs"] = ["📍 Memorize: 已写入三层记忆"]
        return state

    # ========== 编排 ==========

    def _should_retrieve(self, state: AgentState) -> str:
        decision = state.get("routing", {}).get("decision", "retrieve_short")
        if Router.needs_retrieval(decision):
            return "retriever"
        return "direct"

    def _should_retry(self, state: AgentState) -> str:
        last_log = state.get("logs", [])[-1] if state.get("logs") else ""
        if "重检" in last_log and state.get("attempts", 0) < 3:
            return "retry"
        return "generate"

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)

        graph.add_node("router", self._node_router)
        graph.add_node("direct", self._node_direct_answer)
        graph.add_node("rewriter", self._node_rewriter)
        graph.add_node("retriever", self._node_retriever)
        graph.add_node("verifier", self._node_verifier)
        graph.add_node("generate", self._node_generate)
        graph.add_node("memorize", self._node_memorize)

        graph.set_entry_point("router")
        graph.add_conditional_edges(
            "router", self._should_retrieve,
            {"retriever": "rewriter", "direct": "direct"},
        )
        graph.add_edge("rewriter", "retriever")
        graph.add_edge("retriever", "verifier")
        graph.add_conditional_edges(
            "verifier", self._should_retry,
            {"retry": "rewriter", "generate": "generate"},
        )
        graph.add_edge("generate", "memorize")
        graph.add_edge("direct", "memorize")
        graph.add_edge("memorize", END)

        return graph.compile()

    # ========== 入口 ==========

    def chat(self, user_id: str, query: str, history: List[Dict] = None) -> Dict:
        """与用户对话一轮"""
        # 注入历史(从 L1 自动加载)
        if history is None:
            raw = self.l1.get_recent(user_id, n=settings.l1_max_messages)
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in raw
            ]

        initial_state: AgentState = {
            "user_id": user_id,
            "query": query,
            "history": history,
            "routing": {},
            "rewritten": {},
            "retrieved": {},
            "attempts": 0,
            "evidence": [],
            "answer": "",
            "final_answer": "",
            "sources": [],
            "logs": [],
        }

        try:
            final_state = self.graph.invoke(initial_state)
        except Exception as e:
            logger.error(f"图执行失败: {e}")
            return {"final_answer": f"服务异常: {e}", "sources": [], "logs": [str(e)]}

        return {
            "user_id": user_id,
            "query": query,
            "final_answer": final_state["final_answer"],
            "sources": final_state["sources"],
            "routing": final_state["routing"],
            "rewritten": final_state["rewritten"],
            "logs": final_state["logs"],
        }

    # ========== 工具函数 ==========

    @staticmethod
    def _format_history(history: List[Dict]) -> str:
        if not history:
            return "(无)"
        return "\n".join(
            f"[{m.get('role','user')}] {m.get('content','')}"
            for m in history[-10:]
        )

    @staticmethod
    def _format_profile(profile: Dict) -> str:
        if not profile:
            return "(暂无画像)"
        return "\n".join(f"- {k}: {v}" for k, v in profile.items())
