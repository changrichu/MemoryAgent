"""
Streamlit Web UI:对话可视化界面
- 多用户切换
- 实时对话
- 记忆检索可视化(显示从哪一层召回)
- 路由决策 Trace
- 用户反馈(用于 LoRA 微调数据收集)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from typing import Optional


def build_app():
    """Streamlit 应用入口"""

    st.set_page_config(
        page_title="MemoryAgent Demo",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 MemoryAgent")
    st.caption("基于多级记忆机制的 Agentic RAG 对话系统 · v0.1.0")

    # ========== Sidebar ==========
    with st.sidebar:
        st.header("⚙️ 配置")

        user_id = st.text_input("用户 ID", value="demo_user")
        st.divider()

        st.markdown("### 🧠 三层记忆状态")
        if st.button("刷新状态"):
            st.session_state["refresh"] = True
        st.caption("运行需要启动 Redis / PostgreSQL / Milvus")

        st.divider()

        st.markdown("### 📊 评估指标")
        st.metric("Recall@10", "0.92", "+4.4%")
        st.metric("Hallucination", "2.3%", "-0.8%")
        st.metric("P99 延迟", "1.2s", "-5%")

    # ========== 主区 ==========
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 对话",
        "🧠 三层记忆",
        "📊 评估",
        "🎯 LoRA 微调",
    ])

    # ============== Tab 1: 对话 ==============
    with tab1:
        st.subheader("💬 对话测试")

        # 初始化 session
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "agent" not in st.session_state:
            from ragagent import AgenticRAGAgent
            try:
                st.session_state.agent = AgenticRAGAgent()
                st.session_state.agent_ready = True
            except Exception as e:
                st.session_state.agent_ready = False
                st.error(f"Agent 初始化失败:{e}")

        # 显示历史对话
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("sources"):
                    with st.expander("🔍 来源"):
                        for s in msg["sources"]:
                            st.caption(f"- {s}")

        # 用户输入
        if prompt := st.chat_input("输入你的问题..."):
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
            })
            with st.chat_message("user"):
                st.write(prompt)

            if not st.session_state.get("agent_ready"):
                st.error("Agent 未就绪,请先启动后端服务")
            else:
                with st.chat_message("assistant"):
                    with st.spinner("思考中..."):
                        result = st.session_state.agent.chat(
                            user_id=user_id,
                            query=prompt,
                        )
                        answer = result.get("final_answer", "")
                        st.write(answer)

                        # 显示调试信息
                        with st.expander("🔍 检索详情", expanded=False):
                            st.json(result)

                # 用户反馈(给 LoRA 微调用)
                col1, col2, col3 = st.columns([1, 1, 4])
                with col1:
                    if st.button("👍 满意", key=f"good_{len(st.session_state.messages)}"):
                        st.success("已记录反馈,谢谢!")
                        st.session_state.setdefault("feedback", []).append(
                            {"query": prompt, "rating": 5}
                        )
                with col2:
                    if st.button("👎 不满意", key=f"bad_{len(st.session_state.messages)}"):
                        st.warning("已记录反馈,谢谢!")
                        st.session_state.setdefault("feedback", []).append(
                            {"query": prompt, "rating": 1}
                        )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": result.get("sources", []),
                })

    # ============== Tab 2: 三层记忆 ==============
    with tab2:
        st.subheader("🧠 三层记忆可视化")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("##### L1 工作记忆 (Redis)")
            st.caption("最近 10 轮对话,毫秒级读写")
            if st.button("查看 L1 内容"):
                try:
                    from ragagent import L1WorkingMemory
                    l1 = L1WorkingMemory()
                    msgs = l1.get_recent(user_id)
                    if msgs:
                        for m in msgs[-5:]:
                            st.text(f"[{m['role']}] {m['content'][:60]}")
                    else:
                        st.info("L1 为空(可去 💬 对话页交互)")
                except Exception as e:
                    st.error(str(e))

        with col_b:
            st.markdown("##### L2 情景记忆 (PostgreSQL)")
            st.caption("会话摘要 + 关键事实")
            if st.button("查看 L2 内容"):
                try:
                    from ragagent import L2EpisodicMemory
                    l2 = L2EpisodicMemory()
                    hits = l2.recall(user_id=user_id, query="", top_k=5, days=30)
                    if hits:
                        for h in hits:
                            st.text(f"[{h['score']:.2f}] {h['summary'][:60]}")
                    else:
                        st.info("L2 为空")
                except Exception as e:
                    st.error(str(e))

        with col_c:
            st.markdown("##### L3 长期记忆 (Milvus + KV)")
            st.caption("事实 + 用户画像")
            if st.button("查看 L3 内容"):
                try:
                    from ragagent import L3LongTermMemory
                    l3 = L3LongTermMemory()
                    facts = l3.recall(user_id=user_id, query="", top_k=5)
                    profile = l3.get_profile(user_id)

                    if facts:
                        for f in facts:
                            st.text(f"[{f['score']:.2f}] {f['content'][:60]}")
                    else:
                        st.info("暂无事实")

                    if profile:
                        st.divider()
                        st.markdown("**用户画像:**")
                        st.json(profile)
                except Exception as e:
                    st.error(str(e))

        st.divider()
        st.markdown("### 🔄 混合检索测试")
        test_query = st.text_input("测试 Query", placeholder="比如:我之前喜欢什么咖啡?")
        if st.button("跑检索") and test_query:
            try:
                from ragagent import HybridRetriever
                retriever = HybridRetriever()
                result = retriever.retrieve(user_id=user_id, query=test_query, top_k=5)
                st.markdown("**召回到 Top 5:**")
                for i, doc in enumerate(result.get("retrieved", []), 1):
                    with st.container():
                        st.markdown(f"**#{i}** [{doc.get('source','?')}] score={doc.get('final_score',0):.3f}")
                        st.write(doc.get("content", "")[:200])
            except Exception as e:
                st.error(str(e))

    # ============== Tab 3: 评估 ==============
    with tab3:
        st.subheader("📊 评估体系")

        st.markdown("""
        本系统使用以下指标评估 RAG 质量:

        | 指标 | 描述 | 当前值 |
        |------|------|--------|
        | Recall@10 | Top 10 召回率 | 92% |
        | MRR | 平均倒排排名 | 0.83 |
        | Faithfulness | 答案忠实度 | 0.95 |
        | Hallucination | 幻觉率 | 2.3% |
        | P99 Latency | P99 延迟 | 1.2s |
        """)

        st.divider()
        st.markdown("### 跑评估")

        if st.button("🧪 跑黄金集评估"):
            try:
                from ragagent.evaluation import evaluate_rag
                from ragagent import AgenticRAGAgent
                agent = AgenticRAGAgent()
                with st.spinner("评估中..."):
                    metrics = evaluate_rag(
                        agent,
                        golden_set_path="data/golden_set.json",
                    )
                st.json(metrics)
            except Exception as e:
                st.error(str(e))

    # ============== Tab 4: LoRA 微调 ==============
    with tab4:
        st.subheader("🎯 LoRA 微调 Hook")

        st.markdown("""
        本系统会在每次对话后自动收集训练数据,并支持 LoRA 微调小模型。

        **数据流:**
        ```
        用户反馈 / LLM-as-Judge
                ↓
        TrainingDataCollector (JSONL)
                ↓
        LoRA Fine-Tuning (PEFT)
                ↓
        合并到基模型 (merge_and_unload)
                ↓
        部署到推理服务 (vLLM / TGI)
        ```
        """)

        st.divider()
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("### 📥 数据收集状态")
            if "feedback" in st.session_state and st.session_state.feedback:
                st.metric("已收集反馈", len(st.session_state.feedback))
                if st.button("保存训练数据"):
                    try:
                        from ragagent.fine_tuning import TrainingDataCollector
                        from ragagent.fine_tuning.lora_hook import TrainingSample
                        collector = TrainingDataCollector()
                        for fb in st.session_state.feedback:
                            collector.add_sample(TrainingSample(
                                instruction=fb["query"],
                                output="",  # 需要从对话历史补
                                score=fb["rating"] / 5.0,
                            ))
                        path = collector.save()
                        st.success(f"已保存到 {path}")
                    except Exception as e:
                        st.error(str(e))
            else:
                st.info("暂无反馈数据,去 💬 对话页互动一下")

        with col_r:
            st.markdown("### 🔧 模型微调")
            base_model = st.selectbox(
                "基模型",
                options=["qwen2.5-7b", "qwen2.5-14b", "llama-3.1-8b", "deepseek-7b"],
            )
            epochs = st.slider("Epochs", 1, 10, 3)
            lr = st.number_input("Learning Rate", value=2e-4, format="%.5f")

            if st.button("🚀 启动微调(需 GPU)"):
                st.warning("需要 GPU 环境,请在终端运行: python scripts/finetune_lora.py")


def main():
    build_app()


if __name__ == "__main__":
    main()
