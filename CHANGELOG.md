# Changelog

所有版本变更记录。

## [0.1.0] - 2026-09-22

### ✨ 新增

- 三层记忆架构(L1 Redis + L2 PostgreSQL + L3 Milvus)
- Agentic RAG 路由决策(Router → Rewriter → Retriever → Verifier → Generate)
- BM25 + 向量 + Re-ranking 混合检索
- Self-RAG 答案自检机制
- LangGraph 完整编排
- FastAPI 入口(`/chat`、`/health`、`/memory/*`)
- LLM Client 统一封装(OpenAI 兼容,支持 DeepSeek/Qwen/通义千问)
- 评估脚本(Ragas 风格 + 简化版)
- Docker Compose 一键启动
- 完整文档(架构、API、评估、部署)

### 🔧 配置

- 环境变量配置(.env.example)
- 可插拔 LLM Provider
- PostgreSQL pgvector 双能力

### 📚 文档

- README.md(中英双版)
- docs/architecture.md
- docs/api_reference.md
- docs/evaluation.md
- docs/deployment.md

## [Unreleased]

### 计划

- [ ] GraphRAG 集成
- [ ] 流式输出(SSE)
- [ ] 多 Agent 协作(检索员+分析师+客服)
- [ ] Web UI(Streamlit / Gradio)
- [ ] Docker Registry 镜像
- [ ] K8s Helm Chart
- [ ] 完整 Ragas 集成
- [ ] LoRA 微调 hook
