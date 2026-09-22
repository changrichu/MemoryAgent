"""
FastAPI 入口 - 提供 HTTP 接口
"""
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import settings
from .agent import AgenticRAGAgent
from .llm_client import get_llm_client
from .utils.logger import get_logger

logger = get_logger(__name__)

# ============ 全局 Agent 单例 ============
_agent: Optional[AgenticRAGAgent] = None


def get_agent() -> AgenticRAGAgent:
    global _agent
    if _agent is None:
        _agent = AgenticRAGAgent()
    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化"""
    logger.info("MemoryAgent 启动中...")
    try:
        # 预热 LLM client
        get_llm_client()
        logger.info("LLM Client 就绪")
    except Exception as e:
        logger.warning(f"启动警告: {e}")
    yield
    logger.info("MemoryAgent 关闭")


app = FastAPI(
    title="MemoryAgent",
    description="基于多级记忆机制的 Agentic RAG 对话系统",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Schema ============

class ChatRequest(BaseModel):
    user_id: str = Field(..., description="用户唯一标识")
    query: str = Field(..., description="用户问题")


class ChatResponse(BaseModel):
    user_id: str
    query: str
    answer: str
    sources: List[str] = []
    routing: Optional[Dict[str, Any]] = None
    rewritten_query: Optional[str] = None
    logs: List[str] = []


class HealthResponse(BaseModel):
    status: str
    components: Dict[str, str]


# ============ 路由 ============

@app.get("/")
async def root():
    return {
        "name": "MemoryAgent",
        "version": "0.1.0",
        "description": "基于多级记忆机制的 Agentic RAG 对话系统",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    agent = get_agent()
    components = agent.retriever.health_check()
    overall = "ok" if all(
        v.startswith("ok") for v in components.values()
    ) else "degraded"
    return HealthResponse(status=overall, components=components)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """主对话接口"""
    try:
        agent = get_agent()
        result = agent.chat(user_id=req.user_id, query=req.query)
        return ChatResponse(
            user_id=req.user_id,
            query=req.query,
            answer=result["final_answer"],
            sources=result.get("sources", []),
            routing=result.get("routing"),
            rewritten_query=result.get("rewritten", {}).get("rewritten_query"),
            logs=result.get("logs", []),
        )
    except Exception as e:
        logger.error(f"/chat 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/clear/{user_id}")
async def clear_l1(user_id: str):
    """清空指定用户的工作记忆(L1)"""
    agent = get_agent()
    agent.l1.clear(user_id)
    return {"ok": True, "user_id": user_id, "cleared": "L1"}


@app.get("/memory/profile/{user_id}")
async def get_profile(user_id: str):
    """获取用户画像"""
    agent = get_agent()
    profile = agent.l3.get_profile(user_id)
    return {"user_id": user_id, "profile": profile}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "ragagent.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
