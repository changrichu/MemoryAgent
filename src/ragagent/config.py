"""
MemoryAgent 配置管理
"""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置,从 .env 文件加载"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ----- LLM -----
    openai_api_key: str = "sk-empty"
    openai_api_base: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"

    # ----- Embedding -----
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_device: str = "cpu"

    # ----- Reranker -----
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"

    # ----- Redis -----
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

    # ----- PostgreSQL -----
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "memoryagent"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres123"

    # ----- Milvus -----
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_user: str = "root"
    milvus_password: str = "Milvus"

    # ----- Neo4j -----
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j123"

    # ----- 业务参数 -----
    l1_max_messages: int = 10
    l2_summary_trigger: int = 10
    l3_top_k: int = 20
    final_top_k: int = 5
    retrieval_score_threshold: float = 0.6

    # ----- Langfuse -----
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "http://localhost:3000"

    # ----- API -----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    @property
    def redis_url(self) -> str:
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
