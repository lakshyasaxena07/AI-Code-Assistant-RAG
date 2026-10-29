from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import json


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    # LLM
    llm_provider: str = "gemini"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "gemini_api_key", "GEMINI_API_KEY", "GOOGLE_API_KEY", "google_api_key"
        ),
    )
    groq_api_key: str = ""
    llm_model: str = "gemini-1.5-flash"
    llm_connect_timeout: float = 15.0
    llm_read_timeout: float = 60.0
    llm_max_retries: int = 3
    llm_retry_delay: float = 1.0
    llm_retry_jitter: float = 0.5

    # Embedding & Reranking
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    embedding_device: str = "cpu"  # "cuda" if GPU available
    enable_reranker: bool = True
    rerank_max_candidates: int = 10

    # Storage paths
    chroma_persist_dir: str = "./data/chroma"
    repos_dir: str = "./data/repos"
    graph_dir: str = "./data/graphs"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = '["http://localhost:3000"]'

    # Retrieval
    hybrid_alpha: float = 0.5  # 0 = pure BM25, 1 = pure vector
    retrieval_top_k: int = 15
    rerank_top_k: int = 8
    graph_expansion_hops: int = 1
    max_context_tokens: int = 6000

    # Chunking
    max_chunk_tokens: int = 512
    min_chunk_tokens: int = 30

    @property
    def cors_origins_list(self) -> list[str]:
        return json.loads(self.cors_origins)

    @property
    def chroma_path(self) -> Path:
        p = Path(self.chroma_persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def repos_path(self) -> Path:
        p = Path(self.repos_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def graph_path(self) -> Path:
        p = Path(self.graph_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
