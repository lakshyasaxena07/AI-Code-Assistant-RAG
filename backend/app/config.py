from pydantic_settings import BaseSettings
from pathlib import Path
import json


class Settings(BaseSettings):
    # LLM
    llm_provider: str = "groq"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"

    # Embedding & Reranking
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    embedding_device: str = "cpu"  # "cuda" if GPU available

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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        # If .env is missing (e.g. on HF Spaces), just use OS env vars
        env_ignore_empty = True

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
