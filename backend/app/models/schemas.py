"""Pydantic request/response schemas for the API layer."""
from __future__ import annotations
from pydantic import BaseModel, Field


# ── Ingestion ──────────────────────────────────────────────
class IngestGitHubRequest(BaseModel):
    github_url: str = Field(..., description="Public GitHub repo URL")


class IngestResponse(BaseModel):
    repo_id: str
    repo_name: str
    total_files: int
    total_chunks: int
    languages: list[str]
    status: str = "indexed"


# ── Query ──────────────────────────────────────────────────
class QueryRequest(BaseModel):
    repo_id: str
    question: str
    conversation_id: str | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    file_path: str
    name: str
    chunk_type: str
    start_line: int
    end_line: int
    content: str
    score: float
    language: str


class RetrievalInfo(BaseModel):
    original_query: str
    rewritten_query: str
    intent: str
    bm25_count: int
    vector_count: int
    reranked_count: int
    latency_ms: float
    retrieved_chunks: list[RetrievedChunk]


class QueryResponse(BaseModel):
    answer: str
    retrieval_info: RetrievalInfo
    conversation_id: str


# ── Repository ─────────────────────────────────────────────
class FileNode(BaseModel):
    path: str
    name: str
    is_dir: bool
    children: list[FileNode] = []
    language: str | None = None
    chunk_count: int = 0


class GraphNode(BaseModel):
    id: str
    label: str
    type: str  # "file" | "class" | "function"
    file_path: str


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str  # "imports" | "calls" | "inherits" | "contains"


class GraphData(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class RepoInfo(BaseModel):
    repo_id: str
    repo_name: str
    total_files: int
    total_chunks: int
    languages: dict[str, int]
    file_tree: FileNode


# Resolve forward references
FileNode.model_rebuild()
