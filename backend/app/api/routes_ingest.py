"""Ingestion API routes — handles repository upload and indexing."""
from __future__ import annotations
import time

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.models.schemas import IngestGitHubRequest, IngestResponse
from app.ingestion.upload import clone_github_repo, save_upload, collect_files
from app.ingestion.chunker import chunk_repository
from app.ingestion.graph_builder import build_code_graph, save_graph
from app.ingestion.indexer import index_chunks
import json
from pathlib import Path

from app.config import settings
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/api/ingest", tags=["ingestion"])

REGISTRY_FILE = Path(settings.repos_dir) / "registry.json"

def _load_registry() -> dict[str, dict]:
    if REGISTRY_FILE.exists():
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Failed to load registry: {e}")
    return {}

def _save_registry():
    try:
        with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(_repo_registry, f, indent=2)
    except Exception as e:
        log.error(f"Failed to save registry: {e}")

# In-memory repo registry (maps repo_id → metadata)
_repo_registry: dict[str, dict] = _load_registry()


def get_repo_registry() -> dict[str, dict]:
    return _repo_registry


@router.post("/upload", response_model=IngestResponse)
async def upload_repo(file: UploadFile = File(...)):
    """Upload a ZIP file containing a code repository."""
    if not file.filename or not file.filename.endswith(".zip"):
        raise HTTPException(400, "Only ZIP files are accepted")

    start = time.time()
    log.info(f"Received upload: {file.filename}")

    content = await file.read()
    repo_id, repo_path = await save_upload(content, file.filename)

    return await _process_repo(repo_id, repo_path, file.filename.replace(".zip", ""), start)


@router.post("/github", response_model=IngestResponse)
async def ingest_github(request: IngestGitHubRequest):
    """Clone and ingest a public GitHub repository."""
    start = time.time()
    log.info(f"Cloning GitHub repo: {request.github_url}")

    try:
        repo_id, repo_path = clone_github_repo(request.github_url)
    except Exception as e:
        raise HTTPException(400, f"Failed to clone repository: {e}")

    repo_name = request.github_url.rstrip("/").split("/")[-1].replace(".git", "")
    return await _process_repo(repo_id, repo_path, repo_name, start)


@router.get("/repos")
async def list_repos():
    """List all indexed repositories."""
    return {"repos": list(_repo_registry.values())}


async def _process_repo(repo_id: str, repo_path, repo_name: str, start_time: float) -> IngestResponse:
    """Common processing pipeline for uploaded/cloned repos."""
    # Collect source files
    files = collect_files(repo_path)
    if not files:
        raise HTTPException(400, "No supported source files found in repository")

    # Chunk
    chunks = chunk_repository(repo_path, files, repo_id)

    # Build code graph
    graph = build_code_graph(chunks, repo_id)
    save_graph(graph, repo_id)

    # Index into ChromaDB + BM25
    stats = index_chunks(chunks, repo_id)

    elapsed = time.time() - start_time
    log.info(f"Ingestion complete in {elapsed:.1f}s: {stats['total_chunks']} chunks")

    # Register repo
    languages = list(stats.get("languages", {}).keys())
    _repo_registry[repo_id] = {
        "repo_id": repo_id,
        "repo_name": repo_name,
        "total_files": len(files),
        "total_chunks": stats["total_chunks"],
        "languages": languages,
        "repo_path": str(repo_path),
    }
    _save_registry()

    return IngestResponse(
        repo_id=repo_id,
        repo_name=repo_name,
        total_files=len(files),
        total_chunks=stats["total_chunks"],
        languages=languages,
    )
