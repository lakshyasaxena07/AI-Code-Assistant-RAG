"""FastAPI application entry point."""
from __future__ import annotations
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes_ingest import router as ingest_router
from app.api.routes_query import router as query_router
from app.api.routes_repo import router as repo_router
from app.utils.logger import get_logger

log = get_logger(__name__)

app = FastAPI(
    title="AI Codebase Intelligence Assistant",
    description="Advanced RAG-powered code analysis, debugging, and architecture reasoning engine",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(repo_router)


@app.on_event("startup")
async def startup_warmup():
    """Preload embedding model at server startup to eliminate first-query cold start delay."""
    try:
        from app.ingestion.embedder import get_embedding_model
        log.info("Preloading embedding model on startup...")
        get_embedding_model()
        log.info("Embedding model preloaded successfully.")
    except Exception as e:
        log.warning(f"Model startup preload deferred: {e}")


@app.get("/")
async def root():
    return {
        "name": "AI Codebase Intelligence Assistant",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        reload_dirs=["app"],
        reload_excludes=["data/*", "*/data/*"],
    )
