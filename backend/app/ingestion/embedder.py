"""Embedding generator using sentence-transformers.

Uses a code-optimized embedding model to convert CodeChunks into vectors.
The to_embedding_text() method on CodeChunk prepends metadata (language, type,
file path) to improve retrieval quality by giving the model structural context.
"""
from __future__ import annotations

from sentence_transformers import SentenceTransformer

from app.config import settings
from app.models.chunk import CodeChunk
from app.utils.logger import get_logger

log = get_logger(__name__)

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the embedding model (heavy, only load once)."""
    global _model
    if _model is None:
        log.info(f"Loading embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(
            settings.embedding_model,
            device=settings.embedding_device,
        )
        log.info("Embedding model loaded")
    return _model


def embed_chunks(chunks: list[CodeChunk], batch_size: int = 64) -> list[list[float]]:
    """Generate embeddings for a list of CodeChunks."""
    model = get_embedding_model()
    texts = [chunk.to_embedding_text() for chunk in chunks]
    log.info(f"Embedding {len(texts)} chunks (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # Cosine similarity via dot product
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    model = get_embedding_model()
    embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )
    return embedding[0].tolist()
