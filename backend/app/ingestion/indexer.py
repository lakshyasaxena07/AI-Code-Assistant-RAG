"""Dual indexer: Vector (PyTorch) + BM25 (lexical).

Indexes CodeChunks into both stores to enable hybrid retrieval.
We use a custom PyTorch-based in-memory vector store to ensure 
100% portability and avoid C++ compilation errors on Windows (which ChromaDB suffers from).
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.config import settings
from app.models.chunk import CodeChunk
from app.ingestion.embedder import embed_chunks
from app.utils.logger import get_logger

log = get_logger(__name__)


def index_chunks(chunks: list[CodeChunk], repo_id: str) -> dict:
    """Index chunks into PyTorch Vector Store and build a BM25 index."""
    clear_indexer_cache(repo_id)
    if not chunks:
        return {"total_chunks": 0}

    # ── PyTorch Vector Index ──────────────────────────────
    embeddings = embed_chunks(chunks)

    ids = [chunk.chunk_id for chunk in chunks]
    documents = [chunk.to_embedding_text() for chunk in chunks]
    # We store the raw content in the metadata dict for easier retrieval later
    metadatas = [chunk.to_metadata_dict() for chunk in chunks]
    for i, meta in enumerate(metadatas):
        meta["_raw_content"] = chunks[i].content

    vector_data = {
        "ids": ids,
        "embeddings": embeddings,
        "documents": documents,
        "metadatas": metadatas,
    }

    vector_path = settings.chroma_path / f"{repo_id}_vectors.pkl"
    with open(vector_path, "wb") as f:
        pickle.dump(vector_data, f)

    log.info(f"Indexed {len(ids)} chunks into local PyTorch Vector Store")

    # ── BM25 Lexical Index ─────────────────────────────────
    tokenized_docs = [_tokenize_code(chunk.content) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_docs)

    bm25_path = settings.graph_path / f"{repo_id}_bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "chunk_ids": ids,
            "chunks_data": [
                {
                    "chunk_id": c.chunk_id,
                    "content": c.content,
                    "file_path": c.file_path,
                    "name": c.name,
                    "chunk_type": c.chunk_type,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "language": c.language,
                }
                for c in chunks
            ],
        }, f)

    log.info(f"Built BM25 index with {len(tokenized_docs)} documents")

    # Stats
    languages = {}
    for c in chunks:
        languages[c.language] = languages.get(c.language, 0) + 1

    return {
        "total_chunks": len(chunks),
        "languages": languages,
    }


_bm25_cache: dict[str, tuple[BM25Okapi, list[str], list[dict]]] = {}
_vector_cache: dict[str, dict] = {}


def clear_indexer_cache(repo_id: str | None = None):
    """Clear in-memory cached index data (useful on repo re-indexing)."""
    if repo_id:
        _bm25_cache.pop(repo_id, None)
        _vector_cache.pop(repo_id, None)
    else:
        _bm25_cache.clear()
        _vector_cache.clear()


def load_bm25(repo_id: str) -> tuple[BM25Okapi, list[str], list[dict]] | None:
    if repo_id in _bm25_cache:
        return _bm25_cache[repo_id]
    path = settings.graph_path / f"{repo_id}_bm25.pkl"
    if path.exists():
        with open(path, "rb") as f:
            data = pickle.load(f)
        result = (data["bm25"], data["chunk_ids"], data["chunks_data"])
        _bm25_cache[repo_id] = result
        return result
    return None


def load_vector_store(repo_id: str) -> dict | None:
    if repo_id in _vector_cache:
        return _vector_cache[repo_id]
    path = settings.chroma_path / f"{repo_id}_vectors.pkl"
    if path.exists():
        with open(path, "rb") as f:
            data = pickle.load(f)
        _vector_cache[repo_id] = data
        return data
    return None


def _tokenize_code(text: str) -> list[str]:
    import re
    tokens = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+|\w+', text)
    tokens.extend(text.split())
    tokens = [t.lower() for t in tokens if len(t) > 1]
    return tokens
