"""Hybrid retriever: BM25 + PyTorch Vector search with Reciprocal Rank Fusion."""
from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from rank_bm25 import BM25Okapi

from app.config import settings
from app.ingestion.indexer import load_bm25, load_vector_store, _tokenize_code
from app.ingestion.embedder import embed_query
from app.utils.logger import get_logger

log = get_logger(__name__)


def hybrid_retrieve(
    query: str,
    repo_id: str,
    top_k: int | None = None,
    alpha: float | None = None,
) -> list[dict]:
    top_k = top_k or settings.retrieval_top_k
    alpha = alpha if alpha is not None else settings.hybrid_alpha
    candidate_k = top_k * 3

    # Vector Search
    vector_results = _vector_search(query, repo_id, candidate_k)

    # BM25 Search
    bm25_results = _bm25_search(query, repo_id, candidate_k)

    # Reciprocal Rank Fusion
    fused = _reciprocal_rank_fusion(bm25_results, vector_results, alpha=alpha)
    final = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)[:top_k]

    log.info(
        f"Hybrid retrieval: {len(bm25_results)} BM25 + {len(vector_results)} vector → {len(final)} fused"
    )
    return final


def _vector_search(query: str, repo_id: str, top_k: int) -> list[dict]:
    """Semantic search via PyTorch Cosine Similarity."""
    data = load_vector_store(repo_id)
    if data is None:
        log.warning(f"Vector store not found for {repo_id}")
        return []

    # Get query embedding
    query_emb = torch.tensor([embed_query(query)])  # [1, D]
    doc_embs = torch.tensor(data["embeddings"])     # [N, D]

    if len(doc_embs) == 0:
        return []

    # Cosine similarity
    query_emb = F.normalize(query_emb, p=2, dim=1)
    doc_embs = F.normalize(doc_embs, p=2, dim=1)
    sims = torch.mm(query_emb, doc_embs.transpose(0, 1))[0]  # [N]

    # Get top-K
    k = min(top_k, len(sims))
    top_scores, top_indices = torch.topk(sims, k)

    parsed = []
    for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
        meta = data["metadatas"][idx]
        parsed.append({
            "chunk_id": data["ids"][idx],
            "content": meta.get("_raw_content", data["documents"][idx]),
            "metadata": meta,
            "distance": 1.0 - score,
            "source": "vector",
        })

    return parsed


def _bm25_search(query: str, repo_id: str, top_k: int) -> list[dict]:
    """Lexical search via BM25."""
    bm25_data = load_bm25(repo_id)
    if bm25_data is None:
        return []

    bm25, chunk_ids, chunks_data = bm25_data
    query_tokens = _tokenize_code(query)

    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "chunk_id": chunk_ids[idx],
                "content": chunks_data[idx]["content"],
                "metadata": chunks_data[idx],
                "bm25_score": float(scores[idx]),
                "source": "bm25",
            })

    return results


def _reciprocal_rank_fusion(
    bm25_results: list[dict],
    vector_results: list[dict],
    alpha: float = 0.5,
    k: int = 60,
) -> dict[str, dict]:
    fused: dict[str, dict] = {}
    for rank, result in enumerate(bm25_results):
        cid = result["chunk_id"]
        if cid not in fused:
            fused[cid] = {**result, "rrf_score": 0, "bm25_rank": rank, "vector_rank": -1}
        fused[cid]["rrf_score"] += (1 - alpha) / (k + rank)
        fused[cid]["bm25_rank"] = rank

    for rank, result in enumerate(vector_results):
        cid = result["chunk_id"]
        if cid not in fused:
            fused[cid] = {**result, "rrf_score": 0, "bm25_rank": -1, "vector_rank": rank}
        fused[cid]["rrf_score"] += alpha / (k + rank)
        fused[cid]["vector_rank"] = rank

    return fused
