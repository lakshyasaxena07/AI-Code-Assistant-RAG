"""Cross-encoder reranker for second-stage precision.

First-stage retrieval (bi-encoder) optimizes for recall — it's fast but approximate.
The cross-encoder reranker sees query + document together, enabling deep interaction
between tokens. This gives much more precise relevance scoring at the cost of speed.

The two-stage pipeline: retrieve 15-20 candidates fast → rerank to top 6-8 precisely.
"""
from __future__ import annotations

from app.config import settings
from app.utils.logger import get_logger

log = get_logger(__name__)

_reranker = None


def get_reranker():
    """Lazy-load the cross-encoder reranker."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        log.info(f"Loading reranker: {settings.reranker_model}")
        _reranker = CrossEncoder(
            settings.reranker_model,
            max_length=512,
            device=settings.embedding_device,
        )
        log.info("Reranker loaded")
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
    """Rerank candidate chunks using a cross-encoder.

    Args:
        query: The user's query
        candidates: List of dicts from hybrid retrieval
        top_k: Number of top results to return after reranking

    Returns:
        Reranked list with added 'rerank_score' field.
    """
    top_k = top_k or settings.rerank_top_k

    if not candidates:
        return []

    reranker = get_reranker()

    # Prepare query-document pairs
    pairs = []
    for c in candidates:
        # Use the raw code content for reranking, not the metadata-enriched version
        content = c.get("content", "")
        # Truncate to avoid exceeding cross-encoder max length
        if len(content) > 1500:
            content = content[:1500]
        pairs.append((query, content))

    # Score all pairs
    scores = reranker.predict(pairs)

    # Attach scores and sort
    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    log.info(
        f"Reranked {len(candidates)} → top {top_k} "
        f"(best={reranked[0]['rerank_score']:.3f}, worst={reranked[-1]['rerank_score']:.3f})"
    )

    return reranked[:top_k]
