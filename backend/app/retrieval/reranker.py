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
    """Rerank candidate chunks using a cross-encoder with CPU optimizations.

    Args:
        query: The user's query
        candidates: List of dicts from hybrid retrieval
        top_k: Number of top results to return after reranking

    Returns:
        Reranked list with added 'rerank_score' field.
    """
    import time
    top_k = top_k or settings.rerank_top_k

    if not candidates:
        return []

    # Allow bypassing reranker for low-resource CPU setups
    if not getattr(settings, "enable_reranker", True):
        log.info(f"[TIMING] Reranker disabled by config. Returning top {top_k} from hybrid RRF.")
        return candidates[:top_k]

    t0_rerank = time.perf_counter()

    # Bounded candidate pool: only rerank top N candidates to keep CPU latency predictable
    max_candidates = getattr(settings, "rerank_max_candidates", 10)
    pool = candidates[:max_candidates]
    remainder = candidates[max_candidates:]

    reranker = get_reranker()

    # Prepare query-document pairs (truncate to 1000 chars for faster CPU cross-attention)
    pairs = []
    for c in pool:
        content = c.get("content", "")
        if len(content) > 1000:
            content = content[:1000]
        pairs.append((query, content))

    # Score candidate pairs
    scores = reranker.predict(pairs)

    # Attach scores and sort
    for i, c in enumerate(pool):
        c["rerank_score"] = float(scores[i])

    reranked = sorted(pool, key=lambda x: x["rerank_score"], reverse=True)
    t_rerank_ms = (time.perf_counter() - t0_rerank) * 1000

    log.info(
        f"[TIMING] Reranked {len(pool)} candidates -> top {top_k} in {t_rerank_ms:.1f}ms "
        f"(best={reranked[0]['rerank_score']:.3f}, worst={reranked[-1]['rerank_score']:.3f})"
    )

    return reranked[:top_k]
