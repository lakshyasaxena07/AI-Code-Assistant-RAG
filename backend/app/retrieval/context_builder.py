"""Context builder — assembles the final context window for the LLM.

Takes reranked + graph-expanded chunks and constructs a coherent,
deduplicated context string that fits within the token budget.
Prioritizes by rerank score, deduplicates overlapping code, and
formats each chunk with file/line metadata for citation.
"""
from __future__ import annotations

from app.config import settings
from app.utils.logger import get_logger

log = get_logger(__name__)


def build_context(
    chunks: list[dict],
    max_tokens: int | None = None,
) -> tuple[str, list[dict]]:
    """Assemble the context window from retrieved chunks.

    Returns:
        Tuple of (formatted context string, list of chunks actually used).
    """
    max_tokens = max_tokens or settings.max_context_tokens

    # Sort: reranked first (by score), then graph-expanded
    scored = [c for c in chunks if c.get("rerank_score", 0) > -1]
    expanded = [c for c in chunks if c.get("source") == "graph_expansion"]
    scored.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

    ordered = scored + expanded

    # Deduplicate by content hash
    seen_content = set()
    unique = []
    for c in ordered:
        content_key = c.get("content", "")[:200]
        if content_key not in seen_content:
            seen_content.add(content_key)
            unique.append(c)

    # Build context within token budget (rough: 1 token ≈ 4 chars)
    context_parts = []
    used_chunks = []
    current_chars = 0
    char_limit = max_tokens * 4

    for chunk in unique:
        meta = chunk.get("metadata", {})
        file_path = meta.get("file_path", "unknown")
        start_line = meta.get("start_line", "?")
        end_line = meta.get("end_line", "?")
        chunk_type = meta.get("chunk_type", "code")
        name = meta.get("name", "")
        content = chunk.get("content", "")

        # Format with citation markers
        header = f"--- [{file_path}:{start_line}-{end_line}] ({chunk_type}: {name}) ---"
        section = f"{header}\n{content}\n"

        if current_chars + len(section) > char_limit:
            break

        context_parts.append(section)
        used_chunks.append(chunk)
        current_chars += len(section)

    context = "\n".join(context_parts)

    log.info(
        f"[TIMING] Context built: {len(used_chunks)} chunks, "
        f"~{current_chars // 4} tokens"
    )

    return context, used_chunks
