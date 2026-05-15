"""Hallucination guard — post-generation verification.

Checks that the LLM's response only references files and code that
actually exist in the retrieved context. Flags unverifiable claims.
"""
from __future__ import annotations
import re

from app.utils.logger import get_logger

log = get_logger(__name__)


def verify_response(response: str, used_chunks: list[dict]) -> dict:
    """Verify that citations in the response match retrieved context.

    Returns a verification report with:
    - citations_found: number of citations in response
    - citations_verified: number that match retrieved context
    - unverified_citations: list of citations not found in context
    - confidence: overall confidence score
    """
    # Extract citations like [filename:line-line]
    citation_pattern = r'\[([^\]]+?):(\d+)-(\d+)\]'
    citations = re.findall(citation_pattern, response)

    # Build set of valid file paths from context
    valid_files = set()
    for chunk in used_chunks:
        meta = chunk.get("metadata", {})
        valid_files.add(meta.get("file_path", ""))

    verified = []
    unverified = []

    for file_path, start, end in citations:
        # Check if the cited file exists in our context
        matched = any(
            file_path in vf or vf.endswith(file_path)
            for vf in valid_files
        )
        if matched:
            verified.append(f"[{file_path}:{start}-{end}]")
        else:
            unverified.append(f"[{file_path}:{start}-{end}]")

    total = len(citations)
    verified_count = len(verified)

    # Confidence scoring
    if total == 0:
        confidence = 0.7  # No citations = moderate confidence
    else:
        confidence = verified_count / total if total > 0 else 0.0

    report = {
        "citations_found": total,
        "citations_verified": verified_count,
        "unverified_citations": unverified,
        "confidence": round(confidence, 2),
    }

    if unverified:
        log.warning(f"Unverified citations: {unverified}")

    return report
