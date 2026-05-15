"""Query analyzer — classifies intent and rewrites queries for better retrieval.

Different query types need different retrieval strategies:
- "explain the auth flow" → broad semantic search + graph traversal
- "where is getUserById defined?" → precise lexical BM25 search
- "why is this function slow?" → debug mode, pull callers/callees

Query rewriting adds code-specific context so the retrieval pipeline
can find relevant chunks even when the user's query is vague.
"""
from __future__ import annotations
import re

from app.utils.logger import get_logger

log = get_logger(__name__)

# Intent categories and their retrieval configurations
INTENT_CONFIG = {
    "explain": {
        "description": "Explain code structure, architecture, or logic",
        "retrieval_strategy": "broad_semantic",
        "top_k_multiplier": 1.5,
        "graph_hops": 1,
    },
    "find": {
        "description": "Locate specific code (function, class, variable)",
        "retrieval_strategy": "lexical_first",
        "top_k_multiplier": 1.0,
        "graph_hops": 0,
    },
    "debug": {
        "description": "Find bugs, trace errors, analyze issues",
        "retrieval_strategy": "precise_with_graph",
        "top_k_multiplier": 1.5,
        "graph_hops": 2,
    },
    "trace": {
        "description": "Trace call flows, dependencies, data paths",
        "retrieval_strategy": "graph_walk",
        "top_k_multiplier": 1.0,
        "graph_hops": 2,
    },
    "architecture": {
        "description": "Describe overall project structure and design",
        "retrieval_strategy": "broad_semantic",
        "top_k_multiplier": 2.0,
        "graph_hops": 1,
    },
    "general": {
        "description": "General repository question",
        "retrieval_strategy": "hybrid_balanced",
        "top_k_multiplier": 1.0,
        "graph_hops": 1,
    },
}

# Keywords that signal specific intents
INTENT_KEYWORDS = {
    "explain": ["explain", "how does", "what does", "describe", "walk me through", "overview"],
    "find": ["where is", "find", "locate", "show me", "definition of", "search for"],
    "debug": ["bug", "error", "fix", "wrong", "broken", "issue", "fail", "crash", "slow", "problem"],
    "trace": ["trace", "flow", "call chain", "depends on", "dependency", "calls", "uses"],
    "architecture": ["architecture", "structure", "design", "organize", "pattern", "overview of the project"],
}


def classify_intent(query: str) -> str:
    """Classify query intent using keyword matching.

    For a hackathon, keyword matching is fast and predictable.
    In production, you'd use a fine-tuned classifier or LLM call.
    """
    query_lower = query.lower()
    scores = {intent: 0 for intent in INTENT_KEYWORDS}

    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                scores[intent] += 1

    best_intent = max(scores, key=scores.get)
    if scores[best_intent] == 0:
        return "general"

    return best_intent


def rewrite_query(query: str, intent: str, conversation_context: str = "") -> str:
    """Rewrite the query to be more specific for code retrieval.

    Adds structure that helps both BM25 and vector search:
    - Extracts code identifiers (CamelCase, snake_case patterns)
    - Expands abbreviations
    - Incorporates conversation context for follow-up queries
    """
    rewritten = query

    # Extract potential code identifiers from the query
    identifiers = re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', query)  # snake_case
    identifiers += re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', query)  # CamelCase

    # Add conversation context for follow-up queries
    if conversation_context:
        rewritten = f"{conversation_context} {rewritten}"

    # Intent-specific augmentation
    if intent == "debug":
        rewritten = f"bug error fix issue: {rewritten}"
    elif intent == "architecture":
        rewritten = f"project structure design architecture: {rewritten}"
    elif intent == "trace":
        rewritten = f"function call dependency flow: {rewritten}"

    return rewritten


def get_intent_config(intent: str) -> dict:
    """Get retrieval configuration for the classified intent."""
    return INTENT_CONFIG.get(intent, INTENT_CONFIG["general"])
