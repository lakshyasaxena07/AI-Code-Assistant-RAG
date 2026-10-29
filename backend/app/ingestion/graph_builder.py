"""Code graph builder using NetworkX.

Constructs a directed graph of code relationships:
- Nodes: files, classes, functions/methods
- Edges: imports, calls, contains, inherits

This graph powers dependency-aware retrieval: when a user asks about
a function, we can pull in callers, callees, and related imports
that pure vector search would miss.
"""
from __future__ import annotations
import pickle
from pathlib import Path

import networkx as nx

from app.models.chunk import CodeChunk
from app.config import settings
from app.utils.logger import get_logger

log = get_logger(__name__)


def build_code_graph(chunks: list[CodeChunk], repo_id: str) -> nx.DiGraph:
    """Build a directed code graph from extracted chunks."""
    G = nx.DiGraph()

    # Pass 1: Add all nodes
    file_nodes = set()
    for chunk in chunks:
        node_id = _chunk_node_id(chunk)

        G.add_node(node_id, **{
            "type": chunk.chunk_type,
            "name": chunk.name,
            "file_path": chunk.file_path,
            "chunk_id": chunk.chunk_id,
            "language": chunk.language,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
        })

        # File-level node
        if chunk.file_path not in file_nodes:
            G.add_node(chunk.file_path, type="file", name=chunk.file_path,
                       file_path=chunk.file_path, language=chunk.language)
            file_nodes.add(chunk.file_path)

        # "contains" edge: file → function/class
        if chunk.chunk_type != "module_summary":
            G.add_edge(chunk.file_path, node_id, relation="contains")

        # Method → class "contains" edge
        if chunk.parent_class:
            parent_id = f"{chunk.file_path}::{chunk.parent_class}"
            if G.has_node(parent_id):
                G.add_edge(parent_id, node_id, relation="contains")

    # Pass 2: Add call edges (fuzzy matching on function names)
    name_to_nodes = {}
    for chunk in chunks:
        if chunk.chunk_type in ("function", "method", "class"):
            name_to_nodes.setdefault(chunk.name, []).append(_chunk_node_id(chunk))

    for chunk in chunks:
        caller_id = _chunk_node_id(chunk)
        for call_target in chunk.calls:
            # Strip prefixes like "self." or "module."
            short_name = call_target.split(".")[-1]
            if short_name in name_to_nodes:
                for target_id in name_to_nodes[short_name]:
                    if target_id != caller_id:
                        G.add_edge(caller_id, target_id, relation="calls")

    # Pass 3: Add import edges between files
    for chunk in chunks:
        if chunk.chunk_type == "module_summary":
            for imp in chunk.imports:
                # Try to match import to a file in the repo
                for file_node in file_nodes:
                    if _import_matches_file(imp, file_node):
                        G.add_edge(chunk.file_path, file_node, relation="imports")

    log.info(f"Built code graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


_graph_cache: dict[str, nx.DiGraph] = {}


def clear_graph_cache(repo_id: str | None = None):
    """Clear in-memory cached graph."""
    if repo_id:
        _graph_cache.pop(repo_id, None)
    else:
        _graph_cache.clear()


def save_graph(graph: nx.DiGraph, repo_id: str) -> Path:
    """Persist graph to disk and update cache."""
    _graph_cache[repo_id] = graph
    path = settings.graph_path / f"{repo_id}.gpickle"
    with open(path, "wb") as f:
        pickle.dump(graph, f)
    return path


def load_graph(repo_id: str) -> nx.DiGraph | None:
    """Load a persisted graph with in-memory caching."""
    if repo_id in _graph_cache:
        return _graph_cache[repo_id]
    path = settings.graph_path / f"{repo_id}.gpickle"
    if path.exists():
        with open(path, "rb") as f:
            graph = pickle.load(f)
        _graph_cache[repo_id] = graph
        return graph
    return None


def get_graph_data(graph: nx.DiGraph) -> dict:
    """Convert graph to serializable format for the frontend (D3.js)."""
    nodes = []
    for node_id, data in graph.nodes(data=True):
        nodes.append({
            "id": node_id,
            "label": data.get("name", node_id),
            "type": data.get("type", "unknown"),
            "file_path": data.get("file_path", ""),
        })
    edges = []
    for src, tgt, data in graph.edges(data=True):
        edges.append({
            "source": src,
            "target": tgt,
            "relation": data.get("relation", "related"),
        })
    return {"nodes": nodes, "edges": edges}


def _chunk_node_id(chunk: CodeChunk) -> str:
    """Create a unique, readable graph node ID."""
    if chunk.parent_class:
        return f"{chunk.file_path}::{chunk.parent_class}.{chunk.name}"
    return f"{chunk.file_path}::{chunk.name}"


def _import_matches_file(import_name: str, file_path: str) -> bool:
    """Heuristic: check if an import name could refer to a file in the repo."""
    # "app.auth.jwt" → match "app/auth/jwt.py"
    import_as_path = import_name.replace(".", "/")
    clean_file = file_path.replace("\\", "/")
    return (
        clean_file.endswith(f"{import_as_path}.py")
        or clean_file.endswith(f"{import_as_path}/index.ts")
        or clean_file.endswith(f"{import_as_path}/index.js")
        or import_as_path in clean_file
    )
