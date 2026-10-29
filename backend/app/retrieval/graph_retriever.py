"""Graph-augmented retrieval — expands retrieved results with structurally connected code."""
from __future__ import annotations
import networkx as nx

from app.ingestion.graph_builder import load_graph
from app.ingestion.indexer import load_vector_store
from app.config import settings
from app.utils.logger import get_logger

log = get_logger(__name__)


def graph_expand(
    retrieved: list[dict],
    repo_id: str,
    hops: int | None = None,
) -> list[dict]:
    hops = hops if hops is not None else settings.graph_expansion_hops
    if hops == 0:
        return retrieved

    graph = load_graph(repo_id)
    if graph is None:
        log.warning(f"No graph found for {repo_id}, skipping expansion")
        return retrieved

    existing_ids = {r["chunk_id"] for r in retrieved}
    chunk_to_graph = _map_chunks_to_graph_nodes(retrieved, graph)

    neighbor_node_ids = set()
    for graph_node_id in chunk_to_graph.values():
        if graph_node_id not in graph:
            continue
        try:
            for neighbor in graph.predecessors(graph_node_id):
                neighbor_node_ids.add(neighbor)
            for neighbor in graph.successors(graph_node_id):
                neighbor_node_ids.add(neighbor)
        except nx.NetworkXError:
            continue

    import time
    t0_graph = time.perf_counter()
    expanded_chunks = _fetch_chunks_by_graph_ids(neighbor_node_ids, repo_id, existing_ids)
    t_graph_ms = (time.perf_counter() - t0_graph) * 1000

    log.info(
        f"[TIMING] Graph expansion ({len(chunk_to_graph)} seed nodes -> "
        f"{len(expanded_chunks)} added) in {t_graph_ms:.1f}ms"
    )

    for c in expanded_chunks:
        c["source"] = "graph_expansion"
        c["rrf_score"] = 0.0
        c["rerank_score"] = -1.0

    return retrieved + expanded_chunks


def _map_chunks_to_graph_nodes(chunks: list[dict], graph: nx.DiGraph) -> dict[str, str]:
    mapping = {}
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        file_path = meta.get("file_path", "")
        name = meta.get("name", "")
        parent = meta.get("parent_class", "")

        node_id = f"{file_path}::{parent}.{name}" if parent else f"{file_path}::{name}"
        if node_id in graph:
            mapping[chunk["chunk_id"]] = node_id

    return mapping


def _fetch_chunks_by_graph_ids(
    node_ids: set[str],
    repo_id: str,
    exclude_ids: set[str],
) -> list[dict]:
    """Fetch chunk data for graph node IDs from PyTorch vector store."""
    data = load_vector_store(repo_id)
    if data is None:
        return []

    matched = []
    for i, meta in enumerate(data["metadatas"]):
        chunk_id = data["ids"][i]
        if chunk_id in exclude_ids:
            continue

        file_path = meta.get("file_path", "")
        name = meta.get("name", "")
        parent = meta.get("parent_class", "")

        node_id = f"{file_path}::{parent}.{name}" if parent else f"{file_path}::{name}"

        if node_id in node_ids:
            matched.append({
                "chunk_id": chunk_id,
                "content": meta.get("_raw_content", data["documents"][i]),
                "metadata": meta,
            })

    return matched
