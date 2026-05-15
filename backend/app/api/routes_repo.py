"""Repository exploration routes — file tree, graph data, repo info."""
from __future__ import annotations
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.models.schemas import GraphData, GraphNode, GraphEdge, FileNode, RepoInfo
from app.ingestion.graph_builder import load_graph, get_graph_data
from app.api.routes_ingest import get_repo_registry
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/api/repo", tags=["repository"])


@router.get("/{repo_id}/info", response_model=RepoInfo)
async def get_repo_info(repo_id: str):
    """Get repository metadata and file tree."""
    registry = get_repo_registry()
    if repo_id not in registry:
        raise HTTPException(404, "Repository not found")

    repo = registry[repo_id]
    repo_path = Path(repo["repo_path"])

    file_tree = _build_file_tree(repo_path, repo_path)

    return RepoInfo(
        repo_id=repo_id,
        repo_name=repo["repo_name"],
        total_files=repo["total_files"],
        total_chunks=repo["total_chunks"],
        languages={lang: 0 for lang in repo["languages"]},
        file_tree=file_tree,
    )


@router.get("/{repo_id}/graph", response_model=GraphData)
async def get_graph(repo_id: str):
    """Get the code dependency graph for visualization."""
    graph = load_graph(repo_id)
    if graph is None:
        raise HTTPException(404, "Graph not found for this repository")

    data = get_graph_data(graph)

    nodes = [GraphNode(**n) for n in data["nodes"]]
    edges = [GraphEdge(**e) for e in data["edges"]]

    return GraphData(nodes=nodes, edges=edges)


@router.get("/{repo_id}/file")
async def get_file_content(repo_id: str, path: str):
    """Get the content of a specific file in the repository."""
    registry = get_repo_registry()
    if repo_id not in registry:
        raise HTTPException(404, "Repository not found")

    repo_path = Path(registry[repo_id]["repo_path"])
    file_path = repo_path / path

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, "File not found")

    # Security: ensure path doesn't escape repo directory
    try:
        file_path.resolve().relative_to(repo_path.resolve())
    except ValueError:
        raise HTTPException(403, "Access denied")

    content = file_path.read_text(encoding="utf-8", errors="ignore")
    return {"path": path, "content": content, "lines": content.count("\n") + 1}


def _build_file_tree(path: Path, root: Path) -> FileNode:
    """Recursively build a file tree structure."""
    relative = str(path.relative_to(root)).replace("\\", "/")
    if relative == ".":
        relative = root.name

    node = FileNode(
        path=relative,
        name=path.name,
        is_dir=path.is_dir(),
    )

    if path.is_dir():
        try:
            children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            for child in children:
                # Skip hidden dirs and common non-source dirs
                if child.name.startswith(".") or child.name in (
                    "node_modules", "__pycache__", "venv", ".venv",
                    "dist", "build", ".git",
                ):
                    continue
                node.children.append(_build_file_tree(child, root))
        except PermissionError:
            pass

    return node
