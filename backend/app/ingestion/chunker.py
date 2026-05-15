"""AST-aware chunking engine.

Converts parsed AST structures into CodeChunk objects. Key design decisions:
- Functions and methods become individual chunks (primary retrieval unit)
- Classes get a summary chunk (signature + docstring + method list)
- Each file gets a module-level summary chunk (imports + structure overview)
- Large functions that exceed max_chunk_tokens are split at logical boundaries
- Non-parseable files (markdown, config) become single whole-file chunks
"""
from __future__ import annotations
from pathlib import Path

from app.config import settings
from app.models.chunk import CodeChunk
from app.ingestion.parser import parse_file_ast
from app.ingestion.upload import detect_language
from app.utils.logger import get_logger

log = get_logger(__name__)


def chunk_file(file_path: Path, repo_root: Path, repo_id: str) -> list[CodeChunk]:
    """Chunk a single source file into CodeChunk objects."""
    try:
        source_code = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        log.warning(f"Cannot read {file_path}: {e}")
        return []

    if not source_code.strip():
        return []

    relative_path = str(file_path.relative_to(repo_root)).replace("\\", "/")
    language = detect_language(file_path)
    chunks: list[CodeChunk] = []

    # Parse AST if language is supported
    if language:
        parsed = parse_file_ast(file_path, source_code, language)
        # Extract function chunks
        for fn in parsed["functions"]:
            chunks.append(_make_chunk(fn, relative_path, language, repo_id))

        # Extract class chunks (summary + individual methods)
        for cls in parsed["classes"]:
            # Class summary chunk
            summary_content = _build_class_summary(cls)
            chunks.append(CodeChunk(
                content=summary_content,
                file_path=relative_path,
                language=language,
                chunk_type="class",
                name=cls["name"],
                start_line=cls["start_line"],
                end_line=cls["end_line"],
                docstring=cls.get("docstring"),
                signature=cls.get("signature"),
                calls=cls.get("calls", []),
                repo_id=repo_id,
            ))
            # Individual method chunks
            for method in cls.get("methods", []):
                chunks.append(_make_chunk(method, relative_path, language, repo_id))

        # Module summary chunk — always created for files with parsed content
        chunks.append(_build_module_summary(
            relative_path, language, source_code, parsed, repo_id
        ))
        
        # FIX: If no semantic chunks (functions/classes) were found (e.g. parser failed or empty file),
        # add the whole file content as a chunk to ensure RAG has access to it.
        if len(chunks) <= 1: # Only contains module summary
            chunks.append(CodeChunk(
                content=source_code[:settings.max_chunk_tokens * 8],
                file_path=relative_path,
                language=language,
                chunk_type="file_fallback",
                name=file_path.name,
                start_line=1,
                end_line=source_code.count("\n") + 1,
                repo_id=repo_id,
            ))
    else:
        # Non-parseable file: treat as single chunk
        chunks.append(CodeChunk(
            content=source_code[:settings.max_chunk_tokens * 4],  # rough char limit
            file_path=relative_path,
            language=file_path.suffix.lstrip("."),
            chunk_type="file",
            name=file_path.name,
            start_line=1,
            end_line=source_code.count("\n") + 1,
            repo_id=repo_id,
        ))

    return chunks


def chunk_repository(repo_path: Path, files: list[Path], repo_id: str) -> list[CodeChunk]:
    """Chunk all files in a repository."""
    all_chunks = []
    for f in files:
        file_chunks = chunk_file(f, repo_path, repo_id)
        all_chunks.extend(file_chunks)
    log.info(f"Generated {len(all_chunks)} chunks from {len(files)} files")
    return all_chunks


def _make_chunk(fn_info: dict, file_path: str, language: str, repo_id: str) -> CodeChunk:
    """Create a CodeChunk from a parsed function/method dict."""
    return CodeChunk(
        content=fn_info["content"],
        file_path=file_path,
        language=language,
        chunk_type="method" if fn_info.get("parent_class") else "function",
        name=fn_info["name"],
        start_line=fn_info["start_line"],
        end_line=fn_info["end_line"],
        docstring=fn_info.get("docstring"),
        signature=fn_info.get("signature"),
        calls=fn_info.get("calls", []),
        parent_class=fn_info.get("parent_class"),
        repo_id=repo_id,
    )


def _build_class_summary(cls: dict) -> str:
    """Build a concise class summary: signature + docstring + method signatures."""
    lines = [cls.get("signature", f"class {cls['name']}")]
    if cls.get("docstring"):
        lines.append(f'    """{cls["docstring"]}"""')
    for m in cls.get("methods", []):
        sig = m.get("signature", f"    def {m['name']}(...)")
        lines.append(f"    {sig}")
        if m.get("docstring"):
            lines.append(f'        """{m["docstring"]}"""')
    return "\n".join(lines)


def _build_module_summary(
    file_path: str,
    language: str,
    source_code: str,
    parsed: dict,
    repo_id: str,
) -> CodeChunk:
    """Build a module-level summary chunk capturing file structure."""
    lines = [f"# Module: {file_path}"]
    if parsed.get("module_docstring"):
        lines.append(f"# {parsed['module_docstring']}")
    if parsed["imports"]:
        lines.append("\n# Imports:")
        for imp in parsed["imports"][:15]:
            lines.append(imp)
    if parsed["classes"]:
        lines.append("\n# Classes:")
        for cls in parsed["classes"]:
            lines.append(f"#   - {cls['name']} ({len(cls.get('methods', []))} methods)")
    if parsed["functions"]:
        lines.append("\n# Functions:")
        for fn in parsed["functions"]:
            sig = fn.get("signature", fn["name"])
            lines.append(f"#   - {sig}")

    return CodeChunk(
        content="\n".join(lines),
        file_path=file_path,
        language=language,
        chunk_type="module_summary",
        name=file_path.split("/")[-1],
        start_line=1,
        end_line=source_code.count("\n") + 1,
        docstring=parsed.get("module_docstring"),
        imports=[_extract_import_name(i) for i in parsed["imports"]],
        repo_id=repo_id,
    )


def _extract_import_name(import_stmt: str) -> str:
    """Extract the module name from an import statement."""
    parts = import_stmt.replace("from ", "").replace("import ", "").split()
    return parts[0] if parts else import_stmt
