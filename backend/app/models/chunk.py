"""Core data model for code chunks — the atomic unit of the RAG pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field
import uuid


@dataclass
class CodeChunk:
    """A semantically meaningful unit of code extracted via AST parsing."""

    content: str
    file_path: str
    language: str
    chunk_type: str  # "function" | "class" | "method" | "module_summary" | "import_block" | "config"
    name: str  # function/class name or filename for module summaries

    # Location
    start_line: int = 0
    end_line: int = 0

    # Semantic metadata
    docstring: str | None = None
    signature: str | None = None  # e.g., "def verify_token(token: str) -> dict"

    # Graph relationships
    imports: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    called_by: list[str] = field(default_factory=list)
    parent_class: str | None = None

    # Repository context
    repo_id: str = ""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_embedding_text(self) -> str:
        """Prepare text for embedding with metadata prefix for improved retrieval."""
        prefix = f"[{self.language}] [{self.chunk_type}] {self.file_path}"
        if self.name:
            prefix += f" :: {self.name}"
        prefix += "\n"
        if self.docstring:
            prefix += f"Description: {self.docstring}\n"
        if self.signature:
            prefix += f"Signature: {self.signature}\n"
        return prefix + self.content

    def to_metadata_dict(self) -> dict:
        """Flat metadata dict for ChromaDB storage."""
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "language": self.language,
            "chunk_type": self.chunk_type,
            "name": self.name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "docstring": self.docstring or "",
            "signature": self.signature or "",
            "imports": "|".join(self.imports),
            "calls": "|".join(self.calls),
            "parent_class": self.parent_class or "",
            "repo_id": self.repo_id,
        }

    @property
    def display_location(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"
