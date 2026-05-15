"""Upload handler — accepts ZIP files or GitHub URLs and extracts repos to disk."""
from __future__ import annotations
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path

from git import Repo as GitRepo

from app.config import settings
from app.utils.logger import get_logger

log = get_logger(__name__)

# File extensions we process (expand as needed)
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs",
    ".cpp", ".c", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".kt", ".scala", ".r", ".sql", ".sh", ".bash", ".yaml", ".yml",
    ".json", ".toml", ".cfg", ".ini", ".md", ".txt", ".html", ".css",
}

# Directories to skip during ingestion
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv", "env",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".next", ".nuxt", "vendor", "target", ".idea", ".vscode",
}

# Max file size to process (500 KB)
MAX_FILE_SIZE = 500_000


def clone_github_repo(github_url: str) -> tuple[str, Path]:
    """Clone a public GitHub repo and return (repo_id, repo_path)."""
    repo_id = str(uuid.uuid4())[:8]
    repo_name = github_url.rstrip("/").split("/")[-1].replace(".git", "")
    dest = settings.repos_path / f"{repo_id}_{repo_name}"
    log.info(f"Cloning {github_url} → {dest}")
    GitRepo.clone_from(github_url, str(dest), depth=1)  # shallow clone for speed
    return repo_id, dest


def extract_zip(zip_path: Path) -> tuple[str, Path]:
    """Extract an uploaded ZIP and return (repo_id, repo_path)."""
    repo_id = str(uuid.uuid4())[:8]
    dest = settings.repos_path / repo_id
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    # If ZIP contains a single top-level folder, use that as root
    children = list(dest.iterdir())
    if len(children) == 1 and children[0].is_dir():
        actual_root = children[0]
        repo_name = actual_root.name
    else:
        actual_root = dest
        repo_name = repo_id
    log.info(f"Extracted ZIP → {actual_root} ({repo_name})")
    return repo_id, actual_root


async def save_upload(file_content: bytes, filename: str) -> tuple[str, Path]:
    """Save uploaded ZIP bytes to temp, extract, and return repo path."""
    tmp = Path(tempfile.mkdtemp(dir=str(settings.repos_path)))
    zip_path = tmp / filename
    zip_path.write_bytes(file_content)
    repo_id, repo_path = extract_zip(zip_path)
    shutil.rmtree(tmp, ignore_errors=True)
    return repo_id, repo_path


def collect_files(repo_path: Path) -> list[Path]:
    """Walk the repo and collect processable source files."""
    files = []
    for f in repo_path.rglob("*"):
        if f.is_dir():
            continue
        # Skip ignored directories
        if any(skip in f.parts for skip in SKIP_DIRS):
            continue
        if f.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if f.stat().st_size > MAX_FILE_SIZE:
            continue
        files.append(f)
    log.info(f"Collected {len(files)} files from {repo_path.name}")
    return files


def detect_language(file_path: Path) -> str:
    """Map file extension to tree-sitter language name."""
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".tsx": "typescript", ".jsx": "javascript", ".java": "java",
        ".go": "go", ".rs": "rust", ".cpp": "cpp", ".c": "c",
        ".h": "c", ".hpp": "cpp", ".cs": "c_sharp", ".rb": "ruby",
        ".php": "php", ".swift": "swift", ".kt": "kotlin",
        ".scala": "scala", ".r": "r",
    }
    return ext_map.get(file_path.suffix.lower(), "")
