"""Universal document loader. Routes by file extension to format-specific loaders."""
from pathlib import Path

from src.ingest import format_loaders

SUPPORTED_EXTENSIONS = {
    ".pdf": format_loaders.load_pdf,
    ".txt": format_loaders.load_txt,
    ".md": format_loaders.load_md,
    ".markdown": format_loaders.load_md,
    ".docx": format_loaders.load_docx,
    ".py": format_loaders.load_code,
    ".js": format_loaders.load_code,
    ".ts": format_loaders.load_code,
    ".tsx": format_loaders.load_code,
    ".jsx": format_loaders.load_code,
    ".java": format_loaders.load_code,
    ".cpp": format_loaders.load_code,
    ".c": format_loaders.load_code,
    ".h": format_loaders.load_code,
    ".go": format_loaders.load_code,
    ".rs": format_loaders.load_code,
    ".rb": format_loaders.load_code,
    ".php": format_loaders.load_code,
    ".sh": format_loaders.load_code,
}


def load_document(path: str | Path) -> list[dict]:
    """Load a document of any supported format.

    Returns list of dicts: [{"page_num": int, "text": str, "source": str}, ...]
    Raises ValueError if extension is unsupported.
    Raises FileNotFoundError if file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS.keys()))
        raise ValueError(
            f"Unsupported file format: {ext}. Supported: {supported}"
        )

    loader_fn = SUPPORTED_EXTENSIONS[ext]
    return loader_fn(path)


def load_pdf(path: str | Path) -> list[dict]:
    """Backward-compatible PDF loader. Prefer load_document() going forward."""
    return format_loaders.load_pdf(Path(path))
