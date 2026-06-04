"""Format-specific document loaders.

Each loader returns: list[dict] with keys {"page_num", "text", "source"}.
For non-paginated formats (TXT, MD, code), "page_num" is a logical
block index (1-based), where each block is N consecutive lines.
"""
from pathlib import Path

LINES_PER_PROSE_BLOCK = 50
LINES_PER_CODE_BLOCK = 30


def load_pdf(path: Path) -> list[dict]:
    """Load a PDF file, one entry per real page."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({
                "page_num": i,
                "text": text,
                "source": path.name,
            })
    return pages


def load_txt(path: Path) -> list[dict]:
    """Load a plain text file."""
    text = _read_text_with_fallback(path)
    return _split_into_blocks(text, path.name, LINES_PER_PROSE_BLOCK)


def load_md(path: Path) -> list[dict]:
    """Load a Markdown file (treated as text, formatting preserved)."""
    text = _read_text_with_fallback(path)
    return _split_into_blocks(text, path.name, LINES_PER_PROSE_BLOCK)


def load_docx(path: Path) -> list[dict]:
    """Load a Word document. Paragraphs are joined with newlines, then split into blocks."""
    from docx import Document

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)
    return _split_into_blocks(text, path.name, LINES_PER_PROSE_BLOCK)


def load_code(path: Path) -> list[dict]:
    """Load a code file. Uses smaller block size since code is denser."""
    text = _read_text_with_fallback(path)
    return _split_into_blocks(text, path.name, LINES_PER_CODE_BLOCK)


def _read_text_with_fallback(path: Path) -> str:
    """Read a text file; try UTF-8, then chardet detection, then latin-1."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            import chardet
            raw = path.read_bytes()
            detected = chardet.detect(raw)
            encoding = detected.get("encoding") or "latin-1"
            return raw.decode(encoding, errors="replace")
        except Exception:
            return path.read_text(encoding="latin-1", errors="replace")


def _split_into_blocks(text: str, source: str, lines_per_block: int) -> list[dict]:
    """Split text into logical 'pages' by line count."""
    lines = text.splitlines()
    if not lines:
        return []

    blocks = []
    for block_idx, start in enumerate(
        range(0, len(lines), lines_per_block), start=1
    ):
        block_lines = lines[start:start + lines_per_block]
        block_text = "\n".join(block_lines)
        if block_text.strip():
            blocks.append({
                "page_num": block_idx,
                "text": block_text,
                "source": source,
            })
    return blocks
