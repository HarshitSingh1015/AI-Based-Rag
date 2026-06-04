"""Tests for multi-format document loaders and dispatcher."""
from pathlib import Path

import pytest

from src.ingest.format_loaders import (
    _split_into_blocks,
    load_code,
    load_md,
    load_txt,
)
from src.ingest.loader import SUPPORTED_EXTENSIONS, load_document


def test_split_empty_returns_empty() -> None:
    assert _split_into_blocks("", "test.txt", 50) == []


def test_split_short_text_one_block() -> None:
    blocks = _split_into_blocks("Line one\nLine two", "test.txt", 50)
    assert len(blocks) == 1
    assert blocks[0]["page_num"] == 1
    assert "Line one" in blocks[0]["text"]
    assert blocks[0]["source"] == "test.txt"


def test_split_multiple_blocks() -> None:
    text = "\n".join(f"Line {i}" for i in range(120))
    blocks = _split_into_blocks(text, "test.txt", lines_per_block=50)
    assert len(blocks) == 3
    assert [b["page_num"] for b in blocks] == [1, 2, 3]


def test_load_txt(tmp_path: Path) -> None:
    f = tmp_path / "sample.txt"
    f.write_text("Hello world.\nThis is line 2.", encoding="utf-8")
    pages = load_txt(f)
    assert len(pages) == 1
    assert pages[0]["source"] == "sample.txt"
    assert "Hello world" in pages[0]["text"]


def test_load_md(tmp_path: Path) -> None:
    f = tmp_path / "notes.md"
    f.write_text("# Heading\n\nSome content here.", encoding="utf-8")
    pages = load_md(f)
    assert len(pages) == 1
    assert "Heading" in pages[0]["text"]
    assert pages[0]["source"] == "notes.md"


def test_load_code(tmp_path: Path) -> None:
    f = tmp_path / "module.py"
    f.write_text(
        "def foo():\n    return 42\n\nclass Bar:\n    pass\n",
        encoding="utf-8",
    )
    pages = load_code(f)
    assert len(pages) == 1
    assert "def foo" in pages[0]["text"]
    assert pages[0]["source"] == "module.py"


def test_load_code_uses_smaller_blocks(tmp_path: Path) -> None:
    """Code should split into 30-line blocks (vs 50 for prose)."""
    f = tmp_path / "long.py"
    f.write_text("\n".join(f"x = {i}" for i in range(100)), encoding="utf-8")
    pages = load_code(f)
    assert len(pages) >= 3


def test_load_txt_handles_latin1(tmp_path: Path) -> None:
    """Latin-1 encoded file should not crash."""
    f = tmp_path / "latin.txt"
    f.write_bytes("Cafe resume".encode("latin-1"))
    pages = load_txt(f)
    assert len(pages) == 1
    assert pages[0]["text"]


def test_dispatcher_supports_expected_extensions() -> None:
    for ext in [".pdf", ".txt", ".md", ".docx", ".py", ".js", ".ts"]:
        assert ext in SUPPORTED_EXTENSIONS, f"{ext} should be supported"


def test_dispatcher_unsupported_extension(tmp_path: Path) -> None:
    f = tmp_path / "weird.xyz"
    f.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported"):
        load_document(f)


def test_dispatcher_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "does_not_exist.txt")


def test_dispatcher_routes_txt(tmp_path: Path) -> None:
    f = tmp_path / "sample.txt"
    f.write_text("Hello world.", encoding="utf-8")
    pages = load_document(f)
    assert len(pages) == 1
    assert pages[0]["source"] == "sample.txt"


def test_dispatcher_routes_md(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text("# Title", encoding="utf-8")
    pages = load_document(f)
    assert len(pages) == 1


def test_dispatcher_routes_code(tmp_path: Path) -> None:
    f = tmp_path / "script.py"
    f.write_text("print('hi')", encoding="utf-8")
    pages = load_document(f)
    assert len(pages) == 1
    assert pages[0]["source"] == "script.py"
