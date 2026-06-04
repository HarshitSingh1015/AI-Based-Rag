"""Unit tests for the chunker."""
from src.ingest.chunker import chunk_pages, chunk_text


def test_chunk_empty_text() -> None:
    assert chunk_text("") == []


def test_chunk_short_text_returns_one_chunk() -> None:
    chunks = chunk_text("short text", chunk_size=500)
    assert chunks == ["short text"]


def test_chunk_long_text_returns_multiple_chunks() -> None:
    text = "Sentence one. " * 100
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 600


def test_chunk_pages_preserves_metadata() -> None:
    pages = [
        {"page_num": 1, "text": "Page one content.", "source": "test.pdf"},
        {"page_num": 2, "text": "Page two content.", "source": "test.pdf"},
    ]
    chunks = chunk_pages(pages, chunk_size=500)
    assert len(chunks) >= 2
    assert all("chunk_id" in c for c in chunks)
    assert all(c["source"] == "test.pdf" for c in chunks)
    assert {c["page_num"] for c in chunks} == {1, 2}


def test_chunk_ids_are_unique() -> None:
    pages = [
        {"page_num": 1, "text": "x. " * 200, "source": "test.pdf"},
    ]
    chunks = chunk_pages(pages, chunk_size=200, overlap=20)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
