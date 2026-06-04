"""Unit tests for the citation validator (deterministic logic)."""
from src.generate.citation_validator import (
    extract_citations,
    is_refusal,
    validate_citations,
)


def make_chunk(source: str, page: int, text: str = "x") -> dict:
    return {"source": source, "page_num": page, "text": text}


def test_extract_simple_citation() -> None:
    cits = extract_citations("Asset puts money [book.pdf, page 67].")
    assert len(cits) == 1
    assert cits[0].source == "book.pdf"
    assert cits[0].page == 67


def test_extract_multiple_citations() -> None:
    text = (
        "An asset [book.pdf, page 67] differs from a liability "
        "[book.pdf, page 71]."
    )
    cits = extract_citations(text)
    assert len(cits) == 2
    assert sorted(c.page for c in cits) == [67, 71]


def test_no_citations() -> None:
    assert extract_citations("This text has no citations.") == []


def test_is_refusal_positive() -> None:
    assert is_refusal(
        "I don't have enough information in the available books to "
        "answer this confidently."
    )
    assert is_refusal("I do not have enough information to answer.")


def test_is_refusal_negative() -> None:
    assert not is_refusal(
        "An asset puts money in your pocket [book.pdf, page 67]."
    )


def test_validate_valid_citation() -> None:
    chunks = [make_chunk("book.pdf", 67)]
    v = validate_citations("An asset [book.pdf, page 67].", chunks)
    assert v.is_valid
    assert len(v.valid) == 1
    assert len(v.invalid) == 0


def test_validate_hallucinated_page() -> None:
    chunks = [make_chunk("book.pdf", 67)]
    v = validate_citations("An asset [book.pdf, page 999].", chunks)
    assert not v.is_valid
    assert len(v.invalid) == 1


def test_validate_hallucinated_source() -> None:
    chunks = [make_chunk("book.pdf", 67)]
    v = validate_citations("Harshit studies at IIT [fake.pdf, page 1].", chunks)
    assert not v.is_valid
    assert len(v.invalid) == 1


def test_validate_no_citation_makes_invalid() -> None:
    chunks = [make_chunk("book.pdf", 67)]
    v = validate_citations("An asset puts money.", chunks)
    assert not v.is_valid


def test_validate_refusal_is_valid() -> None:
    chunks = [make_chunk("book.pdf", 67)]
    v = validate_citations(
        "I don't have enough information in the available books to "
        "answer this confidently.",
        chunks,
    )
    assert v.is_valid
    assert v.is_refusal


def test_validate_case_insensitive_source() -> None:
    chunks = [make_chunk("Book.PDF", 67)]
    v = validate_citations("An asset [book.pdf, page 67].", chunks)
    assert v.is_valid
