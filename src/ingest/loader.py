"""PDF loading utility for the RAG ingestion pipeline."""
from pathlib import Path
from pypdf import PdfReader


def load_pdf(pdf_path: str | Path) -> list[dict]:
    """Load a PDF file and return a list of page-level documents.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list of dicts, each with keys:
        - page_num (int, 1-indexed)
        - text (str, the extracted text from that page)
        - source (str, the PDF filename)

        Pages with no extractable text are skipped.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages: list[dict] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({
                "page_num": i,
                "text": text,
                "source": pdf_path.name,
            })
    return pages


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingest.loader <pdf_path>")
        sys.exit(1)
    pages = load_pdf(sys.argv[1])
    print(f"Loaded {len(pages)} pages with text")
    print(f"First page preview: {pages[0]['text'][:200]}...")
