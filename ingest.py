"""Ingest one or more PDFs into ChromaDB.

Usage:
    uv run python ingest.py <path_to_pdf>     # Ingest a single file
    uv run python ingest.py --all              # Ingest every PDF in data/raw/
"""
import argparse
import sys
import time
from pathlib import Path

from src.ingest.loader import load_pdf
from src.ingest.chunker import chunk_pages
from src.ingest.embedder import embed_batch
from src.ingest.store import get_client, get_or_create_collection, add_chunks

RAW_DIR = Path("data/raw")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def ingest_one(pdf_path: Path, collection) -> dict:
    """Ingest a single PDF into the collection. Returns stats."""
    print(f"\n--- Ingesting {pdf_path.name} ---")

    print(f"[1/3] Loading PDF...")
    pages = load_pdf(pdf_path)
    print(f"      Loaded {len(pages)} pages with text")

    print(f"[2/3] Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    print(f"      Created {len(chunks)} chunks")

    print(f"[3/3] Embedding & storing {len(chunks)} chunks "
          f"(may take 5-30 min on CPU)...")
    start = time.time()
    embeddings = embed_batch([c["text"] for c in chunks])
    add_chunks(collection, chunks, embeddings)
    elapsed = time.time() - start
    print(f"      Done in {elapsed:.1f}s")

    return {
        "source": pdf_path.name,
        "pages": len(pages),
        "chunks": len(chunks),
        "elapsed_s": elapsed,
    }


def find_pdfs(directory: Path) -> list[Path]:
    """Return a sorted list of PDF files in the directory."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() == ".pdf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into ChromaDB for RAG.",
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        type=Path,
        default=None,
        help="Path to a single PDF to ingest (or omit and use --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Ingest every PDF found in {RAW_DIR}/",
    )
    args = parser.parse_args()

    if not args.pdf_path and not args.all:
        parser.error("Provide either a PDF path or --all")
    if args.pdf_path and args.all:
        parser.error("Cannot specify both a PDF path and --all")

    if args.all:
        pdfs = find_pdfs(RAW_DIR)
        if not pdfs:
            print(f"No PDFs found in {RAW_DIR}/")
            sys.exit(1)
        print(f"Found {len(pdfs)} PDF(s) in {RAW_DIR}:")
        for p in pdfs:
            print(f"  - {p.name}")
    else:
        if not args.pdf_path.exists():
            print(f"Error: {args.pdf_path} does not exist")
            sys.exit(1)
        pdfs = [args.pdf_path]

    client = get_client()
    collection = get_or_create_collection(client)
    print(f"\nCollection has {collection.count()} items before ingestion")

    stats = []
    for pdf in pdfs:
        stats.append(ingest_one(pdf, collection))

    print(f"\n=== SUMMARY ===")
    print(f"Ingested {len(stats)} file(s):")
    for s in stats:
        print(f"  - {s['source']}: {s['pages']} pages, "
              f"{s['chunks']} chunks ({s['elapsed_s']:.1f}s)")
    print(f"Collection now has {collection.count()} total items")
    print("\nDone!")


if __name__ == "__main__":
    main()
