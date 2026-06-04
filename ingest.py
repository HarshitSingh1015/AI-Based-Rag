"""Top-level ingestion script. Accepts any supported document format.

Usage:
    uv run python ingest.py <file_path>           # ingest a single file
    uv run python ingest.py <file1> <file2> ...   # ingest multiple files
    uv run python ingest.py --all                 # ingest all supported files in data/raw/
"""
import argparse
import sys
from pathlib import Path

from src.ingest.chunker import chunk_pages
from src.ingest.embedder import embed_batch
from src.ingest.loader import SUPPORTED_EXTENSIONS, load_document
from src.ingest.store import add_chunks, get_client, get_or_create_collection

RAW_DIR = Path("data/raw")


def ingest_file(path: Path, collection) -> int:
    """Ingest a single file. Returns number of NEW chunks added (0 if skipped or all duplicates)."""
    print(f"\n[{path.name}] Loading...")
    try:
        pages = load_document(path)
    except ValueError as e:
        print(f"[{path.name}] SKIP: {e}")
        return 0
    except FileNotFoundError as e:
        print(f"[{path.name}] SKIP: {e}")
        return 0

    if not pages:
        print(f"[{path.name}] No text extracted. Skipping.")
        return 0

    print(f"[{path.name}] Loaded {len(pages)} pages/blocks")

    chunks = chunk_pages(pages)
    print(f"[{path.name}] Chunked into {len(chunks)} pieces")

    texts = [c["text"] for c in chunks]
    embeddings = embed_batch(texts)

    before = collection.count()
    add_chunks(collection, chunks, embeddings)
    after = collection.count()
    added = after - before
    print(
        f"[{path.name}] Added {added} new chunks "
        f"(collection now has {after})"
    )
    return added


def discover_files() -> list[Path]:
    """Find all supported files in data/raw/."""
    files: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(RAW_DIR.glob(f"*{ext}"))
    return sorted(set(files))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG vector store."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Specific files to ingest.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all supported files in data/raw/.",
    )
    args = parser.parse_args()

    if args.all:
        files = discover_files()
        if not files:
            print(f"No supported files found in {RAW_DIR}")
            print(f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS.keys())}")
            sys.exit(1)
        print(f"Found {len(files)} files to ingest:")
        for f in files:
            print(f"  - {f.name}")
    elif args.paths:
        files = [Path(p) for p in args.paths]
    else:
        parser.print_help()
        sys.exit(1)

    client = get_client()
    collection = get_or_create_collection(client)

    total = 0
    for f in files:
        total += ingest_file(f, collection)

    print("\n=========================================")
    print(f"Ingestion complete. Total new chunks: {total}")
    print(f"Collection size: {collection.count()}")
    print("=========================================")


if __name__ == "__main__":
    main()
