"""Ingest the Rich Dad Poor Dad PDF into ChromaDB."""
import time
from pathlib import Path

from src.ingest.loader import load_pdf
from src.ingest.chunker import chunk_pages
from src.ingest.embedder import embed_batch
from src.ingest.store import get_client, get_or_create_collection, add_chunks

PDF_PATH = Path("data/raw/rich_dad_poor_dad.pdf")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def main() -> None:
    print(f"=== INGESTION PIPELINE ===\n")

    print(f"[1/4] Loading PDF: {PDF_PATH}")
    pages = load_pdf(PDF_PATH)
    print(f"      Loaded {len(pages)} pages with text\n")

    print(f"[2/4] Chunking pages (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    print(f"      Created {len(chunks)} chunks")
    print(f"      Sample chunk [{chunks[0]['chunk_id']}]:")
    print(f"        {chunks[0]['text'][:150]}...\n")

    print(f"[3/4] Embedding {len(chunks)} chunks via Ollama")
    print(f"      (this may take 5-30 minutes on CPU — be patient)")
    start = time.time()
    embeddings = embed_batch([c["text"] for c in chunks])
    elapsed = time.time() - start
    print(f"      Done in {elapsed:.1f}s "
          f"({elapsed / len(chunks) * 1000:.0f}ms per chunk)")
    print(f"      Embedding dimension: {len(embeddings[0])}\n")

    print(f"[4/4] Storing in ChromaDB")
    client = get_client()
    collection = get_or_create_collection(client)
    add_chunks(collection, chunks, embeddings)
    print(f"      Collection now has {collection.count()} total items\n")

    print("DONE - Ingestion complete!")


if __name__ == "__main__":
    main()
