"""ChromaDB vector store wrapper."""
from pathlib import Path
import chromadb

DB_DIR = "chroma_db"
COLLECTION_NAME = "rich_dad_chunks"


def get_client(persist_dir: str = DB_DIR):
    """Return a persistent ChromaDB client."""
    Path(persist_dir).mkdir(exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def get_or_create_collection(client, name: str = COLLECTION_NAME):
    """Get or create the chunks collection. Uses cosine similarity."""
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(collection, chunks: list[dict], embeddings: list[list[float]]) -> None:
    """Add chunks (with precomputed embeddings) to the collection.

    Skips chunks whose IDs already exist to make this script idempotent.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must be the same length")

    existing = set(collection.get(include=[])["ids"])
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing]
    new_embeddings = [e for c, e in zip(chunks, embeddings) if c["chunk_id"] not in existing]

    if not new_chunks:
        print("  All chunks already in collection — nothing to add.")
        return

    collection.add(
        ids=[c["chunk_id"] for c in new_chunks],
        embeddings=new_embeddings,
        documents=[c["text"] for c in new_chunks],
        metadatas=[
            {"page_num": c["page_num"], "source": c["source"]}
            for c in new_chunks
        ],
    )
    print(f"  Added {len(new_chunks)} new chunks "
          f"(skipped {len(chunks) - len(new_chunks)} already-existing)")


def query(collection, query_embedding: list[float], top_k: int = 5) -> dict:
    """Query the collection and return top-k most similar chunks."""
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
