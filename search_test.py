"""Test similarity search against the ingested ChromaDB."""
from src.ingest.embedder import embed_text
from src.ingest.store import get_client, get_or_create_collection, query

TEST_QUERIES = [
    "What is the difference between an asset and a liability?",
    "Who is the rich dad and who is the poor dad?",
    "What is the rat race?",
    "What does Kiyosaki say about traditional education?",
    "Why do most people stay poor?",
]


def main() -> None:
    client = get_client()
    collection = get_or_create_collection(client)
    print(f"Collection has {collection.count()} items\n")
    print("=" * 70)

    for q in TEST_QUERIES:
        print(f"\nQ: {q}")
        q_emb = embed_text(q)
        results = query(collection, q_emb, top_k=3)

        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            print(f"  [{i + 1}] page {meta['page_num']} | distance {dist:.3f}")
            preview = doc.replace("\n", " ").strip()[:160]
            print(f"      {preview}...")
        print("-" * 70)


if __name__ == "__main__":
    main()
