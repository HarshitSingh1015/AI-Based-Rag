"""Retrieval interface — wraps embedder + vector store for clean RAG calls."""
from src.ingest.embedder import embed_text
from src.ingest.store import get_client, get_or_create_collection, query as store_query


class Retriever:
    """Simple vector retriever wrapping ChromaDB."""

    def __init__(self) -> None:
        client = get_client()
        self.collection = get_or_create_collection(client)

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        """Retrieve top-k chunks relevant to the question.

        Returns:
            List of dicts with keys:
            - chunk_id, text, page_num, source, distance
        """
        q_emb = embed_text(question)
        results = store_query(self.collection, q_emb, top_k=top_k)

        chunks: list[dict] = []
        for doc, meta, dist, cid in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            chunks.append({
                "chunk_id": cid,
                "text": doc,
                "page_num": meta["page_num"],
                "source": meta["source"],
                "distance": dist,
            })
        return chunks
