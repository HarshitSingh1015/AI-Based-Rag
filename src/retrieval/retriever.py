"""Retrieval interface — wraps embedder + vector store for clean RAG calls."""
from langfuse.decorators import observe, langfuse_context

from src.ingest.embedder import embed_text
from src.ingest.store import get_client, get_or_create_collection, query as store_query


class Retriever:
    """Simple vector retriever wrapping ChromaDB."""

    def __init__(self) -> None:
        client = get_client()
        self.collection = get_or_create_collection(client)

    @observe(name="retrieve")
    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        """Retrieve top-k chunks relevant to the question."""
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

        langfuse_context.update_current_observation(
            input={"question": question, "top_k": top_k},
            output={
                "num_chunks": len(chunks),
                "chunk_ids": [c["chunk_id"] for c in chunks],
                "min_distance": min((c["distance"] for c in chunks), default=None),
                "max_distance": max((c["distance"] for c in chunks), default=None),
            },
        )
        return chunks
