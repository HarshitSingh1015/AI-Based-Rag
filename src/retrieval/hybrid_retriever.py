"""Hybrid retriever: BM25 + vector search, fused via Reciprocal Rank Fusion."""
from langfuse.decorators import langfuse_context, observe

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.retriever import Retriever as VectorRetriever


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    top_k: int = 5,
) -> list[dict]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for each chunk = sum over each list it appears in of 1 / (k + rank_in_that_list).
    k=60 is the standard from Cormack et al. 2009 — robust and rarely needs tuning.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = chunk

    sorted_ids = sorted(scores.keys(), key=lambda cid: -scores[cid])[:top_k]

    fused: list[dict] = []
    for cid in sorted_ids:
        chunk = dict(chunk_map[cid])
        chunk["rrf_score"] = scores[cid]
        if "distance" not in chunk:
            chunk["distance"] = 0.0
        fused.append(chunk)
    return fused


class HybridRetriever:
    """Combines vector + BM25 retrieval via RRF fusion.

    Backward-compatible with the existing Retriever interface: same `.retrieve()`
    signature and same `.collection` attribute used by the Streamlit sidebar.
    """

    def __init__(self, fetch_k: int = 20) -> None:
        """fetch_k controls how many candidates each retriever fetches before fusion."""
        self.vector = VectorRetriever()
        self.bm25 = BM25Retriever()
        self.fetch_k = fetch_k

    @property
    def collection(self):
        """Pass-through for code that calls `retriever.collection.count()` etc."""
        return self.vector.collection

    @observe(name="hybrid_retrieve")
    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        """Run both retrievers, fuse via RRF, return top-k."""
        vector_chunks = self.vector.retrieve(question, top_k=self.fetch_k)
        bm25_chunks = self.bm25.retrieve(question, top_k=self.fetch_k)

        fused = reciprocal_rank_fusion(
            [vector_chunks, bm25_chunks],
            top_k=top_k,
        )

        langfuse_context.update_current_observation(
            input={
                "question": question,
                "top_k": top_k,
                "fetch_k": self.fetch_k,
            },
            output={
                "num_fused": len(fused),
                "chunk_ids": [c["chunk_id"] for c in fused],
                "vector_top1": vector_chunks[0]["chunk_id"] if vector_chunks else None,
                "bm25_top1": bm25_chunks[0]["chunk_id"] if bm25_chunks else None,
                "sources_in_top_k": list({c["source"] for c in fused}),
            },
        )
        return fused
