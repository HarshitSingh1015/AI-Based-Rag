"""Reranked retriever: hybrid candidates re-scored with a cross-encoder."""
from langfuse.decorators import langfuse_context, observe

from src.rerank.cross_encoder import CrossEncoderReranker
from src.retrieval.hybrid_retriever import HybridRetriever


class RerankedRetriever:
    """Hybrid retrieval followed by cross-encoder reranking.

    Pipeline:
        question
           |-> HybridRetriever (BM25 + Vector + RRF) -> top fetch_k candidates
           |-> CrossEncoderReranker (bge-reranker-v2-m3) -> top_k best chunks

    Same `.retrieve(question, top_k)` interface as VectorRetriever and
    HybridRetriever, so it's a drop-in replacement.
    """

    def __init__(self, fetch_k: int = 20) -> None:
        """fetch_k: how many candidates the hybrid retriever pulls before reranking."""
        self.base = HybridRetriever(fetch_k=fetch_k)
        self.reranker = CrossEncoderReranker()
        self.fetch_k = fetch_k

    @property
    def collection(self):
        """Pass-through for code that calls `retriever.collection.count()` etc."""
        return self.base.collection

    @observe(name="reranked_retrieve")
    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        """Retrieve top_k chunks: hybrid candidates -> cross-encoder rerank."""
        candidates = self.base.retrieve(question, top_k=self.fetch_k)
        final = self.reranker.rerank(question, candidates, top_k=top_k)

        langfuse_context.update_current_observation(
            input={
                "question": question,
                "top_k": top_k,
                "fetch_k": self.fetch_k,
            },
            output={
                "num_final": len(final),
                "chunk_ids": [c["chunk_id"] for c in final],
                "sources_in_top_k": sorted({c["source"] for c in final}),
            },
        )
        return final
