"""Cross-encoder reranker using bge-reranker-v2-m3 from sentence-transformers."""
from langfuse.decorators import langfuse_context, observe
from sentence_transformers import CrossEncoder

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker:
    """Cross-encoder reranker that re-scores (query, chunk) pairs.

    Cross-encoders are much more accurate than bi-encoders (vector embeddings)
    because they consider the query and document JOINTLY rather than encoding
    them separately. They are however much slower — typically used to rerank a
    small candidate pool (20-100) down to a final top-K (3-10).

    bge-reranker-v2-m3 is a strong free multilingual reranker (~570MB) that
    runs on CPU acceptably for batch sizes <=30 candidates.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, max_length: int = 512) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=max_length)

    @observe(name="rerank")
    def rerank(
        self,
        question: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Rerank chunks against the question, return top-k by relevance.

        Each returned chunk is augmented with a 'rerank_score' field
        (higher = more relevant).
        """
        if not chunks:
            return []

        pairs = [(question, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: -float(x[1]),
        )[:top_k]

        reranked: list[dict] = []
        for chunk, score in ranked:
            new_chunk = dict(chunk)
            new_chunk["rerank_score"] = float(score)
            reranked.append(new_chunk)

        langfuse_context.update_current_observation(
            input={
                "question": question,
                "num_candidates": len(chunks),
                "top_k": top_k,
                "model": self.model_name,
            },
            output={
                "num_returned": len(reranked),
                "chunk_ids": [c["chunk_id"] for c in reranked],
                "top_score": reranked[0]["rerank_score"] if reranked else None,
                "bottom_score": reranked[-1]["rerank_score"] if reranked else None,
            },
        )
        return reranked
