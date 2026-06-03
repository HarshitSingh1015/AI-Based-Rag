"""BM25 keyword retriever over the chunks stored in ChromaDB."""
import re

from langfuse.decorators import langfuse_context, observe
from rank_bm25 import BM25Okapi

from src.ingest.store import get_client, get_or_create_collection

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercase + word-character tokenization (splits on whitespace AND punctuation).

    Slightly smarter than .split() so we can isolate alphanumeric IDs like
    '22UEC125' that PDF extraction sometimes glues to adjacent punctuation
    (e.g. 'Roll No.: 22UEC125/envel...'). Still uses no extra packages.
    """
    return _WORD_RE.findall(text.lower())


class BM25Retriever:
    """In-memory BM25 retriever built from the persistent ChromaDB collection.

    BM25 is a classic keyword-matching algorithm that excels at exact-term
    matches (names, IDs, technical terms) where vector embeddings are weak.
    """

    def __init__(self) -> None:
        client = get_client()
        self.collection = get_or_create_collection(client)

        items = self.collection.get(include=["documents", "metadatas"])
        self.ids: list[str] = items["ids"]
        self.documents: list[str] = items["documents"]
        self.metadatas: list[dict] = items["metadatas"]

        tokenized_corpus = [_tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    @observe(name="bm25_retrieve")
    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        """Return top-k chunks ranked by BM25 score (higher = better)."""
        tokenized_query = _tokenize(question)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]

        chunks: list[dict] = []
        for rank, idx in enumerate(top_indices):
            chunks.append({
                "chunk_id": self.ids[idx],
                "text": self.documents[idx],
                "page_num": self.metadatas[idx]["page_num"],
                "source": self.metadatas[idx]["source"],
                "score": float(scores[idx]),
                "rank": rank,
            })

        langfuse_context.update_current_observation(
            input={"question": question, "top_k": top_k},
            output={
                "num_chunks": len(chunks),
                "chunk_ids": [c["chunk_id"] for c in chunks],
                "max_score": float(scores[top_indices[0]]) if top_indices else 0.0,
            },
        )
        return chunks
