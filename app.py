"""Streamlit web UI for the multi-document RAG system."""
import atexit
from collections import Counter

import streamlit as st

from src.observability.tracer import init as init_langfuse, flush as flush_langfuse
from src.retrieval.reranked_retriever import RerankedRetriever
from rag import answer

st.set_page_config(
    page_title="Multi-Doc RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def setup() -> RerankedRetriever:
    """One-time setup: Langfuse client + RerankedRetriever (cached across reruns)."""
    init_langfuse()
    atexit.register(flush_langfuse)
    return RerankedRetriever()


def get_library_stats(retriever: RerankedRetriever) -> dict[str, int]:
    """Count chunks per source filename in the collection."""
    items = retriever.collection.get(include=["metadatas"])
    sources = [m["source"] for m in items["metadatas"]]
    return dict(Counter(sources))


retriever = setup()

if "history" not in st.session_state:
    st.session_state.history = []


def render_citation_badge(val: dict) -> None:
    """Render a citation-status badge for an answer."""
    if val.get("fallback_used"):
        st.error(
            "🛡️ Refused to answer — citations could not be verified "
            "even after retry. The library may not contain enough info."
        )
    elif val.get("is_refusal"):
        st.info("🛡️ Refused — insufficient information in the library.")
    elif val.get("is_valid"):
        label = f"✅ All {val.get('num_valid', 0)} citation(s) verified"
        if val.get("retried"):
            label += " (after retry)"
        st.success(label)
    else:
        st.warning("⚠️ Could not verify citations — answer may be unreliable.")


def render_entry(entry: dict) -> None:
    """Render a single Q&A entry."""
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(entry["answer"])

        render_citation_badge(entry.get("citation_validation", {}))

        with st.expander(f"📖 Sources ({len(entry['chunks'])} chunks)"):
            for c in entry["chunks"]:
                st.markdown(
                    f"**{c['source']}** — page {c['page_num']} — "
                    f"distance `{c['distance']:.3f}`"
                )
                st.text(c["text"])
                st.divider()

        cols = st.columns(3)
        cols[0].metric("Retrieve", f"{entry['latency_retrieve_s']:.2f}s")
        cols[1].metric("Generate", f"{entry['latency_generate_s']:.2f}s")
        cols[2].metric("Total", f"{entry['latency_total_s']:.2f}s")


with st.sidebar:
    st.title("📚 Multi-Doc RAG")
    st.markdown(
        """
A local, free RAG system that answers questions
across your personal book library.

**Stack**
- LLM: `llama3.1:8b` (Ollama)
- Embeddings: `nomic-embed-text` (Ollama)
- Retrieval: BM25 + Vector (RRF) + cross-encoder rerank
- Reranker: bge-reranker-v2-m3
- Vector DB: ChromaDB
- Observability: Langfuse
- UI: Streamlit
"""
    )

    st.divider()
    st.subheader("📚 Library")
    library = get_library_stats(retriever)
    if library:
        st.metric("Total chunks", retriever.collection.count())
        st.write(f"**{len(library)} book(s) loaded:**")
        for source, count in sorted(library.items()):
            st.markdown(f"- `{source}` ({count} chunks)")
    else:
        st.warning(
            "No books loaded yet. From the project folder run:\n\n"
            "`uv run python ingest.py data/raw/<your_book.pdf>`"
        )

    st.divider()
    st.subheader("Settings")
    top_k = st.slider(
        "Chunks to retrieve (top_k)",
        min_value=1, max_value=10, value=5,
    )

    st.divider()
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    if st.button("🔄 Refresh library", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.caption("Traces: [Langfuse dashboard](https://cloud.langfuse.com)")


st.title("Ask Your Library")
st.caption("Get answers from your books, with cited sources and page numbers.")

with st.expander("💡 Tips for asking good questions"):
    st.markdown(
        """
- **Be specific**: *"What does Kiyosaki say about assets?"* beats *"tell me about money"*
- **Ask one thing at a time** for best results
- **Scope by book** if you want: *"In Atomic Habits, what is habit stacking?"*
- **Citations** show which book + page each answer is drawn from
- Add more books anytime with `uv run python ingest.py <pdf_path>`
"""
    )

for entry in st.session_state.history:
    render_entry(entry)

query = st.chat_input("Ask a question about anything in your library...")

if query:
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        with st.spinner(
            f"Searching {retriever.collection.count()} chunks across "
            f"{len(library)} book(s)... (20-60s on CPU)"
        ):
            result = answer(query, retriever, top_k=top_k, verbose=False)
            flush_langfuse()

        st.markdown(result["answer"])

        render_citation_badge(result.get("citation_validation", {}))

        with st.expander(f"📖 Sources ({len(result['chunks'])} chunks)"):
            for c in result["chunks"]:
                st.markdown(
                    f"**{c['source']}** — page {c['page_num']} — "
                    f"distance `{c['distance']:.3f}`"
                )
                st.text(c["text"])
                st.divider()

        cols = st.columns(3)
        cols[0].metric("Retrieve", f"{result['latency_retrieve_s']:.2f}s")
        cols[1].metric("Generate", f"{result['latency_generate_s']:.2f}s")
        cols[2].metric("Total", f"{result['latency_total_s']:.2f}s")

    st.session_state.history.append(result)
