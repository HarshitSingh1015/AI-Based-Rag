"""Streamlit web UI for the Rich Dad Poor Dad RAG system."""
import atexit
import streamlit as st

from src.observability.tracer import init as init_langfuse, flush as flush_langfuse
from src.retrieval.retriever import Retriever
from rag import answer

st.set_page_config(
    page_title="Rich Dad Poor Dad RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def setup() -> Retriever:
    """One-time setup: Langfuse client + Retriever (cached across reruns)."""
    init_langfuse()
    atexit.register(flush_langfuse)
    return Retriever()


retriever = setup()

if "history" not in st.session_state:
    st.session_state.history = []


def render_entry(entry: dict) -> None:
    """Render a single Q&A entry (answer + sources + latency metrics)."""
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(entry["answer"])

        with st.expander(f"📖 Sources ({len(entry['chunks'])} chunks)"):
            for c in entry["chunks"]:
                st.markdown(
                    f"**Page {c['page_num']}** — distance `{c['distance']:.3f}`"
                )
                st.text(c["text"])
                st.divider()

        cols = st.columns(3)
        cols[0].metric("Retrieve", f"{entry['latency_retrieve_s']:.2f}s")
        cols[1].metric("Generate", f"{entry['latency_generate_s']:.2f}s")
        cols[2].metric("Total", f"{entry['latency_total_s']:.2f}s")


with st.sidebar:
    st.title("📚 Rich Dad Poor Dad RAG")
    st.markdown(
        """
A local, free RAG system over Robert Kiyosaki's
*Rich Dad Poor Dad*.

**Stack**
- LLM: `llama3.1:8b` (Ollama)
- Embeddings: `nomic-embed-text` (Ollama)
- Vector DB: ChromaDB
- Observability: Langfuse
- UI: Streamlit
"""
    )

    st.divider()
    st.subheader("Settings")
    top_k = st.slider("Chunks to retrieve (top_k)", min_value=1, max_value=10, value=5)

    st.divider()
    st.metric("Indexed chunks", retriever.collection.count())

    st.divider()
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.caption(
        "Traces are live at "
        "[Langfuse dashboard](https://cloud.langfuse.com)"
    )

st.title("Ask Rich Dad Poor Dad")
st.caption(
    "Answers grounded in the book, with cited page numbers and full observability."
)

with st.expander("💡 Example questions to try"):
    st.markdown(
        """
- What is the difference between an asset and a liability?
- Who is the rich dad and who is the poor dad?
- What is the rat race?
- What does Kiyosaki say about traditional education?
- Why do most people stay poor according to Kiyosaki?
- What does Kiyosaki say about owning a house?
- What are the six main lessons in the book?
"""
    )

for entry in st.session_state.history:
    render_entry(entry)

query = st.chat_input("Ask a question about the book...")

if query:
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        with st.spinner(
            f"Searching {retriever.collection.count()} chunks and generating answer "
            "(this takes 20-60s on CPU)..."
        ):
            result = answer(query, retriever, top_k=top_k, verbose=False)
            flush_langfuse()

        st.markdown(result["answer"])

        with st.expander(f"📖 Sources ({len(result['chunks'])} chunks)"):
            for c in result["chunks"]:
                st.markdown(
                    f"**Page {c['page_num']}** — distance `{c['distance']:.3f}`"
                )
                st.text(c["text"])
                st.divider()

        cols = st.columns(3)
        cols[0].metric("Retrieve", f"{result['latency_retrieve_s']:.2f}s")
        cols[1].metric("Generate", f"{result['latency_generate_s']:.2f}s")
        cols[2].metric("Total", f"{result['latency_total_s']:.2f}s")

    st.session_state.history.append(result)
