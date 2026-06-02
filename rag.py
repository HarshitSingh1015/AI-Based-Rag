"""End-to-end RAG: question -> retrieve -> generate -> answer.
Each call is fully traced in Langfuse.
"""
import atexit
import time

from langfuse.decorators import observe, langfuse_context

from src.observability.tracer import init as init_langfuse, flush as flush_langfuse
from src.retrieval.retriever import Retriever
from src.generate.prompt import build_messages
from src.generate.llm import generate

init_langfuse()
atexit.register(flush_langfuse)

TEST_QUESTIONS = [
    "What is the difference between an asset and a liability?",
    "Who is the rich dad and who is the poor dad?",
    "What is the rat race?",
    "What does Kiyosaki say about traditional education?",
    "Why do most people stay poor according to Kiyosaki?",
]
TOP_K = 5


@observe(name="rag_query")
def answer(question: str, retriever: Retriever, top_k: int = TOP_K, verbose: bool = True) -> dict:
    """Run a full RAG cycle and return the answer + metadata."""
    t0 = time.time()
    chunks = retriever.retrieve(question, top_k=top_k)
    t_retrieve = time.time() - t0

    messages = build_messages(question, chunks)

    t1 = time.time()
    response = generate(messages)
    t_generate = time.time() - t1

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Q: {question}")
        print(f"{'=' * 70}")
        print(f"\nRetrieved {len(chunks)} chunks in {t_retrieve:.2f}s:")
        for c in chunks:
            preview = c["text"].replace("\n", " ")[:80]
            print(f"  - page {c['page_num']} | dist {c['distance']:.3f} | {preview}...")
        print(f"\nAnswer (generated in {t_generate:.2f}s):")
        print(response)

    result = {
        "question": question,
        "answer": response,
        "chunks": chunks,
        "latency_retrieve_s": t_retrieve,
        "latency_generate_s": t_generate,
        "latency_total_s": t_retrieve + t_generate,
    }

    langfuse_context.update_current_observation(
        input={"question": question, "top_k": top_k},
        output=response,
        metadata={
            "latency_retrieve_s": t_retrieve,
            "latency_generate_s": t_generate,
            "latency_total_s": t_retrieve + t_generate,
            "num_chunks": len(chunks),
        },
    )
    return result


def main() -> None:
    print("=== END-TO-END RAG TEST (with Langfuse tracing) ===")
    retriever = Retriever()
    print(f"Collection size: {retriever.collection.count()}\n")

    results = []
    for q in TEST_QUESTIONS:
        results.append(answer(q, retriever))

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total questions: {len(results)}")
    avg_total = sum(r["latency_total_s"] for r in results) / len(results)
    avg_retrieve = sum(r["latency_retrieve_s"] for r in results) / len(results)
    avg_generate = sum(r["latency_generate_s"] for r in results) / len(results)
    print(f"Avg total latency:    {avg_total:.2f}s")
    print(f"Avg retrieve latency: {avg_retrieve:.2f}s")
    print(f"Avg generate latency: {avg_generate:.2f}s")
    print("\nView traces at: https://cloud.langfuse.com")
    flush_langfuse()


if __name__ == "__main__":
    main()
