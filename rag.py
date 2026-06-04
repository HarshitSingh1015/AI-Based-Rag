"""End-to-end RAG with citation enforcement.

Pipeline: question -> retrieve -> generate -> validate citations
              -> if invalid: retry once -> if still invalid: safe refusal
"""
import atexit
import sys
import time

from langfuse.decorators import langfuse_context, observe

from src.generate.citation_validator import validate_citations
from src.generate.llm import generate
from src.generate.prompt import build_messages, build_retry_messages
from src.observability.tracer import flush as flush_langfuse
from src.observability.tracer import init as init_langfuse
from src.retrieval.reranked_retriever import RerankedRetriever

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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

SAFE_REFUSAL = (
    "I don't have enough information in the available books to "
    "answer this confidently."
)


@observe(name="rag_query")
def answer(
    question: str,
    retriever: RerankedRetriever,
    top_k: int = TOP_K,
    verbose: bool = True,
    max_retries: int = 1,
) -> dict:
    """Run a full RAG cycle with citation enforcement."""
    t0 = time.time()
    chunks = retriever.retrieve(question, top_k=top_k)
    t_retrieve = time.time() - t0

    messages = build_messages(question, chunks)
    t1 = time.time()
    response = generate(messages)
    t_generate = time.time() - t1

    validation = validate_citations(response, chunks)
    retried = False
    fallback_used = False

    if not validation.is_valid and max_retries > 0:
        retried = True
        retry_messages = build_retry_messages(messages, response)
        t2 = time.time()
        response = generate(retry_messages)
        t_generate += time.time() - t2
        validation = validate_citations(response, chunks)

    if not validation.is_valid:
        fallback_used = True
        response = SAFE_REFUSAL
        validation = validate_citations(response, chunks)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Q: {question}")
        print(f"{'=' * 70}")
        print(f"\nRetrieved {len(chunks)} chunks in {t_retrieve:.2f}s:")
        for c in chunks:
            preview = c["text"].replace("\n", " ")[:80]
            print(f"  - page {c['page_num']} ({c['source']}) | {preview}...")
        print(f"\nAnswer (generated in {t_generate:.2f}s):")
        print(response)
        if fallback_used:
            print("\n[!] Citation validation failed after retry — answered with safe refusal.")
        elif validation.is_refusal:
            print("\n[i] System refused to answer (no relevant information).")
        else:
            print(f"\n[+] {len(validation.valid)} valid citation(s) verified.")
            if retried:
                print("    (regenerated once after first attempt had invalid citations)")

    result = {
        "question": question,
        "answer": response,
        "chunks": chunks,
        "latency_retrieve_s": t_retrieve,
        "latency_generate_s": t_generate,
        "latency_total_s": t_retrieve + t_generate,
        "citation_validation": {
            "is_valid": validation.is_valid,
            "is_refusal": validation.is_refusal,
            "num_parsed": len(validation.parsed),
            "num_valid": len(validation.valid),
            "num_invalid": len(validation.invalid),
            "retried": retried,
            "fallback_used": fallback_used,
        },
    }

    langfuse_context.update_current_observation(
        input={"question": question, "top_k": top_k},
        output=response,
        metadata={
            "latency_retrieve_s": t_retrieve,
            "latency_generate_s": t_generate,
            "latency_total_s": t_retrieve + t_generate,
            "num_chunks": len(chunks),
            "citation_is_valid": validation.is_valid,
            "citation_is_refusal": validation.is_refusal,
            "citation_num_valid": len(validation.valid),
            "citation_num_invalid": len(validation.invalid),
            "citation_retried": retried,
            "citation_fallback_used": fallback_used,
        },
    )
    return result


def main() -> None:
    print("=== END-TO-END RAG WITH CITATION ENFORCEMENT ===")
    retriever = RerankedRetriever()
    print(f"Collection size: {retriever.collection.count()}\n")

    results = [answer(q, retriever) for q in TEST_QUESTIONS]

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total questions: {len(results)}")
    print(f"Avg total latency:    "
          f"{sum(r['latency_total_s'] for r in results) / len(results):.2f}s")
    valid = sum(r["citation_validation"]["is_valid"] for r in results)
    refusals = sum(r["citation_validation"]["is_refusal"] for r in results)
    retries = sum(r["citation_validation"]["retried"] for r in results)
    fallbacks = sum(r["citation_validation"]["fallback_used"] for r in results)
    print(f"Valid (verified or honest refusal): {valid}/{len(results)}")
    print(f"  - Refusals: {refusals}")
    print(f"  - Required retry: {retries}")
    print(f"  - Fallback used: {fallbacks}")
    print("\nView traces at: https://cloud.langfuse.com")
    flush_langfuse()


if __name__ == "__main__":
    main()
