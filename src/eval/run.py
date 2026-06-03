"""RAG evaluation runner: golden set -> RAG -> retrieval recall + RAGAS scores."""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness
from ragas.run_config import RunConfig

from src.observability.tracer import flush as flush_langfuse
from src.observability.tracer import init as init_langfuse
from src.generate.llm import generate
from src.generate.prompt import build_messages
from src.retrieval.hybrid_retriever import HybridRetriever

GOLDEN_SET_PATH = Path("evals/golden_set.jsonl")
RESULTS_DIR = Path("evals/results")
TOP_K = 5
JUDGE_LLM = "llama3.1:8b"
JUDGE_EMBED = "nomic-embed-text"


def load_golden_set(path: Path) -> list[dict]:
    items: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_system_on_item(item: dict, retriever: HybridRetriever) -> dict:
    """Run our RAG on a single question; capture all the data RAGAS needs."""
    t0 = time.time()
    chunks = retriever.retrieve(item["question"], top_k=TOP_K)
    t_retrieve = time.time() - t0

    messages = build_messages(item["question"], chunks)
    t1 = time.time()
    answer_text = generate(messages)
    t_generate = time.time() - t1

    return {
        "id": item["id"],
        "question": item["question"],
        "ground_truth": item.get("ground_truth", ""),
        "expected_sources": item.get("expected_sources", []),
        "answer": answer_text,
        "contexts": [c["text"] for c in chunks],
        "retrieved_sources": [c["source"] for c in chunks],
        "retrieved_pages": [c["page_num"] for c in chunks],
        "latency_retrieve_s": t_retrieve,
        "latency_generate_s": t_generate,
    }


def compute_retrieval_recall(rows: list[dict]) -> dict:
    """For each item, did ANY expected_source appear in retrieved_sources?"""
    per_item: list[dict] = []
    hits = 0
    total = 0
    for r in rows:
        if not r["expected_sources"]:
            continue
        total += 1
        hit = any(s in r["retrieved_sources"] for s in r["expected_sources"])
        hits += int(hit)
        per_item.append({
            "id": r["id"],
            "question": r["question"][:60],
            "expected": r["expected_sources"],
            "retrieved_sources_unique": sorted(set(r["retrieved_sources"])),
            "hit": hit,
        })
    return {
        "recall_at_k": hits / total if total else 0.0,
        "hits": hits,
        "total": total,
        "per_item": per_item,
    }


def run_ragas(rows: list[dict]) -> dict:
    """Use RAGAS with Ollama as the LLM-as-judge. Returns per-metric mean scores."""
    print(
        f"  Running RAGAS with judge={JUDGE_LLM} (CPU) "
        f"on {len(rows)} rows × 3 metrics..."
    )
    print(f"  Expected time: ~{len(rows) * 3 * 30 // 60}-{len(rows) * 3 * 60 // 60} minutes.")

    judge_llm = ChatOllama(model=JUDGE_LLM, temperature=0.0)
    judge_embeddings = OllamaEmbeddings(model=JUDGE_EMBED)

    dataset = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in rows
    ])

    run_config = RunConfig(timeout=600, max_workers=1, max_retries=2)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=LangchainLLMWrapper(judge_llm),
        embeddings=LangchainEmbeddingsWrapper(judge_embeddings),
        raise_exceptions=False,
        run_config=run_config,
    )

    df = result.to_pandas()
    metric_cols = [c for c in df.columns if c in {"faithfulness", "answer_relevancy", "context_precision"}]
    mean_scores = {}
    for col in metric_cols:
        try:
            mean_scores[col] = float(df[col].astype(float).mean())
        except Exception:
            mean_scores[col] = None
    per_item = df.to_dict(orient="records")
    return {"mean_scores": mean_scores, "per_item": per_item}


def main() -> None:
    print("=" * 60)
    print("RAG EVALUATION RUN")
    print("=" * 60)
    init_langfuse()

    golden = load_golden_set(GOLDEN_SET_PATH)
    print(f"\nLoaded {len(golden)} questions from {GOLDEN_SET_PATH}")

    print("\n[1/3] Running RAG system on each question...")
    retriever = HybridRetriever()
    rows = []
    for i, item in enumerate(golden, start=1):
        print(f"  [{i}/{len(golden)}] {item['id']}: {item['question'][:60]}...")
        rows.append(run_system_on_item(item, retriever))
        flush_langfuse()
    print(f"  Done. Avg latency: "
          f"{sum(r['latency_retrieve_s'] + r['latency_generate_s'] for r in rows) / len(rows):.1f}s")

    print("\n[2/3] Computing retrieval recall@k...")
    recall = compute_retrieval_recall(rows)
    print(f"  Retrieval recall@{TOP_K}: {recall['recall_at_k']:.1%} "
          f"({recall['hits']}/{recall['total']})")

    print("\n[3/3] Computing RAGAS LLM-as-judge metrics...")
    ragas_results = run_ragas(rows)
    print("\nRAGAS mean scores:")
    for metric, score in ragas_results["mean_scores"].items():
        score_str = f"{score:.3f}" if score is not None else "N/A"
        print(f"  {metric:25s} {score_str}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "config": {
            "top_k": TOP_K,
            "retrieval_strategy": "hybrid_bm25_vector_rrf",
            "llm_model": "llama3.1:8b",
            "embed_model": "nomic-embed-text",
            "judge_llm": JUDGE_LLM,
            "judge_embed": JUDGE_EMBED,
            "num_questions": len(rows),
        },
        "avg_latency": {
            "retrieve_s": sum(r["latency_retrieve_s"] for r in rows) / len(rows),
            "generate_s": sum(r["latency_generate_s"] for r in rows) / len(rows),
        },
        "retrieval_recall": recall,
        "ragas_mean_scores": ragas_results["mean_scores"],
        "ragas_per_item": ragas_results["per_item"],
        "system_outputs": rows,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to {out_path}")
    flush_langfuse()


if __name__ == "__main__":
    main()
