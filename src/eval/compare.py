"""Compare two eval result files side by side.

Usage:
    uv run python -m src.eval.compare                    # auto-picks latest two
    uv run python -m src.eval.compare <baseline> <new>   # explicit files
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = Path("evals/results")


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt_delta(delta: float, is_pct: bool = False) -> str:
    sign = "+" if delta >= 0 else ""
    if is_pct:
        return f"{sign}{delta:.1%}"
    return f"{sign}{delta:.3f}"


def compare(baseline_path: Path, new_path: Path) -> None:
    b = load(baseline_path)
    n = load(new_path)

    print(f"\nBaseline: {baseline_path.name}")
    print(f"New:      {new_path.name}\n")

    print("=" * 72)
    print(f"{'Metric':<28} {'Baseline':>12} {'New':>12} {'Delta':>14}")
    print("=" * 72)

    b_recall = b["retrieval_recall"]["recall_at_k"]
    n_recall = n["retrieval_recall"]["recall_at_k"]
    print(f"{'Retrieval recall@k':<28} {b_recall:>12.1%} {n_recall:>12.1%} "
          f"{fmt_delta(n_recall - b_recall, is_pct=True):>14}")

    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        b_val = b["ragas_mean_scores"].get(metric)
        n_val = n["ragas_mean_scores"].get(metric)
        if b_val is None or n_val is None:
            print(f"{metric:<28} {'N/A':>12} {'N/A':>12} {'N/A':>14}")
            continue
        print(f"{metric:<28} {b_val:>12.3f} {n_val:>12.3f} "
              f"{fmt_delta(n_val - b_val):>14}")

    print("-" * 72)

    b_lat_r = b["avg_latency"]["retrieve_s"]
    n_lat_r = n["avg_latency"]["retrieve_s"]
    print(f"{'Avg retrieve latency (s)':<28} {b_lat_r:>12.2f} {n_lat_r:>12.2f} "
          f"{fmt_delta(n_lat_r - b_lat_r):>14}")

    b_lat_g = b["avg_latency"]["generate_s"]
    n_lat_g = n["avg_latency"]["generate_s"]
    print(f"{'Avg generate latency (s)':<28} {b_lat_g:>12.2f} {n_lat_g:>12.2f} "
          f"{fmt_delta(n_lat_g - b_lat_g):>14}")

    print("\n" + "=" * 72)
    print("Per-question retrieval hits (did expected source appear in top-k?)")
    print("=" * 72)
    print(f"{'ID':<14} {'Baseline':<12} {'New':<12} {'Change':<20}")
    print("-" * 72)

    b_items = {item["id"]: item for item in b["retrieval_recall"]["per_item"]}
    n_items = {item["id"]: item for item in n["retrieval_recall"]["per_item"]}

    improvements = 0
    regressions = 0
    for qid in b_items:
        b_hit = b_items[qid]["hit"]
        n_hit = n_items.get(qid, {}).get("hit", False)
        b_str = "HIT" if b_hit else "MISS"
        n_str = "HIT" if n_hit else "MISS"
        change = ""
        if b_hit and not n_hit:
            change = "REGRESSED"
            regressions += 1
        elif n_hit and not b_hit:
            change = "IMPROVED"
            improvements += 1
        elif b_hit and n_hit:
            change = "stable (hit)"
        else:
            change = "stable (miss)"
        print(f"{qid:<14} {b_str:<12} {n_str:<12} {change:<20}")

    print("-" * 72)
    print(f"Net: {improvements} improvement(s), {regressions} regression(s)")
    print()


def main() -> None:
    if len(sys.argv) == 3:
        compare(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        results = sorted(
            RESULTS_DIR.glob("eval_*.json"),
            key=lambda p: p.stat().st_mtime,
        )
        if len(results) < 2:
            print(f"Need at least 2 result files in {RESULTS_DIR}/")
            sys.exit(1)
        compare(results[-2], results[-1])


if __name__ == "__main__":
    main()
