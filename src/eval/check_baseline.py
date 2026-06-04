"""Compare the latest eval summary against the frozen baseline.

Exit codes:
    0 = all metrics within thresholds (PASS)
    1 = at least one regression detected (FAIL)

Usage:
    uv run python -m src.eval.check_baseline                       # uses evals/latest_summary.json
    uv run python -m src.eval.check_baseline <summary_path>        # explicit path
"""
import json
import sys
from pathlib import Path

BASELINE_PATH = Path("evals/baseline.json")
DEFAULT_SUMMARY = Path("evals/latest_summary.json")

# Thresholds for each metric.
# - Negative value means "lower is worse" — current must be >= baseline + value
#   (e.g., -0.05 allows up to 5pp drop)
# - Positive value means "higher is worse" — current must be <= baseline + value
#   (e.g., +0.05 allows up to 5pp rise)
THRESHOLDS = {
    "retrieval_recall_at_k": -0.05,
    "ragas_faithfulness": -0.05,
    "ragas_answer_relevancy": -0.05,
    "ragas_context_precision": -0.05,
    "citation_valid_rate": -0.05,
    "citation_hallucinated_rate": +0.05,
    "avg_latency_retrieve_s": +2.0,
    "avg_latency_generate_s": +30.0,
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_metric(d: dict, key: str) -> float | None:
    return d.get("metrics", {}).get(key)


def main() -> int:
    if not BASELINE_PATH.exists():
        print(f"FAIL: Baseline file not found at {BASELINE_PATH}")
        return 1

    summary_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SUMMARY
    if not summary_path.exists():
        print(f"FAIL: Summary file not found at {summary_path}")
        return 1

    baseline = load_json(BASELINE_PATH)
    current = load_json(summary_path)

    print(f"Baseline: {BASELINE_PATH}")
    print(f"Current:  {summary_path}")
    print()
    print(f"{'Metric':<32} {'Baseline':>10} {'Current':>10} "
          f"{'Delta':>10} {'Allowed':>10} {'Status':>8}")
    print("-" * 86)

    failures: list[tuple] = []
    for metric, allowed_delta in THRESHOLDS.items():
        b_val = get_metric(baseline, metric)
        c_val = get_metric(current, metric)

        if b_val is None or c_val is None:
            print(f"{metric:<32} {'N/A':>10} {'N/A':>10} "
                  f"{'-':>10} {'-':>10} {'SKIP':>8}")
            continue

        delta = c_val - b_val
        if allowed_delta < 0:
            passed = delta >= allowed_delta
        else:
            passed = delta <= allowed_delta

        status = "PASS" if passed else "FAIL"
        print(f"{metric:<32} {b_val:>10.3f} {c_val:>10.3f} "
              f"{delta:>+10.3f} {allowed_delta:>+10.3f} {status:>8}")
        if not passed:
            failures.append((metric, b_val, c_val, delta, allowed_delta))

    print()
    if failures:
        print(f"REGRESSION: {len(failures)} metric(s) outside threshold:")
        for m, b, c, d, a in failures:
            print(f"  - {m}: {b:.3f} -> {c:.3f} "
                  f"(delta {d:+.3f}, allowed {a:+.3f})")
        print("\nThe PR should NOT be merged until these are fixed or the baseline is intentionally updated.")
        return 1

    print("OK: All metrics within acceptable bounds. No regression detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
