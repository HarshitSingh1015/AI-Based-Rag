# Contributing

This project uses a two-tier CI strategy:

## Tier 1 — Fast checks (run in GitHub Actions on every PR)
- Unit tests via `pytest`
- Linting via `ruff` (non-blocking)
- Baseline metrics gate via `src/eval/check_baseline.py`

These run automatically. PRs that fail the baseline gate cannot be merged.

## Tier 2 — Full evaluation (run manually before significant RAG changes)
- Full RAGAS eval via `uv run python -m src.eval.run`
- Takes ~90-180 minutes on CPU
- Saves results to `evals/results/eval_<timestamp>.json`
- Also auto-writes `evals/latest_summary.json` (which CI reads)

## Workflow for changes that affect retrieval, generation, or citation logic

1. Make your code change.
2. Run unit tests locally: `uv run pytest tests/ -v`
3. Run the full eval locally: `uv run python -m src.eval.run`
4. Inspect `evals/latest_summary.json` — confirm metrics are acceptable.
5. Compare against baseline locally:
   `uv run python -m src.eval.check_baseline`
6. Commit code + `evals/latest_summary.json` together.
7. Push and open a PR. GitHub Actions will re-verify.

## Workflow for changes that DON'T affect quality

(docs, UI tweaks, refactors with no behavior change)

1. Make your code change.
2. Run unit tests locally.
3. Push and open a PR. The baseline gate uses the existing
   `evals/latest_summary.json` so it should still pass.

## Updating the baseline

When you make an INTENTIONAL improvement that beats the current baseline:

1. Run the full eval.
2. Verify the new metrics are real (re-run if scores look suspicious).
3. Copy `evals/latest_summary.json` to `evals/baseline.json`.
4. Update the `snapshot_date` and `source_result_file` fields in
   `baseline.json`.
5. Commit the new baseline with a clear message explaining the improvement.

## Running tests locally

```bash
uv run pytest tests/ -v
uv run ruff check src/ tests/
```
