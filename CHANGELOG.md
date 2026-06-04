# Changelog

Day-by-day progression of the project. Each "Day" is one focused
sub-phase that builds on the previous one.

All metric numbers below are real, measured on the 10-question golden
set in `evals/golden_set.jsonl`.

---

## Phase 1 — Build the system

### Day 1 — Project setup
- Initialized `rich-dad-rag/` with `uv`, dependency management,
  `.gitignore`, and project structure (`src/`, `data/`, `evals/`,
  `docs/`).
- Tech stack chosen: Ollama (local LLM + embeddings), ChromaDB,
  Streamlit, Langfuse, RAGAS.
- Pulled `nomic-embed-text` (274 MB) and `llama3.1:8b` (4.9 GB) via
  Ollama.
- Initialized git repo, created GitHub remote.

### Day 2 — Hello-world RAG (ingestion + vector search)
- Built `src/ingest/` modules: `loader.py` (pypdf), `chunker.py`
  (overlapping ~500-char chunks with boundary detection),
  `embedder.py` (nomic-embed-text), `store.py` (ChromaDB).
- Ingested *Rich Dad Poor Dad* PDF: 1057 chunks.
- Built `search_test.py` to verify vector similarity search works.

### Day 3 — End-to-end RAG
- Built `src/generate/prompt.py` (context formatter + chat-message
  builder) and `src/generate/llm.py` (Ollama LLM wrapper).
- Built `src/retrieval/retriever.py` with a clean `Retriever`
  interface returning `{chunk_id, text, page_num, source, distance}`.
- Wired `rag.py` (orchestration) and `chat.py` (interactive CLI).
- First end-to-end retrieve → prompt → generate cycle on real
  questions.

### Day 4 — Langfuse observability
- Added `src/observability/tracer.py` for Langfuse client init.
- Wrapped every component (`embed_text`, `retrieve`, `llm_generate`,
  `rag_query`) with `@observe` decorators.
- Captured token usage, latency, and full traces per query.
- Verified spans appear in Langfuse cloud UI.

### Day 5 — Streamlit UI
- Built `app.py` with chat history (`st.session_state`), top_k
  slider, source expander showing retrieved chunks, and Retrieve /
  Generate / Total latency metrics.
- First demoable browser UI.

### Day 5.5 — Multi-document support
- Generalized ingestion CLI (`ingest.py --all` or specific paths).
- Made system prompt book-agnostic; citations now include source
  filename, not just page number.
- Sidebar "Library" panel shows loaded books and chunk counts.

---

## Phase 2 — Measure

### Day 6 — Golden set + RAGAS baseline
- Hand-curated 10 Q&A pairs in `evals/golden_set.jsonl`, mixing
  *Rich Dad Poor Dad* with a resume PDF to stress multi-document
  retrieval.
- Built `src/eval/run.py` with retrieval recall@k + RAGAS
  LLM-as-judge metrics (faithfulness, answer_relevancy,
  context_precision).
- Used `RunConfig(timeout=600, max_workers=1)` to keep RAGAS from
  timing out on CPU.
- **Baseline (vector-only) metrics:**
  - Recall@5: **70.0%**
  - Faithfulness: **0.719**
  - Answer relevancy: **0.649**
  - Context precision: **0.580**

---

## Phase 3 — Improve retrieval

### Day 7 — Hybrid retrieval (BM25 + vector + RRF)
- Built `src/retrieval/bm25_retriever.py` (rank-bm25, word-character
  tokenizer for PDF-extraction edge cases) and
  `src/retrieval/hybrid_retriever.py` (Reciprocal Rank Fusion with
  k=60).
- Added `src/eval/compare.py` for side-by-side metric deltas between
  result files.
- **After hybrid:**
  - Recall@5: 70.0% → **100.0%** (+30.0 pp)
  - Faithfulness: 0.719 → **0.850** (+0.131)
  - Answer relevancy: 0.649 → **0.820** (+0.171)
  - Context precision: 0.580 → 0.719 (+0.139)

### Day 8 — Cross-encoder reranking
- Built `src/rerank/cross_encoder.py` using
  `BAAI/bge-reranker-v2-m3` from `sentence-transformers`.
- Composed into `src/retrieval/reranked_retriever.py`:
  hybrid fetches 20 candidates → cross-encoder reranks down to
  top-5.
- **After rerank:**
  - Recall@5: 100.0% (held)
  - Faithfulness: 0.850 → **0.842** (~flat)
  - Answer relevancy: 0.820 → **0.832** (+0.012)
  - Context precision: 0.719 → **0.895** (+0.176)

---

## Phase 4 — Make it trustworthy

### Day 9 — Citation enforcement
- Built `src/generate/citation_validator.py` with regex-based
  citation parsing (`[filename.pdf, page N]`) and validation against
  retrieved chunks.
- Modified `rag.py` to enforce the pipeline:
  generate → validate → retry once → safe refusal.
- Hardened the system prompt: every factual claim must be followed
  by a citation; fake page numbers explicitly forbidden; a literal
  refusal string is required when context is insufficient.
- Added Streamlit citation status badges
  (✅ verified / 🛡️ refused / ⚠️ unverified).
- **After citation enforcement:**
  - Recall@5: 100.0% (held)
  - Faithfulness: **0.833** (~flat vs Day 8)
  - Answer relevancy: **0.812** (~flat)
  - Context precision: **0.895** (held)
  - Citation valid rate: **100.0%**
  - **Hallucinated citation rate: 0.0%**
  - Retry rate: 10% (1 of 10 questions caught + corrected)
  - Forced-fallback rate: 0%

---

## Phase 5 — Protect it

### Day 10 — CI regression gating
- Added pytest unit tests for citation validator, chunker, and RRF
  math (21 tests total, all passing in ~3.5s).
- Created `evals/baseline.json` — frozen snapshot of Day 9 metrics.
- Built `src/eval/check_baseline.py` — exit-code-based regression
  gate with per-metric thresholds (5pp drop allowed for quality
  metrics, +2.0s for retrieve latency, +30s for generate latency).
- Modified `src/eval/run.py` to auto-write `evals/latest_summary.json`
  after every eval, so CI always has a current artifact to check.
- Added `.github/workflows/ci.yml` with two jobs:
  - `Tests + Lint` — pytest + ruff (non-blocking)
  - `Eval Baseline Gate` — runs `check_baseline.py`
- Documented the two-tier workflow (fast = CI, slow = local manual
  eval) in `CONTRIBUTING.md`.
- **Verified the gate works** both locally (tampered summary →
  exit 1) and on GitHub (PR with regression → red `Eval Baseline
  Gate` job, no merge).

---

## Phase 6 — Polish & ship

### Day 11 — Visual assets & README rewrite
- Captured all 6 planned screenshots in `docs/screenshots/`:
  * `01_main_ui.png` — full Streamlit hero (sidebar + answer + verified badge)
  * `02_verified_answer.png` — close-up of the cited answer
  * `03_refusal.png` — "What is the capital of France?" → 🛡️ refused
  * `04_langfuse_trace.png` — full trace tree (rag_query → reranked_retrieve → hybrid → rerank → llm_generate) with the right-panel detail view for the `rerank` span
  * `05_ci_passing.png` — GitHub Actions list showing all CI runs
    (greens + the deliberate red regression demo)
  * `06_ci_blocking.png` — drill-in of the failed `Eval Baseline Gate` job, showing the metrics table with `retrieval_recall_at_k` FAIL and the `REGRESSION: 1 metric(s) outside threshold` line
- Added a mermaid architecture diagram (renders natively on GitHub —
  no separate SVG file needed).
- Rewrote `README.md` with hero image, CI badge, before/after metrics
  table (real numbers from each eval result), quickstart, demo
  questions, tech-stack table, and project structure tree.
- Moved the day-by-day journal out of `README.md` and into this
  `CHANGELOG.md` so the README stays signal-dense.
- Added `docs/demo_link.md` as a placeholder for the 90-second screen
  recording (to be done in Day 16 when the blog post goes out).
- Committed and pushed; verified all screenshots and the mermaid
  diagram render correctly on GitHub.

### Day 12+ (planned)
- **Day 12**: Multi-format ingestion (TXT/MD/DOCX/code).
- **Day 13**: Browser file-uploader UI.
- **Day 14**: Backend abstraction (local Ollama ↔ cloud Groq).
- **Day 15**: Deploy to Hugging Face Spaces (public URL).
- **Day 16**: Demo recording + blog post + LinkedIn share.
