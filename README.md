# Multi-Doc RAG (started with Rich Dad Poor Dad)

A 100% local, free RAG (Retrieval-Augmented Generation) system that lets me ask
questions about *Rich Dad Poor Dad* by Robert Kiyosaki, with answers grounded
in the actual book text and proper source citations.

This is my first AI engineering project, built as a learning exercise and
portfolio piece demonstrating production-grade RAG patterns.

## Goals (v1)
- Hybrid retrieval (BM25 + vector search with RRF fusion)
- Cross-encoder reranking
- Citation enforcement (LLM must cite source chunks; answers without valid citations are rejected)
- Full observability via Langfuse (tracing, latency, simulated cost)
- CI-gated evaluation pipeline (regression-blocking on quality drops)

## Future versions (Plan A -> Plan C roadmap)
- v2: Add study notes (markdown/txt)
- v3: Add research papers (PDF) + company docs (DOCX)
- v4: Add codebase support (tree-sitter chunking)
- v5: Add legal contracts (clause-level chunking)

## Stack
- **LLM**: `llama3.1:8b` via Ollama (local)
- **Embeddings**: `nomic-embed-text` via Ollama (local)
- **Vector DB**: ChromaDB (file-based, local)
- **BM25**: `rank-bm25` (Python)
- **Reranker**: `bge-reranker-v2-m3` (via sentence-transformers, local)
- **Observability**: Langfuse Cloud (free tier)
- **Eval**: RAGAS (with local Ollama LLM-as-judge)
- **UI**: Streamlit
- **Package manager**: uv

## Example questions the v1 system should answer well
1. What's the difference between an asset and a liability according to Kiyosaki?
2. Who is the "rich dad" and who is the "poor dad"?
3. What does Kiyosaki mean by "the rat race"?
4. What's his view on traditional education vs. financial education?
5. What does he say about working for money vs. having money work for you?
6. What are the six main lessons in the book?
7. What does Kiyosaki say about fear and greed in financial decisions?

## Sample Q&A (Day 3 baseline)

### Q: What is the difference between an asset and a liability?
**A**: According to Robert Kiyosaki, the difference between an asset and a liability is that:

* An asset puts money in my pocket (Source 1 — page 61, Source 5 — page 63).
* A liability takes money out of my pocket (Source 1 — page 61, Source 5 — page 63).

Sources: [page 61, page 63]

_Latency: retrieve 60ms, generate 108.4s_

### Q: Who is the rich dad and who is the poor dad?
**A**: The rich dad is Robert Kiyosaki's best friend's father, a high school dropout who was wealthy. The poor dad is Robert Kiyosaki's own biological father, who was highly educated but struggled financially.

Sources: [page 229, page 20]

_Latency: retrieve 70ms, generate 82.7s_

> Note: This is the unoptimized baseline (vector-only retrieval, no reranking, no
> citation validation, llama3.1:8b on CPU). Hybrid retrieval, reranking, and
> citation enforcement come in later phases and will improve both accuracy and
> source faithfulness. The ~90s generate latency is CPU-bound — would drop
> dramatically on GPU.

## Observability Baseline (Day 4)

All RAG queries are traced via Langfuse. Each query produces a nested trace
tree:

```
rag_query                          (root span — full user query)
├── retrieve                       (Retriever.retrieve method)
│   └── embed_text                 (nomic-embed-text on the question)
└── llm_generate                   (llama3.1:8b — tagged as "generation")
```

After running `rag.py` with the 5 test questions, baseline numbers from the
Langfuse dashboard:

| Metric | Value |
|--------|-------|
| Avg retrieve latency (incl. query embed) | ~0.67s |
| Avg LLM generation latency | ~95.5s |
| Avg total query latency | ~96.2s |
| Avg input tokens per LLM call | ~900 (5 chunks × ~150 tokens + system + question) |
| Avg output tokens per LLM call | ~80 (concise grounded answers) |
| LLM model | llama3.1:8b (local, Ollama) |
| Embedding model | nomic-embed-text (local, Ollama) |

![Langfuse trace example](docs/langfuse_trace.png)

These are the **pre-optimization baselines**. Future phases will add hybrid
retrieval, reranking, citation enforcement, and CI-gated evals — each
measured against these numbers. The "Cost" column in the Langfuse dashboard
will show $0 because Langfuse doesn't ship pricing for local Ollama models;
that's expected.

## Web UI (Day 5)

A clean Streamlit interface for browser-based interaction.

![App screenshot](docs/app_screenshot.png)

**Run locally:**

```bash
uv run streamlit run app.py
```

Then open http://localhost:8501 in your browser.

Features:
- Chat-style Q&A with persistent history within a session
- Adjustable `top_k` retrieval slider in the sidebar
- Expandable source panel showing each retrieved chunk + page number + cosine distance
- Per-query latency metrics (retrieve / generate / total)
- Every query is still traced in Langfuse — open the dashboard alongside

## Adding More Books (Day 5.5)

The system supports any number of PDF books, papers, or notes. To add more:

1. Drop any PDF into `data/raw/`
2. Run:
   ```bash
   uv run python ingest.py data/raw/<your_file>.pdf
   ```
3. Or ingest everything in the folder at once:
   ```bash
   uv run python ingest.py --all
   ```
4. Refresh the Streamlit app — the sidebar will list the new book.

All books share one ChromaDB collection. Retrieval picks the most relevant
chunks regardless of source, and citations include the source filename
and page number.

**Note**: Embedding takes ~5-30 minutes per book on CPU. Be patient.

## Baseline Evaluation (Day 6 — Phase 3)

10-question hand-curated golden set covering both books in the library
(7 Rich Dad Poor Dad + 3 resume questions). All metrics computed using
local Ollama as both the system LLM AND the LLM-as-judge.

| Metric | Score | Notes |
|--------|-------|-------|
| Retrieval recall@5 | **70.0%** (7/10) | Did at least one chunk come from the expected source? |
| RAGAS faithfulness | 0.719 | Is the answer grounded in the retrieved context? |
| RAGAS answer relevancy | 0.649 | Does the answer address the question? |
| RAGAS context precision | 0.580 | Are the retrieved chunks relevant to the question? |
| Avg retrieve latency | 0.15s | Vector search only — fast |
| Avg generate latency | 76.4s | CPU inference is the bottleneck |

### Known Baseline Gap

Resume-related questions (`resume_01`, `resume_02`, `resume_03`) currently
retrieve **0 chunks from the resume**, falling back to Rich Dad Poor Dad
content. Per-question breakdown:

| ID | Expected source | Retrieved from | Hit? |
|----|----------------|----------------|------|
| rdp_01 … rdp_07 | rich_dad_poor_dad.pdf | rich_dad_poor_dad.pdf | ✅ 7/7 |
| resume_01 | harshit_resume.pdf | rich_dad_poor_dad.pdf | ❌ |
| resume_02 | harshit_resume.pdf | rich_dad_poor_dad.pdf | ❌ |
| resume_03 | harshit_resume.pdf | rich_dad_poor_dad.pdf | ❌ |

This is a known limitation of pure vector retrieval when one document
heavily dominates the corpus (1057 book chunks vs 9 resume chunks — 99.2%
of the index is one source). For "soft" semantic queries like
*"educational background"* or *"programming languages,"* the book's
broader vocabulary outweighs the resume's much smaller surface area.

**Phase 4 (hybrid retrieval with BM25)** is expected to close this gap.
BM25 rewards exact term overlap, so a query like *"What programming
languages does Harshit Singh know?"* will heavily favor the resume chunks
that contain the literal token "Harshit." The current numbers are the
deliberate baseline we will measure against.

Implementation note: RAGAS 0.2.x defaults to a 180s per-job timeout and
high parallelism, both of which break with a slow local LLM judge. The
runner uses `RunConfig(timeout=600, max_workers=1, max_retries=2)` to
serialize judge calls — this is why the full eval takes ~2 hours on
CPU.

Latest results: `evals/results/eval_20260603_122556.json` (gitignored;
re-run anytime with `uv run python -m src.eval.run`).

## Phase 4 — Retrieval Improvements

Three retrieval strategies, same 10-question golden set, measured against
the Day 6 baseline.

| Stage | Retrieval recall@5 | Faithfulness | Answer relevancy | Context precision | Avg retrieve (s) | Avg generate (s) |
|-------|---------------------|--------------|------------------|-------------------|------------------|------------------|
| **Day 6 baseline** (vector only) | 70.0% | 0.719 | 0.649 | 0.580 | 0.15 | 76.4 |
| **Day 7 hybrid** (BM25 + vector + RRF) | 100.0% | 0.850 | 0.820 | 0.719 | 0.17 | 105.8 |
| **Day 8 + rerank** (cross-encoder bge-reranker-v2-m3) | **100.0%** | **0.842** | **0.832** | **0.895** | **11.78** | **101.4** |

**End-to-end improvement (baseline → Day 8):**
- Retrieval recall@5: 70.0% → 100.0% (**+30.0%**)
- Context precision: 0.580 → 0.895 (**+0.315**)
- Faithfulness: 0.719 → 0.842 (**+0.123**)
- Answer relevancy: 0.649 → 0.832 (**+0.183**)
- Latency cost: +11.6 seconds retrieve (cross-encoder on CPU)

**Incremental gain from reranking alone (Day 7 → Day 8):**
- Recall@5: 100% → 100% (already saturated by hybrid)
- Context precision: 0.719 → 0.895 (**+0.176**) ← the reranker's true contribution
- Answer relevancy: 0.820 → 0.832 (+0.013)
- Faithfulness: 0.850 → 0.842 (-0.008, within RAGAS noise)
- Generate latency: 105.8s → 101.4s (-4.4s, noise — the LLM sometimes responds faster on better-curated context)

### Per-question retrieval hits

| Question ID | Baseline | Hybrid | + Rerank |
|-------------|----------|--------|----------|
| rdp_01      | HIT  | HIT  | HIT  |
| rdp_02      | HIT  | HIT  | HIT  |
| rdp_03      | HIT  | HIT  | HIT  |
| rdp_04      | HIT  | HIT  | HIT  |
| rdp_05      | HIT  | HIT  | HIT  |
| rdp_06      | HIT  | HIT  | HIT  |
| rdp_07      | HIT  | HIT  | HIT  |
| resume_01   | MISS | HIT  | HIT  |
| resume_02   | MISS | HIT  | HIT  |
| resume_03   | MISS | HIT  | HIT  |

### Why each stage helps
- **Vector only**: good semantic similarity, but misses exact-keyword matches
  (proper names, IDs, technical terms) and struggles with corpus imbalance —
  the resume's 9 chunks were drowned out by the book's 1057.
- **+ BM25 (Day 7)**: keyword matching adds rare-term sensitivity ("Harshit",
  "22UEC125"), bringing previously-unreachable chunks into the candidate
  pool. Recall@5 saturates at 100%.
- **+ Cross-encoder rerank (Day 8)**: hybrid already had 100% hit-rate, but
  chunk *ordering* inside top-5 was still mediocre. The reranker re-scores 20
  candidates with a model that considers query + document jointly,
  putting truly relevant chunks at the top — context_precision jumps
  +0.176 even though recall doesn't change.

### Architecture today

```
RerankedRetriever
├── HybridRetriever (fetch_k=20)
│   ├── VectorRetriever (semantic similarity, nomic-embed-text)
│   └── BM25Retriever  (keyword exact-match, rank-bm25)
│       fused via Reciprocal Rank Fusion (k=60)
└── CrossEncoderReranker (BAAI/bge-reranker-v2-m3, ~570MB local)
    re-scores 20 candidates → returns top 5
```

The BM25 tokenizer uses `re.findall(r"\w+", text.lower())` rather than a
plain whitespace split, so PDF-extraction artifacts (e.g., `Roll No.:
22UEC125/envel⌢pe...`) still surface the embedded alphanumeric ID.

### Headline behavior change

The query *"What programming languages does Harshit know?"* on Day 7 (hybrid)
returned *"I don't have enough information"* because, despite both resume
chunks being in the candidate pool, they were drowned out by 18 book chunks
in the top-5 prompt. On Day 8 (hybrid + rerank), the same query correctly
answers with `C++, JavaScript, TypeScript, HTML/CSS` and cites
`harshit_resume.pdf`.

Generate any pairwise comparison with:
```bash
uv run python -m src.eval.compare <file_a> <file_b>
```

Latest results: `evals/results/eval_20260603_190356.json`

## Phase 5 — Citation Enforcement (Day 9)

Every answer now either cites valid retrieved sources or honestly refuses.
The system parses every `[filename.pdf, page N]` citation in the LLM's
response, verifies each one against the actually-retrieved chunks, and:

1. If all citations are valid: returns the answer.
2. If any citation is hallucinated: regenerates ONCE with a stricter
   reminder prompt.
3. If still invalid: returns a safe "I don't have enough information"
   refusal instead of risking hallucination.

### Citation enforcement metrics (Day 9 vs Day 8)

| Metric | Day 8 | Day 9 | Delta |
|--------|-------|-------|-------|
| Valid answer rate | N/A (unmeasured) | **100.0%** | new metric |
| Honest refusal rate | N/A | 0.0% | new metric |
| Retry rate | N/A | 10.0% | new metric |
| Forced-fallback rate | N/A | 0.0% | new metric |
| Hallucinated citation rate | N/A | **0.0%** | new metric |
| Faithfulness | 0.842 | 0.833 | -0.008 (RAGAS noise) |
| Answer relevancy | 0.832 | 0.812 | -0.021 (stricter answers) |
| Context precision | 0.895 | 0.895 | +0.000 (retrieval unchanged) |
| Avg generate latency (s) | 101.4 | 125.4 | +24.0 (1 retry + variance) |

**Reading the numbers:** retrieval is structurally unchanged from Day 8, so
recall@5 stays at 100% and context_precision stays at 0.895. The RAGAS
scores drift down a fraction (within their 5-10% noise band) because
citation-constrained answers tend to be shorter and more literal, which
the judge LLM sometimes rates as less "fluent." The new value lives in
the citation metrics: **100% of answers carried verified citations and
zero of them hallucinated a source or page that wasn't actually
retrieved.** The retry caught the single failure case and corrected it
without ever falling back to the safe refusal.

### Full project story (Day 6 baseline → Day 9)

| Metric | Baseline | Day 9 | Delta |
|--------|----------|-------|-------|
| Retrieval recall@5 | 70.0% | **100.0%** | **+30.0%** |
| Faithfulness | 0.719 | 0.833 | +0.115 |
| Answer relevancy | 0.649 | 0.812 | +0.162 |
| Context precision | 0.580 | 0.895 | +0.315 |
| Hallucinated citation rate | unmeasured | **0.0%** | — |
| Honest refusal capability | none | yes | — |
| Avg retrieve latency (s) | 0.15 | 13.41 | +13.26 |
| Avg generate latency (s) | 76.4 | 125.4 | +49.0 |

The system has transitioned from "demo-quality" (Day 6) to "trustworthy"
(Day 9): every answer is either backed by verified citations or
transparently refused. Out-of-corpus questions like *"What is the capital
of France?"* now correctly refuse instead of hallucinating from the LLM's
training knowledge.

### Architecture today

```
answer()
├── RerankedRetriever (hybrid + cross-encoder rerank)
├── generate (LLM)
├── validate_citations  ──→  parse [filename.pdf, page N], check each
│                            against retrieved chunks
├── if invalid: retry once with stronger reminder prompt
└── if still invalid: replace with safe refusal
```

Latest results: `evals/results/eval_20260604_133832.json`

## Phase 6 — CI Regression Gating (Day 10)

[![CI](https://github.com/HarshitSingh1015/AI-Based-Rag/actions/workflows/ci.yml/badge.svg)](https://github.com/HarshitSingh1015/AI-Based-Rag/actions/workflows/ci.yml)

Every pull request runs through GitHub Actions:
1. **Unit tests** — pytest on the deterministic components (citation
   validator, chunker, RRF math).
2. **Lint** — ruff (non-blocking warnings).
3. **Baseline gate** — `src/eval/check_baseline.py` compares the
   committed `evals/latest_summary.json` against `evals/baseline.json`
   and **fails the build** if any metric regresses past its threshold.

### Gated metrics and thresholds

| Metric | Allowed change |
|--------|----------------|
| Retrieval recall@5 | may drop at most 5pp |
| RAGAS faithfulness | may drop at most 0.05 |
| RAGAS answer relevancy | may drop at most 0.05 |
| RAGAS context precision | may drop at most 0.05 |
| Citation valid rate | may drop at most 5pp |
| Hallucinated citation rate | may rise at most 5pp |
| Avg retrieve latency | may grow by at most 2.0s |
| Avg generate latency | may grow by at most 30s |

A change that exceeds ANY threshold blocks the PR until either:
- The regression is fixed, OR
- The maintainer intentionally updates the baseline.

See `CONTRIBUTING.md` for the full workflow.

### Two-tier strategy

| Tier | What runs | When | Time |
|------|-----------|------|------|
| **Fast** (GitHub Actions) | pytest + ruff + baseline gate | Every PR | ~1-3 min |
| **Slow** (manual, local) | Full RAGAS eval | Before major RAG changes | ~90-180 min |

The slow tier writes `evals/latest_summary.json`; the fast tier reads
it. This keeps PR feedback cycles short while still protecting the
RAG-quality metrics.

## Progress log
- [x] **Day 1**: Project setup, Ollama installed, models pulled, folder structure, deps installed
- [x] **Phase 1**: Hello World RAG (basic vector search + LLM)
- [x] **Phase 2**: Observability (Langfuse tracing, latency, token usage)
- [x] **Day 5**: Streamlit web UI ← demoable!
- [x] **Day 5.5**: Multi-document support via CLI
- [x] **Phase 3**: Evaluation foundation (golden set + RAGAS) — 10 questions, baseline captured
- [x] **Phase 4**: Hybrid retrieval + reranking (both done)
- [x] **Phase 5**: Citation enforcement (parse, validate, retry, refuse)
- [x] **Phase 6**: CI regression gating (pytest + baseline check on every PR)
- [ ] **Day 13**: Multi-format support + browser upload UI
- [ ] **Phase 7**: Deploy to Hugging Face Spaces

## Setup verification (Day 1)
- [x] Ollama installed: `ollama --version`
- [x] LLM pulled: `ollama list` shows `llama3.1:8b`
- [x] Embedding model pulled: `ollama list` shows `nomic-embed-text`
- [x] `OLLAMA_KEEP_ALIVE=30m` set
- [x] Python 3.11+ installed
- [x] `uv` installed
- [x] Project folder structure created
- [x] Initial deps installed (ollama, chromadb, pypdf, python-dotenv)
- [x] Git repo initialized
- [x] Langfuse Cloud account created (for Phase 2)
- [x] `rich_dad_poor_dad.pdf` placed in `data/raw/`

## Daily journal
### Day 1 — 2026-06-02
- Installed Ollama and pulled llama3.1:8b + nomic-embed-text
- Set up project skeleton with uv
- Tested first LLM call from terminal
- Goal for Day 2: write PDF loader and basic chunker

### Day 2 — 2026-06-02
- Built PDF loader, chunker, embedder, and ChromaDB store
- Ingested full book: 227 pages -> 1057 chunks -> all embedded -> stored
- Embedding pipeline took ~4.4 min (~252 ms/chunk on CPU)
- Search test passed: 5/5 queries returned relevant chunks (4 excellent, 1 decent)
- Noted for Phase 4: vector-only retrieval is weak on rare specific phrases (e.g., "rat race") — hybrid (BM25 + vector) will fix this
- Goal for Day 3: wire generation (LLM call) into the pipeline for first end-to-end RAG

### Day 3 — 2026-06-02
- Built LLM wrapper (llama3.1:8b), prompt template, and Retriever class
- First end-to-end RAG working: question -> retrieve -> generate -> answer
- Avg latency: 0.06s retrieve, 89.18s generate (CPU)
- 5/5 questions answered; 4 excellent + 1 OK (rat race — weak retrieval flowed through)
- Captured 2 sample Q&A pairs in README
- Goal for Day 4: improve retrieval (add BM25 + hybrid) OR add Streamlit UI (TBD)

### Day 4 — 2026-06-02
- Added Langfuse observability to embed_text, retrieve, llm_generate, and rag_query
- Created singleton Langfuse client in src/observability/tracer.py
- Verified nested trace tree in Langfuse Cloud with token usage on llm_generate
- Captured baseline latency + token-usage metrics in README
- Saved trace screenshot to docs/langfuse_trace.png (user-captured)
- Goal for Day 5: build Streamlit UI for browser-based interaction

### Day 5 — 2026-06-02
- Built Streamlit web UI (app.py) with chat-style Q&A interface
- Sidebar with stack info, top_k slider (1-10), indexed-chunk count, clear-history button
- Expandable source panel for each answer (page numbers + cosine distance)
- Per-query latency tiles (retrieve / generate / total)
- Verified Langfuse traces still flow correctly from UI-initiated queries — no new instrumentation needed
- Saved demo screenshot to docs/app_screenshot.png
- First demoable version of the app
- Goal for Day 6: start Phase 3 (evaluation foundation) — build golden Q&A set + RAGAS scoring

### Day 5.5 — 2026-06-02
- Added CLI argument support to ingest.py (single file or --all)
- Updated system prompt to be book-agnostic with multi-source citations
- Added Library section to Streamlit sidebar showing all loaded books
- Verified idempotency: re-running ingest on `rich_dad_poor_dad.pdf` left collection unchanged at 1057 chunks (`add_chunks` correctly reported "All chunks already in collection — nothing to add.")
- Multi-doc verification with a second PDF was skipped today; will validate manually when a second book is ingested
- Goal for Day 6: build the eval foundation (golden set + RAGAS)

### Day 6 — 2026-06-03
- Ingested second PDF (`harshit_resume.pdf`, 9 chunks) before starting — library is now 1066 chunks across 2 sources
- Built hand-curated golden set (10 Q&A pairs: 7 book + 3 resume) at `evals/golden_set.jsonl`
- Installed RAGAS + langchain-ollama for local LLM-as-judge evaluation (pinned to `ragas>=0.2.0,<0.3.0` after the 0.4.x line broke against newer `langchain-community`)
- Created `src/eval/run.py` — computes retrieval recall@k + RAGAS faithfulness / answer_relevancy / context_precision
- First attempt produced all-NaN RAGAS scores due to RAGAS 0.2.x's default 180s per-job timeout (llama3.1:8b on CPU is too slow); fixed by passing `RunConfig(timeout=600, max_workers=1, max_retries=2)`. Total eval runtime: ~2h 17m on CPU
- **Baseline captured**: recall@5 70%, faithfulness 0.719, answer_relevancy 0.649, context_precision 0.580
- Confirmed quantitatively: resume retrieval recall is **0/3 (0%)** — the deliberate Phase 4 motivation
- Goal for Day 7: start Phase 4 by adding BM25 + hybrid retrieval, re-run eval, measure improvement (resume recall should jump from 0% → ~100%)

### Day 7 — 2026-06-03
- Added BM25 keyword retriever (`rank-bm25`) at `src/retrieval/bm25_retriever.py` — uses `re.findall(r"\w+", ...)` instead of a bare `.split()` to isolate alphanumeric IDs glued to punctuation by PDF extraction
- Added `HybridRetriever` at `src/retrieval/hybrid_retriever.py` — combines BM25 + vector via Reciprocal Rank Fusion (k=60, fetch_k=20 per retriever)
- Wired all consumers (`rag.py`, `chat.py`, `app.py`, `src/eval/run.py`) to `HybridRetriever`; added `"retrieval_strategy": "hybrid_bm25_vector_rrf"` to saved eval config
- Patched `rag.py` to reconfigure stdout to UTF-8 — verbose mode now handles Unicode chars (e.g. ♂) that resume chunks introduce
- Built `src/eval/compare.py` — side-by-side before/after comparison of two eval result files
- Re-ran full eval with hybrid: **recall@5 jumped from 70% → 100% (+30%)**, all 3 resume questions IMPROVED from MISS to HIT, **0 regressions**
- RAGAS metrics all improved: faithfulness +0.131, answer_relevancy +0.171, context_precision +0.139
- Generate latency went up ~29s due to more diverse / longer combined context — Day 8 reranker is the planned mitigation
- Goal for Day 8: add cross-encoder reranking (`bge-reranker-v2-m3`) on top of hybrid to trim top-20 → top-5 by true relevance, expect biggest gain on context_precision and a generate-latency win

### Day 8 — 2026-06-03
- Added cross-encoder reranker (`BAAI/bge-reranker-v2-m3`, ~570MB) via `sentence-transformers` (which also brought in torch CPU + transformers, ~1.5GB on disk total)
- Built `CrossEncoderReranker` at `src/rerank/cross_encoder.py` — re-scores `(query, chunk)` pairs jointly, attaches `rerank_score` to each chunk, fully Langfuse-traced
- Built `RerankedRetriever` at `src/retrieval/reranked_retriever.py` — composes `HybridRetriever(fetch_k=20)` → cross-encoder rerank → top-5, drop-in replacement for the previous retrievers
- Wired all 4 consumers (`rag.py`, `chat.py`, `app.py`, `src/eval/run.py`) to `RerankedRetriever`; eval config now records `"retrieval_strategy": "hybrid_rrf_plus_cross_encoder_rerank"` and the reranker model name
- Smoke test confirmed the headline behavior: on *"What programming languages does Harshit know?"* the reranker lifted both resume chunks (rerank_score 0.264 / 0.064) above 18 book candidates (all 0.000), and the LLM correctly answered with C++/JS/TS/HTML/CSS — fixing the Day 7 *"I don't have enough information"* failure
- Third full eval run: **recall@5 = 100%**, faithfulness 0.842, answer_relevancy 0.832, **context_precision 0.895** (the headline jump)
- **End-to-end win (baseline → Day 8)**: recall +30.0%, context_precision +0.315, faithfulness +0.123, answer_relevancy +0.183
- **Pure rerank contribution (hybrid → Day 8)**: context_precision +0.176 even though recall was already saturated at 100% — the rerank improves chunk *ordering*, not just hit-rate. Faithfulness drifted -0.008, well within RAGAS noise
- Retrieve latency went from 0.17s → 11.78s on CPU (the trade-off for joint-encoding 20 candidates); generate latency actually dropped slightly (-4.4s) on more focused contexts
- Goal for Day 9: Phase 5 — citation enforcement (LLM must cite valid sources or refuse to answer)

### Day 9 — 2026-06-04
- Built regex-based citation validator at `src/generate/citation_validator.py` — parses `[filename.pdf, page N]` citations, looks each one up against the chunks the retriever actually surfaced, classifies the answer as valid / invalid / refusal
- Hardened the system prompt: every factual claim now MUST be followed by `[filename.pdf, page N]`, "fake" page numbers are explicitly forbidden, and a literal refusal string is required when context is insufficient
- Rewrote `rag.py`'s `answer()` as a 4-step pipeline: retrieve → generate → validate → (if invalid) retry once with a stronger reminder → (if still invalid) replace with a safe refusal. Captures `is_valid`, `is_refusal`, `retried`, `fallback_used`, `num_parsed/valid/invalid` on every call and emits them as Langfuse metadata
- Updated `src/eval/run.py` to delegate to `answer()` (so the eval exercises the same validate/retry/refuse path real users hit), added a `compute_citation_metrics()` aggregator, and a `[2.5/3]` print section reporting valid_rate / refusal_rate / retry_rate / fallback_rate / hallucinated_citation_rate
- Extended `src/eval/compare.py` with a "Citation enforcement" block that handles `None` baselines so old result files still work
- Added a citation status badge to `app.py` — ✅ verified / 🛡️ refused / ⚠️ unverified — rendered in both the history view and live-query view via a shared `render_citation_badge()` helper
- Validator unit test passed all 6 cases; smoke-test confirmed *"What is the capital of France?"* now refuses instead of hallucinating Paris
- **Fourth eval run**: 100% valid answer rate, **0% hallucinated citation rate**, 10% retry rate (1 of 10 caught + corrected), 0% forced-fallback rate. RAGAS drifted slightly (-0.008 faithfulness, -0.021 relevancy) within noise — expected when answers become more literal under citation discipline
- **End-to-end (baseline → Day 9)**: recall@5 +30.0%, faithfulness +0.115, answer_relevancy +0.162, context_precision +0.315, plus the new trust layer
- Goal for Day 10/11: Phase 6 — CI regression gating (GitHub Actions running the eval on every PR, blocking merges if quality drops)
