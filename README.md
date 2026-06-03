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

## Progress log
- [x] **Day 1**: Project setup, Ollama installed, models pulled, folder structure, deps installed
- [x] **Phase 1**: Hello World RAG (basic vector search + LLM)
- [x] **Phase 2**: Observability (Langfuse tracing, latency, token usage)
- [x] **Day 5**: Streamlit web UI ← demoable!
- [x] **Day 5.5**: Multi-document support via CLI
- [x] **Phase 3**: Evaluation foundation (golden set + RAGAS) — 10 questions, baseline captured
- [ ] **Phase 4**: Hybrid retrieval + reranking
- [ ] **Phase 5**: Citation enforcement
- [ ] **Phase 6**: CI regression gating (GitHub Actions)
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
