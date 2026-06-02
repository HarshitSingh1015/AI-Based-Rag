# Rich Dad Poor Dad RAG

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

## Progress log
- [x] **Day 1**: Project setup, Ollama installed, models pulled, folder structure, deps installed
- [ ] **Phase 1**: Hello World RAG (basic vector search + LLM)
- [ ] **Phase 2**: Observability (Langfuse tracing, latency, cost)
- [ ] **Phase 3**: Evaluation foundation (golden set + RAGAS)
- [ ] **Phase 4**: Hybrid retrieval + reranking
- [ ] **Phase 5**: Citation enforcement
- [ ] **Phase 6**: CI regression gating (GitHub Actions)
- [ ] **Phase 7**: Polish + UI + deploy

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
