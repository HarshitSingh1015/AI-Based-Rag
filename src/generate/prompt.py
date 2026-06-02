"""RAG prompt template construction (multi-document)."""

SYSTEM_PROMPT = """You are a helpful assistant answering questions based ONLY on the \
provided book/document context.

Rules you must follow:
1. Answer ONLY using the context provided below. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say exactly:
   "I don't have enough information in the available books to answer this confidently."
3. Be concise and direct. Quote or paraphrase the source where helpful.
4. After your answer, list the source documents and page numbers you used in the
   format: Sources: [book_name, page X], [other_book, page Y]
"""

USER_PROMPT_TEMPLATE = """\
Context from the library:
---
{context}
---

Question: {question}

Answer:"""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block including source filenames."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i} — {c['source']}, page {c['page_num']}]\n{c['text']}"
        )
    return "\n\n".join(parts)


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    """Build chat messages for the LLM call."""
    context = format_context(chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            context=context, question=question,
        )},
    ]
