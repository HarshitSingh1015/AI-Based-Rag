"""RAG prompt template construction (with strict citation requirements)."""

SYSTEM_PROMPT = """You are a helpful assistant answering questions based ONLY on the \
provided book/document context.

STRICT RULES (you MUST follow all):
1. Answer ONLY using the context provided below. Do not use outside knowledge.
2. EVERY factual claim in your answer MUST be immediately followed by a citation in
   this EXACT format: [filename.pdf, page N]
   Example: "Kiyosaki defines an asset as something that puts money in your pocket
   [rich_dad_poor_dad.pdf, page 67]."
3. Only cite sources and page numbers that appear in the provided context.
   DO NOT INVENT citations. If a fact has no supporting source in the context,
   do not include that fact.
4. If the context does not contain enough information to answer the question,
   respond EXACTLY with:
   "I don't have enough information in the available books to answer this confidently."
5. End your answer with a "Sources:" line listing each unique citation used,
   in the same [filename.pdf, page N] format, comma-separated.
"""

USER_PROMPT_TEMPLATE = """\
Context from the library:
---
{context}
---

Question: {question}

Answer (remember: cite every claim with [filename.pdf, page N]):"""


RETRY_INSTRUCTION = """Your previous answer contained citations that don't match the \
provided context, or had factual claims without citations.

Re-answer using ONLY the sources and page numbers shown above. Every factual claim \
MUST be followed by [filename.pdf, page N], using ONLY the exact filenames and page \
numbers from the context. If you cannot find supporting information, respond exactly:
"I don't have enough information in the available books to answer this confidently."
"""


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


def build_retry_messages(
    previous_messages: list[dict],
    previous_response: str,
) -> list[dict]:
    """Build messages for a retry attempt after invalid citations."""
    return previous_messages + [
        {"role": "assistant", "content": previous_response},
        {"role": "user", "content": RETRY_INSTRUCTION},
    ]
