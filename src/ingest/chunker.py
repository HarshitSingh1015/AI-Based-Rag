"""Text chunking utility for the RAG ingestion pipeline."""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by character count.

    Tries to break on sentence/paragraph boundaries when possible so
    chunks read naturally and aren't cut mid-sentence.

    Args:
        text: The text to chunk.
        chunk_size: Target chunk size in characters.
        overlap: Character overlap between consecutive chunks.

    Returns:
        A list of chunk strings.
    """
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            for delim in ["\n\n", ". ", "? ", "! ", "\n"]:
                last = text.rfind(delim, start, end)
                if last > start + chunk_size // 2:
                    end = last + len(delim)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break
        start = max(end - overlap, start + 1)

    return chunks


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """Chunk a list of page documents and preserve page-level metadata.

    Returns:
        A list of chunk dicts with keys:
        - chunk_id (str, unique)
        - text (str)
        - page_num (int)
        - source (str)
    """
    all_chunks: list[dict] = []
    for page in pages:
        for j, chunk in enumerate(chunk_text(page["text"], chunk_size, overlap)):
            all_chunks.append({
                "chunk_id": f"{page['source']}_p{page['page_num']}_c{j}",
                "text": chunk,
                "page_num": page["page_num"],
                "source": page["source"],
            })
    return all_chunks
