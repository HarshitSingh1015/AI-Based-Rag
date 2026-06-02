"""Embedding utility using Ollama's nomic-embed-text model."""
import ollama

EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768


def embed_text(text: str) -> list[float]:
    """Embed a single text string into a vector."""
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def embed_batch(texts: list[str], verbose: bool = True) -> list[list[float]]:
    """Embed a list of texts. Ollama does not batch natively, so this
    calls the model one chunk at a time. Prints progress every 50 items.
    """
    embeddings: list[list[float]] = []
    for i, t in enumerate(texts):
        embeddings.append(embed_text(t))
        if verbose and (i + 1) % 50 == 0:
            print(f"  Embedded {i + 1}/{len(texts)} chunks")
    return embeddings


if __name__ == "__main__":
    vec = embed_text("Hello, Rich Dad Poor Dad!")
    print(f"Embedding dimension: {len(vec)}")
    print(f"First 5 values: {vec[:5]}")
