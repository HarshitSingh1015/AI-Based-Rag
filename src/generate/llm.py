"""LLM wrapper using Ollama's llama3.1:8b."""
import ollama

LLM_MODEL = "llama3.1:8b"


def generate(messages: list[dict], temperature: float = 0.2) -> str:
    """Generate a response from the LLM.

    Lower temperature (0.2) keeps answers grounded and consistent — important for RAG.
    """
    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        options={"temperature": temperature},
    )
    return response["message"]["content"]


if __name__ == "__main__":
    out = generate([
        {"role": "user", "content": "Say 'hello' in one short sentence."}
    ])
    print(out)
