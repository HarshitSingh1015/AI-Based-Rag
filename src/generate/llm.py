"""LLM wrapper using Ollama's llama3.1:8b."""
import ollama
from langfuse.decorators import observe, langfuse_context

LLM_MODEL = "llama3.1:8b"


@observe(as_type="generation", name="llm_generate")
def generate(messages: list[dict], temperature: float = 0.2) -> str:
    """Generate a response from the LLM. Tracked as a Langfuse 'generation'
    with model name, input messages, output text, and token usage.
    """
    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        options={"temperature": temperature},
    )
    output = response["message"]["content"]

    usage = {
        "input": response.get("prompt_eval_count", 0),
        "output": response.get("eval_count", 0),
        "unit": "TOKENS",
    }

    langfuse_context.update_current_observation(
        model=LLM_MODEL,
        input=messages,
        output=output,
        usage=usage,
        metadata={"temperature": temperature},
    )
    return output


if __name__ == "__main__":
    out = generate([
        {"role": "user", "content": "Say 'hello' in one short sentence."}
    ])
    print(out)
