"""Interactive REPL for asking Rich Dad Poor Dad RAG questions.
Each query is traced in Langfuse.
"""
import atexit

from src.observability.tracer import init as init_langfuse, flush as flush_langfuse
from src.retrieval.retriever import Retriever
from rag import answer

init_langfuse()
atexit.register(flush_langfuse)


def main() -> None:
    print("=== Rich Dad Poor Dad RAG — Interactive Chat ===")
    print("(Every query is traced in Langfuse)")
    print("Type your question and press Enter.")
    print("Type 'exit', 'quit', or 'q' to leave.\n")

    retriever = Retriever()
    print(f"Collection ready: {retriever.collection.count()} chunks\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if q.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break
        if not q:
            continue
        answer(q, retriever)
        flush_langfuse()
        print()


if __name__ == "__main__":
    main()
