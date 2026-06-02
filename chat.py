"""Interactive REPL for asking Rich Dad Poor Dad RAG questions."""
from src.retrieval.retriever import Retriever
from rag import answer


def main() -> None:
    print("=== Rich Dad Poor Dad RAG — Interactive Chat ===")
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
        print()


if __name__ == "__main__":
    main()
