"""Langfuse initialization and helper utilities."""
import os
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

_client: Langfuse | None = None


def init() -> Langfuse:
    """Initialize the global Langfuse client (singleton)."""
    global _client
    if _client is None:
        _client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    return _client


def flush() -> None:
    """Flush pending events to Langfuse. Call before program exit."""
    if _client is not None:
        _client.flush()
