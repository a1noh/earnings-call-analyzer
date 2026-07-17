"""RAG helper: retrieve quarter-scoped chunks and format them for grounding.

Every analysis node grounds its answer in the actual transcript text by running a
handful of targeted queries against ChromaDB (filtered to one quarter) and passing
the retrieved excerpts to Claude as the shared, cached context.
"""
from __future__ import annotations

from config import RETRIEVAL_K
from data.models import RetrievedChunk
from ingest import vectorstore


def retrieve(
    ticker: str,
    year: int,
    quarter: str,
    queries: list[str],
    k: int = RETRIEVAL_K,
) -> list[RetrievedChunk]:
    """Retrieve deduped top-k chunks for a quarter across the given queries."""
    collection = vectorstore.get_collection(ticker)
    return vectorstore.query(collection, queries, ticker, year, quarter, k)


def format_context(chunks: list[RetrievedChunk], header: str = "TRANSCRIPT EXCERPTS") -> str:
    """Render retrieved chunks into a labelled grounding block for the prompt."""
    if not chunks:
        return f"{header}:\n(no relevant excerpts found)"
    lines = [f"{header}:", ""]
    for i, c in enumerate(chunks, 1):
        speaker = f" — {c.speaker}" if c.speaker else ""
        lines.append(f"[{i}] (section: {c.section}{speaker})")
        lines.append(c.text)
        lines.append("")
    return "\n".join(lines)
