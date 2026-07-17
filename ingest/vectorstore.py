"""ChromaDB vector store wrappers.

Local, on-disk persistence with Chroma's default onnx ``all-MiniLM-L6-v2``
embedding function — no API key, no torch, no cloud. One collection per company
holds all quarters; retrieval is filtered by ``ticker``/``year``/``quarter``
metadata so quarter-over-quarter comparison can pull from a single collection.
"""
from __future__ import annotations

from chromadb import PersistentClient
from chromadb.utils import embedding_functions

from config import CHROMA_PATH, RETRIEVAL_K
from data.models import Chunk, Document, RetrievedChunk
from ingest.chunker import chunk_document

# Instantiated lazily so importing this module never triggers a model download.
_client: PersistentClient | None = None
_embedder = None


def _get_client() -> PersistentClient:
    global _client
    if _client is None:
        _client = PersistentClient(path=CHROMA_PATH)
    return _client


def _get_embedder():
    """Lazily build the onnx MiniLM embedder (downloads the model on first use)."""
    global _embedder
    if _embedder is None:
        _embedder = embedding_functions.DefaultEmbeddingFunction()
    return _embedder


def collection_name(ticker: str) -> str:
    return f"transcripts_{ticker.lower()}"


def get_collection(ticker: str):
    """Get or create the per-company collection with the onnx embedder."""
    return _get_client().get_or_create_collection(
        name=collection_name(ticker),
        embedding_function=_get_embedder(),
        metadata={"hnsw:space": "cosine"},
    )


def _quarter_filter(ticker: str, year: int, quarter: str) -> dict:
    return {
        "$and": [
            {"ticker": ticker.upper()},
            {"year": year},
            {"quarter": quarter},
        ]
    }


def has_quarter(collection, ticker: str, year: int, quarter: str) -> bool:
    """True if the given quarter is already indexed (so we can skip re-ingest)."""
    got = collection.get(where=_quarter_filter(ticker, year, quarter), limit=1)
    return bool(got.get("ids"))


def upsert_chunks(collection, chunks: list[Chunk]) -> int:
    """Idempotently upsert chunks (stable ids make re-ingest safe)."""
    if not chunks:
        return 0
    collection.upsert(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.metadata() for c in chunks],
    )
    return len(chunks)


def ingest_documents(documents: list[Document]) -> int:
    """Chunk and index a batch of documents; returns total chunks written."""
    total = 0
    for doc in documents:
        collection = get_collection(doc.ticker)
        if has_quarter(collection, doc.ticker, doc.year, doc.quarter):
            continue  # already indexed
        total += upsert_chunks(collection, chunk_document(doc))
    return total


def indexed_quarters(collection, ticker: str) -> list[tuple[int, str]]:
    """Return the sorted (year, quarter) pairs currently indexed for a ticker."""
    got = collection.get(where={"ticker": ticker.upper()})
    seen: set[tuple[int, str]] = set()
    for md in got.get("metadatas", []) or []:
        seen.add((int(md["year"]), str(md["quarter"])))
    return sorted(seen, reverse=True)


def query(
    collection,
    query_texts: list[str],
    ticker: str,
    year: int,
    quarter: str,
    k: int = RETRIEVAL_K,
) -> list[RetrievedChunk]:
    """Run one or more RAG queries scoped to a single quarter; dedupe results."""
    res = collection.query(
        query_texts=query_texts,
        n_results=k,
        where=_quarter_filter(ticker, year, quarter),
    )
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    dists = res.get("distances") or []

    seen: set[str] = set()
    out: list[RetrievedChunk] = []
    for qi in range(len(docs)):
        for di, text in enumerate(docs[qi]):
            if text in seen:
                continue
            seen.add(text)
            md = metas[qi][di] if qi < len(metas) and di < len(metas[qi]) else {}
            dist = dists[qi][di] if qi < len(dists) and di < len(dists[qi]) else 0.0
            out.append(
                RetrievedChunk(
                    text=text,
                    section=str(md.get("section", "")),
                    speaker=md.get("speaker"),
                    distance=float(dist),
                )
            )
    return out
