"""Pydantic data models shared across the ingestion and agent layers.

These give us typed, validated objects so the router, chunker, vector store, and
LangGraph nodes all speak the same language regardless of whether a document came
from FMP (a real transcript) or SEC EDGAR (an 8-K earnings press release).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Where a document ultimately came from. Drives UI labelling and eval claims.
SourceType = Literal["fmp_transcript", "edgar_8k"]


class TranscriptResult(BaseModel):
    """Result of an attempt to fetch a transcript body from FMP.

    We never raise into the UI for an expected gating condition; instead we return
    a typed result so the router can decide whether to fall back to EDGAR.
    """

    ok: bool
    source: SourceType = "fmp_transcript"
    text: str = ""
    call_date: Optional[str] = None
    # Machine-readable reason when ok is False, e.g. "fmp_gated",
    # "fmp_no_key", "fmp_empty", "fmp_http_error", "fmp_not_found".
    reason: Optional[str] = None


class Document(BaseModel):
    """A normalized earnings document ready for chunking and indexing."""

    ticker: str
    year: int
    quarter: str  # e.g. "Q3"
    source: SourceType
    text: str
    call_date: Optional[str] = None
    # Human-readable label for the UI source badge.
    source_label: str = ""


class Chunk(BaseModel):
    """A single chunk of a document plus the metadata stored alongside it."""

    id: str
    text: str
    ticker: str
    year: int
    quarter: str
    source: SourceType
    section: str  # e.g. "prepared_remarks", "qa", "press_release"
    speaker: Optional[str] = None
    chunk_index: int = 0
    call_date: Optional[str] = None

    def metadata(self) -> dict:
        """Return a Chroma-safe metadata dict (no None values allowed)."""
        md = {
            "ticker": self.ticker,
            "year": self.year,
            "quarter": self.quarter,
            "source": self.source,
            "section": self.section,
            "chunk_index": self.chunk_index,
        }
        if self.speaker:
            md["speaker"] = self.speaker
        if self.call_date:
            md["call_date"] = self.call_date
        return md


class RetrievedChunk(BaseModel):
    """A chunk returned from a similarity search, with its distance score."""

    text: str
    section: str
    speaker: Optional[str] = None
    distance: float = 0.0

    def cite(self, max_len: int = 400) -> str:
        """A short, display-friendly snippet for inline source quotes."""
        t = " ".join(self.text.split())
        return t if len(t) <= max_len else t[: max_len - 1] + "…"


class ReasoningStep(BaseModel):
    """One step of the agent's visible reasoning, surfaced in the UI."""

    node: str
    queries: list[str] = Field(default_factory=list)
    sections_used: list[str] = Field(default_factory=list)
    num_chunks: int = 0
    summary: str = ""
