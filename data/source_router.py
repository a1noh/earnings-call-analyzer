"""Source router: try real FMP transcripts first, fall back to SEC EDGAR.

This is the "never mock" guarantee. It attempts FMP (real transcripts + Q&A),
detects gating, and falls back to EDGAR 8-K earnings releases. The same codebase
therefore runs free today and auto-upgrades to full transcripts if a paid FMP key
is added — no code change. Returns the documents plus a human-readable note so the
UI can explain exactly which source was used and why.
"""
from __future__ import annotations

from dataclasses import dataclass

from data.edgar_client import EdgarClient, EdgarError
from data.fmp_client import FMPClient
from data.models import Document

# Human-readable explanations for each FMP fallback reason.
_REASON_NOTES = {
    "fmp_no_key": "No FMP API key set — using SEC EDGAR earnings releases.",
    "fmp_gated": "FMP transcripts require a paid plan — fell back to SEC EDGAR earnings releases.",
    "fmp_empty": "FMP returned no transcript data — fell back to SEC EDGAR earnings releases.",
    "fmp_error": "FMP returned an error — fell back to SEC EDGAR earnings releases.",
    "fmp_http_error": "FMP request failed — fell back to SEC EDGAR earnings releases.",
    "fmp_not_found": "FMP has no transcripts for this ticker — fell back to SEC EDGAR earnings releases.",
}


@dataclass
class IngestResult:
    """The outcome of a source-routing attempt."""

    documents: list[Document]
    source: str          # "fmp_transcript" | "edgar_8k"
    source_label: str    # for the UI badge
    note: str            # explanation of the routing decision


class SourceRouter:
    """Orchestrates FMP -> EDGAR document acquisition."""

    def __init__(self) -> None:
        self.fmp = FMPClient()
        self.edgar = EdgarClient()

    def fetch(self, ticker: str, num_quarters: int) -> IngestResult:
        """Return earnings documents for a ticker, preferring real transcripts.

        Raises EdgarError only if BOTH FMP and the EDGAR fallback fail — a genuine
        "no data anywhere" condition the UI surfaces as an error.
        """
        ticker = ticker.upper().strip()

        # 1) Try FMP for real transcripts.
        docs, reason = self.fmp.fetch_earnings_documents(ticker, num_quarters)
        if docs:
            return IngestResult(
                documents=docs,
                source="fmp_transcript",
                source_label=docs[0].source_label,
                note="Using real FMP earnings-call transcripts (includes analyst Q&A).",
            )

        # 2) Fall back to EDGAR.
        note = _REASON_NOTES.get(reason or "", "Using SEC EDGAR earnings releases.")
        try:
            edgar_docs = self.edgar.fetch_earnings_documents(ticker, num_quarters)
        except EdgarError as exc:
            raise EdgarError(
                f"Could not retrieve earnings data for {ticker}. "
                f"FMP: {reason or 'unavailable'}. EDGAR: {exc}"
            ) from exc

        return IngestResult(
            documents=edgar_docs,
            source="edgar_8k",
            source_label=edgar_docs[0].source_label,
            note=note,
        )
