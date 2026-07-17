"""Financial Modeling Prep (FMP) client with runtime gating detection.

FMP's free tier frequently gates the *body* of earnings-call transcripts (it is a
premium feature), while the actual gating signal varies: an HTTP 401/402/403, an
HTTP 200 with an ``{"Error Message": ...}`` body, or an empty payload. Rather than
guess at build time, we probe at runtime and return a typed ``TranscriptResult``
so the router can decide whether to fall back to SEC EDGAR. We never raise an
expected gating condition into the UI, and we never fabricate data.
"""
from __future__ import annotations

from typing import Optional

import requests

from config import FMP_API_KEY, REQUEST_TIMEOUT
from data.models import Document, TranscriptResult

_STABLE_TRANSCRIPT = "https://financialmodelingprep.com/stable/earning-call-transcript"
_V3_TRANSCRIPT = (
    "https://financialmodelingprep.com/api/v3/earning_call_transcript/{ticker}"
)
_V4_DATES = "https://financialmodelingprep.com/api/v4/earning_call_transcript"

_GATED_HINTS = ("subscription", "plan", "legacy", "endpoint", "premium", "upgrade")


class FMPClient:
    """Best-effort fetch of real transcripts, with graceful gating detection."""

    def __init__(self, api_key: Optional[str] = FMP_API_KEY) -> None:
        self.api_key = api_key
        self.session = requests.Session()

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    # -- gating detection ---------------------------------------------------
    @staticmethod
    def _classify_payload(status: int, payload) -> Optional[str]:
        """Return a gating reason string, or None if the payload looks usable."""
        if status in (401, 402, 403):
            return "fmp_gated"
        if status == 404:
            return "fmp_not_found"
        if status >= 400:
            return "fmp_http_error"
        # HTTP 200 but FMP sometimes returns an error object.
        if isinstance(payload, dict):
            msg = str(payload.get("Error Message", "")).lower()
            if msg:
                return "fmp_gated" if any(h in msg for h in _GATED_HINTS) else "fmp_error"
        if not payload:
            return "fmp_empty"
        return None

    def _get(self, url: str, params: dict) -> tuple[int, object]:
        params = {**params, "apikey": self.api_key}
        try:
            resp = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException:
            return 599, None
        try:
            payload = resp.json()
        except ValueError:
            payload = None
        return resp.status_code, payload

    # -- transcript body ----------------------------------------------------
    def get_transcript(self, ticker: str, year: int, quarter: str) -> TranscriptResult:
        """Fetch one transcript body, trying the stable then legacy v3 endpoint."""
        if not self.has_key:
            return TranscriptResult(ok=False, reason="fmp_no_key")

        q_num = int(str(quarter).lstrip("Qq") or 0)
        attempts = [
            (_STABLE_TRANSCRIPT, {"symbol": ticker.upper(), "year": year, "quarter": q_num}),
            (_V3_TRANSCRIPT.format(ticker=ticker.upper()), {"year": year, "quarter": q_num}),
        ]
        last_reason = "fmp_empty"
        for url, params in attempts:
            status, payload = self._get(url, params)
            reason = self._classify_payload(status, payload)
            if reason:
                last_reason = reason
                if reason == "fmp_gated":  # no point trying the other endpoint
                    return TranscriptResult(ok=False, reason=reason)
                continue
            record = payload[0] if isinstance(payload, list) else payload
            content = (record or {}).get("content", "") if isinstance(record, dict) else ""
            if not content or len(content) < 200:
                last_reason = "fmp_empty"
                continue
            return TranscriptResult(
                ok=True,
                text=content,
                call_date=(record.get("date") if isinstance(record, dict) else None),
            )
        return TranscriptResult(ok=False, reason=last_reason)

    # -- available quarters -------------------------------------------------
    def list_recent_quarters(
        self, ticker: str, n: int
    ) -> tuple[list[tuple[int, str]], Optional[str]]:
        """Return up to ``n`` recent (year, quarter) pairs, plus a gating reason.

        Uses the v4 dates endpoint. Returns ([], reason) when gated so the router
        can fall back without hammering the transcript endpoint blindly.
        """
        if not self.has_key:
            return [], "fmp_no_key"
        status, payload = self._get(_V4_DATES, {"symbol": ticker.upper()})
        reason = self._classify_payload(status, payload)
        if reason:
            return [], reason

        pairs: list[tuple[int, str]] = []
        for item in payload if isinstance(payload, list) else []:
            try:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    q_num, yr = int(item[0]), int(item[1])
                elif isinstance(item, dict):
                    q_num, yr = int(item["quarter"]), int(item["year"])
                else:
                    continue
                pairs.append((yr, f"Q{q_num}"))
            except (ValueError, KeyError, TypeError):
                continue
        # Most recent first.
        pairs.sort(key=lambda p: (p[0], p[1]), reverse=True)
        return pairs[:n], None

    def fetch_earnings_documents(
        self, ticker: str, num_quarters: int
    ) -> tuple[list[Document], Optional[str]]:
        """Try to fetch real transcripts for recent quarters.

        Returns (documents, reason). An empty list with a reason means the router
        should fall back to EDGAR.
        """
        if not self.has_key:
            return [], "fmp_no_key"
        quarters, reason = self.list_recent_quarters(ticker, num_quarters)
        if reason:
            return [], reason

        documents: list[Document] = []
        for year, quarter in quarters:
            result = self.get_transcript(ticker, year, quarter)
            if not result.ok:
                if result.reason == "fmp_gated":
                    return [], "fmp_gated"
                continue
            documents.append(
                Document(
                    ticker=ticker.upper(),
                    year=year,
                    quarter=quarter,
                    source="fmp_transcript",
                    text=result.text,
                    call_date=result.call_date,
                    source_label="FMP earnings-call transcript (incl. Q&A)",
                )
            )
        if not documents:
            return [], "fmp_empty"
        return documents, None
