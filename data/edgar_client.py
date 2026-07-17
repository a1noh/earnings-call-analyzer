"""SEC EDGAR client — the free fallback data source.

EDGAR does NOT host earnings-call transcripts (they are not SEC filings). The
closest real, free data is the **8-K earnings press release** (Item 2.02 /
Exhibit 99.1): prepared results and guidance language, but no analyst Q&A. This
client resolves a ticker to a CIK, finds recent earnings 8-Ks, locates the
Exhibit 99.1 press release, and returns clean text as normalized ``Document``s.

All requests send the SEC-required contact ``User-Agent``; missing/generic UAs
get a 403. We keep well under SEC's ~10 req/s guidance with a small delay.
"""
from __future__ import annotations

import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

from config import REQUEST_TIMEOUT, SEC_USER_AGENT
from data.models import Document

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
_ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}"

# Module-level cache for the ticker -> CIK map (fetched once per process).
_ticker_cik_cache: dict[str, str] = {}


class EdgarError(Exception):
    """Raised for unrecoverable EDGAR errors (bad ticker, network)."""


class EdgarClient:
    """Fetches earnings press releases from SEC EDGAR."""

    def __init__(self, user_agent: str = SEC_USER_AGENT) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        )

    # -- low-level HTTP -----------------------------------------------------
    def _get(self, url: str, as_json: bool = True):
        time.sleep(0.15)  # stay well under SEC's rate guidance
        resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json() if as_json else resp.text

    # -- ticker -> CIK ------------------------------------------------------
    def ticker_to_cik(self, ticker: str) -> str:
        """Return the 10-digit zero-padded CIK for a ticker (cached)."""
        global _ticker_cik_cache
        ticker = ticker.upper().strip()
        if not _ticker_cik_cache:
            try:
                data = self._get(_TICKER_MAP_URL)
            except requests.RequestException as exc:  # network failure
                raise EdgarError(f"Could not reach SEC ticker map: {exc}") from exc
            for row in data.values():
                _ticker_cik_cache[str(row["ticker"]).upper()] = (
                    f"{int(row['cik_str']):010d}"
                )
        cik = _ticker_cik_cache.get(ticker)
        if not cik:
            raise EdgarError(f"Ticker '{ticker}' not found in the SEC database.")
        return cik

    # -- filings ------------------------------------------------------------
    def _recent_earnings_filings(self, cik10: str, limit: int) -> list[dict]:
        """Return recent 8-K filings that report results of operations (Item 2.02)."""
        try:
            data = self._get(_SUBMISSIONS_URL.format(cik10=cik10))
        except requests.RequestException as exc:
            raise EdgarError(f"Could not load filings for CIK {cik10}: {exc}") from exc

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        items = recent.get("items", [])
        report_dates = recent.get("reportDate", [])
        filing_dates = recent.get("filingDate", [])

        out: list[dict] = []
        for i, form in enumerate(forms):
            if form != "8-K":
                continue
            item_str = items[i] if i < len(items) else ""
            if "2.02" not in item_str:  # 2.02 = Results of Operations (earnings)
                continue
            out.append(
                {
                    "accession": accessions[i],
                    "primary_doc": primary_docs[i] if i < len(primary_docs) else "",
                    "report_date": report_dates[i] if i < len(report_dates) else "",
                    "filing_date": filing_dates[i] if i < len(filing_dates) else "",
                }
            )
            if len(out) >= limit * 2:  # over-fetch; some may lack a 99.1 exhibit
                break
        return out

    def _find_exhibit_url(
        self, cik_int: int, accession: str, primary_doc: str
    ) -> Optional[str]:
        """Locate the Exhibit 99.1 press-release document within a filing folder."""
        acc_nodash = accession.replace("-", "")
        base = _ARCHIVES.format(cik=cik_int, acc=acc_nodash)
        try:
            index = self._get(f"{base}/index.json")
        except requests.RequestException:
            return None

        htm_items = [
            it["name"]
            for it in index.get("directory", {}).get("item", [])
            if str(it.get("name", "")).lower().endswith((".htm", ".html"))
        ]
        if not htm_items:
            return None

        # Prefer an Exhibit 99.1-style filename, then any 99.x, then the largest
        # .htm that is not the primary 8-K cover document.
        def score(name: str) -> int:
            low = name.lower()
            if any(k in low for k in ("ex99-1", "ex99_1", "ex991", "ex-99.1", "ex99.1")):
                return 3
            if "99" in low and "ex" in low:
                return 2
            if name == primary_doc:
                return -1
            return 1

        best = max(htm_items, key=score)
        return f"{base}/{best}"

    @staticmethod
    def _strip_html(html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        # Collapse excessive blank lines / whitespace.
        lines = [ln.strip() for ln in text.splitlines()]
        return "\n".join(ln for ln in lines if ln)

    @staticmethod
    def _period_to_quarter(date_str: str) -> tuple[int, str]:
        """Map a YYYY-MM-DD period-end date to a (year, calendar-quarter) label.

        Note: this uses calendar quarters. Companies with non-calendar fiscal
        years (e.g. Apple) will be labelled by calendar period, which the UI
        notes. Good enough for cross-quarter comparison in a demo.
        """
        year, month = int(date_str[:4]), int(date_str[5:7])
        quarter = f"Q{(month - 1) // 3 + 1}"
        return year, quarter

    # -- public API ---------------------------------------------------------
    def fetch_earnings_documents(
        self, ticker: str, num_quarters: int
    ) -> list[Document]:
        """Return up to ``num_quarters`` recent earnings press releases as Documents."""
        cik10 = self.ticker_to_cik(ticker)
        cik_int = int(cik10)
        filings = self._recent_earnings_filings(cik10, num_quarters)

        documents: list[Document] = []
        seen: set[tuple[int, str]] = set()
        for f in filings:
            if len(documents) >= num_quarters:
                break
            url = self._find_exhibit_url(cik_int, f["accession"], f["primary_doc"])
            if not url:
                continue
            try:
                html = self._get(url, as_json=False)
            except requests.RequestException:
                continue
            text = self._strip_html(html)
            if len(text) < 500:  # too short to be a real release
                continue

            date_str = f["report_date"] or f["filing_date"]
            year, quarter = self._period_to_quarter(date_str)
            key = (year, quarter)
            if key in seen:
                continue
            seen.add(key)

            documents.append(
                Document(
                    ticker=ticker.upper(),
                    year=year,
                    quarter=quarter,
                    source="edgar_8k",
                    text=text,
                    call_date=date_str or None,
                    source_label="SEC EDGAR 8-K earnings release (no Q&A)",
                )
            )
        if not documents:
            raise EdgarError(
                f"No earnings press releases (8-K Item 2.02) found for {ticker.upper()}."
            )
        return documents
