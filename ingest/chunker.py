"""Section-aware chunking for earnings documents.

Transcripts and press releases are long and structurally meaningful, so we chunk
on line/paragraph boundaries (preserving speaker turns) and tag each chunk with a
section so retrieval and evals can reason about *where* a claim came from:

* FMP transcripts  -> ``prepared_remarks`` vs ``qa`` (split at the Q&A boundary),
  with a best-effort ``speaker`` attached.
* EDGAR 8-K text   -> ``forward_looking`` (safe-harbor language) vs ``press_release``.
"""
from __future__ import annotations

import re

from config import CHUNK_OVERLAP_WORDS, CHUNK_WORDS
from data.models import Chunk, Document

# Markers that indicate the start of the analyst Q&A portion of a call.
_QA_BOUNDARY = re.compile(
    r"(question[- ]and[- ]answer|questions?\s*&\s*answers?|q\s*&\s*a\s*session|"
    r"we (?:will|'ll) now (?:begin|open).{0,30}question)",
    re.IGNORECASE,
)
# A line that looks like a speaker label, e.g. "Tim Cook:" or "Operator:".
_SPEAKER = re.compile(r"^([A-Z][A-Za-z.,'&\- ]{1,48}):\s")
# Forward-looking / safe-harbor language common in press releases.
_FORWARD_LOOKING = re.compile(
    r"forward[- ]looking statement|private securities litigation reform act|"
    r"safe harbor",
    re.IGNORECASE,
)


def _segments(doc: Document) -> list[tuple[str, str]]:
    """Split a document into (section, text) segments before windowing."""
    text = doc.text
    if doc.source == "fmp_transcript":
        match = _QA_BOUNDARY.search(text)
        if match:
            return [
                ("prepared_remarks", text[: match.start()]),
                ("qa", text[match.start() :]),
            ]
        return [("prepared_remarks", text)]

    # EDGAR press release: peel off the safe-harbor / forward-looking block.
    match = _FORWARD_LOOKING.search(text)
    if match:
        return [
            ("press_release", text[: match.start()]),
            ("forward_looking", text[match.start() :]),
        ]
    return [("press_release", text)]


def _speaker_for(lines: list[str]) -> str | None:
    """Return the most recent speaker label found in a group of lines, if any."""
    speaker = None
    for line in lines:
        m = _SPEAKER.match(line.strip())
        if m:
            speaker = m.group(1).strip()
    return speaker


def _window(lines: list[str]) -> list[tuple[str, str | None]]:
    """Group lines into ~CHUNK_WORDS windows with overlap; keep speaker context."""
    chunks: list[tuple[str, str | None]] = []
    buf: list[str] = []
    words = 0
    for line in lines:
        buf.append(line)
        words += len(line.split())
        if words >= CHUNK_WORDS:
            chunks.append(("\n".join(buf).strip(), _speaker_for(buf)))
            # Start next window with an overlap tail for context continuity.
            overlap: list[str] = []
            ov_words = 0
            for prev in reversed(buf):
                overlap.insert(0, prev)
                ov_words += len(prev.split())
                if ov_words >= CHUNK_OVERLAP_WORDS:
                    break
            buf = overlap
            words = ov_words
    if buf and "\n".join(buf).strip():
        chunks.append(("\n".join(buf).strip(), _speaker_for(buf)))
    return chunks


def chunk_document(doc: Document) -> list[Chunk]:
    """Turn a Document into a list of tagged, indexable Chunks."""
    out: list[Chunk] = []
    idx = 0
    for section, seg_text in _segments(doc):
        lines = [ln for ln in seg_text.splitlines() if ln.strip()]
        for chunk_text, speaker in _window(lines):
            if len(chunk_text) < 40:  # skip stubs
                continue
            out.append(
                Chunk(
                    id=f"{doc.ticker}_{doc.year}_{doc.quarter}_{idx}",
                    text=chunk_text,
                    ticker=doc.ticker,
                    year=doc.year,
                    quarter=doc.quarter,
                    source=doc.source,
                    section=section,
                    speaker=speaker,
                    chunk_index=idx,
                    call_date=doc.call_date,
                )
            )
            idx += 1
    return out
