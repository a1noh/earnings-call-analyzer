"""Central configuration for the Earnings Call Analyzer.

Loads environment variables from a local ``.env`` file (if present) and exposes
tunable constants used across the ingestion, agent, and eval layers. Nothing here
hardcodes secrets — API keys are always read from the environment.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root, if it exists. Real env vars always win.
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# --- Secrets (read from environment; never hardcode) ------------------------
def _clean(value: str | None) -> str | None:
    """Strip stray whitespace (incl. non-breaking spaces) from a secret value.

    Copy-pasted keys often carry an invisible trailing space or newline that
    breaks HTTP header encoding; this makes reads robust to that.
    """
    return value.strip() if value else value


ANTHROPIC_API_KEY: str | None = _clean(os.getenv("ANTHROPIC_API_KEY"))
FMP_API_KEY: str | None = _clean(os.getenv("FMP_API_KEY"))
SEC_USER_AGENT: str = _clean(
    os.getenv("SEC_USER_AGENT", "EarningsCallAnalyzer example@example.com")
) or "EarningsCallAnalyzer example@example.com"

# --- Models -----------------------------------------------------------------
# Primary analysis model for the four LangGraph nodes.
ANALYSIS_MODEL: str = os.getenv("ANALYSIS_MODEL", "claude-opus-4-8")
# Cheaper model for the LLM-as-judge groundedness eval pass.
JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "claude-haiku-4-5")

MAX_TOKENS: int = 16000

# --- Vector store -----------------------------------------------------------
CHROMA_PATH: str = str(PROJECT_ROOT / "chroma_db")

# --- Retrieval / chunking ---------------------------------------------------
# Chunk sizing is measured in words (a cheap, dependency-free proxy for tokens).
CHUNK_WORDS: int = 320          # ~ 400-450 tokens per chunk
CHUNK_OVERLAP_WORDS: int = 48   # ~ 15% overlap
RETRIEVAL_K: int = 6            # top-k chunks returned per RAG query

# --- Ingestion --------------------------------------------------------------
NUM_QUARTERS: int = 4  # how many recent quarters to try to ingest

# HTTP
REQUEST_TIMEOUT: int = 30  # seconds


def missing_keys() -> list[str]:
    """Return the names of required secrets that are not set.

    ``FMP_API_KEY`` is optional (the app degrades to EDGAR without it), so it is
    not included here. Used by the UI to show a clear, actionable error.
    """
    missing = []
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    return missing
