# 📊 Earnings Call Analyzer

An agentic RAG application that ingests a public company's recent earnings
communications, runs them through a multi-step **LangGraph** agent powered by
**Claude**, and surfaces structured, quote-grounded intelligence across quarters —
with a visible **eval layer** scoring output quality.

## What this project demonstrates

This is a portfolio project built to show the core skills of an AI forward-deployed
engineer end to end: a **RAG pipeline** (section-aware chunking + a local ChromaDB
vector store with on-device embeddings), **agentic multi-step reasoning** (a
LangGraph `StateGraph` whose nodes each retrieve grounding text and call Claude for
structured extraction), an **eval framework** (deterministic quote-grounding,
completeness, and consistency checks plus an optional LLM-as-judge), and **financial-
domain judgment** (guidance extraction, risk detection, tone analysis, and
quarter-over-quarter comparison). It also handles a real data-availability problem
honestly with a **multi-source ingestion router**, rather than mocking data.

## Architecture

```
                          Ticker (e.g. AAPL)
                                 |
                    +------------v-------------+
                    |      source_router       |   Never mocks. Tries real data,
                    |  FMP transcript?         |   detects gating, falls back.
                    +------+-----------+-------+
                    ok     |           |  gated / no key
                           v           v
                 FMP transcript     SEC EDGAR 8-K (Item 2.02 / Ex-99.1)
                 (incl. Q&A)        earnings press release (no Q&A)
                           +-----+-----+
                                 v
                     chunker  (prepared_remarks / qa / press_release)
                                 v
                     ChromaDB  (PersistentClient, onnx all-MiniLM-L6-v2)
                                 v
        LangGraph StateGraph  (each node: RAG retrieve -> Claude structured JSON)
        +----------+----------+-----------+----------+-----------+
        | guidance |  risk    | sentiment |   qoq     | evaluator |
        +----------+----------+-----------+----------+-----------+
                                 v
        Streamlit UI  --  source badge - live pipeline steps -
        4 collapsible panels w/ inline source quotes - eval scorecard -
        quarter selector + side-by-side compare
```

- **Data (`data/`)** — `fmp_client.py` (real transcripts + runtime gating detection),
  `edgar_client.py` (ticker->CIK, 8-K Item 2.02 -> Exhibit 99.1), and `source_router.py`
  which tries FMP first and falls back to EDGAR.
- **Ingest (`ingest/`)** — `chunker.py` (section-aware, speaker-tagged) and
  `vectorstore.py` (ChromaDB wrappers, idempotent upserts).
- **Agent (`agent/`)** — `state.py`, `rag.py`, five nodes under `nodes/`, wired by
  `graph.py`.
- **LLM (`llm/`)** — `claude.py` (structured-output call with a prompt-JSON fallback,
  refusal-safe, prompt-cached) and `schemas.py`.
- **UI (`app.py`)** — dark-mode Streamlit front end.

## The honest data caveat

FMP's **free** tier generally gates transcript *bodies* (a premium feature), and SEC
EDGAR does **not** host earnings-call transcripts at all. So with only free keys the
app runs on real **8-K earnings press releases** (prepared results + guidance
language) — **not** verbatim transcripts, and with **no analyst Q&A**. The UI labels
the active source on every run. The same code **auto-upgrades** to full transcripts +
Q&A the moment a paid FMP key is present — no code change.

## Run it locally (3 commands)

```bash
python -m venv venv && venv\Scripts\activate      # (a venv is already present here)
pip install -r requirements.txt
streamlit run app.py
```

Before the last command, copy the environment template and fill it in:

```bash
copy .env.example .env      # macOS/Linux: cp .env.example .env
```

| Variable            | Where to get it                                                             | Required |
| ------------------- | --------------------------------------------------------------------------- | -------- |
| `ANTHROPIC_API_KEY` | https://console.anthropic.com/ -> API Keys                                  | yes      |
| `FMP_API_KEY`       | https://site.financialmodelingprep.com/ (free tier is fine)                 | optional |
| `SEC_USER_AGENT`    | any `"AppName your-email@example.com"` string (SEC requires a contact UA)   | yes      |

Without `FMP_API_KEY` the app runs in EDGAR mode automatically.

## Example output

Enter a ticker (e.g. `AAPL`). The app fetches the last four quarters, indexes them,
and streams the pipeline live:

```
[ok] Revenue Guidance Extractor
[ok] Risk Factor Detector
[ok] Sentiment Analyzer
[ok] Quarter-over-Quarter Comparator
[ok] Eval Scorer
```

Then it renders four collapsible panels — **Revenue Guidance**, **Risk Factors**,
**Management Sentiment**, **Quarter-over-Quarter** — each finding paired with the
verbatim source quote it was drawn from, and each panel carrying a grounding badge
(e.g. `100% grounded`). An **eval scorecard** shows Groundedness / Completeness /
Consistency / Overall, and a **Compare** tab places any two quarters side by side.

## Tech stack

Python 3.11 · Streamlit · LangGraph · Anthropic Claude (`claude-opus-4-8`,
`claude-haiku-4-5` judge) · ChromaDB (onnx `all-MiniLM-L6-v2`, no API key) ·
Financial Modeling Prep + SEC EDGAR.

## Tests

```bash
pytest
```

Unit tests cover FMP gating detection, EDGAR helpers, the chunker's section tagging,
and the rule-based groundedness eval — all offline (no network or API key required).

## Notes & Windows tips

- First run downloads the MiniLM embedding model (~90 MB).
- ChromaDB's `onnxruntime` may need the Microsoft Visual C++ Redistributable.
- Do **not** add `sentence-transformers` — its `tokenizers` pin conflicts with
  ChromaDB's; the onnx default embedder needs neither it nor torch.
- No API keys are ever hardcoded or committed; `.env` is gitignored.
