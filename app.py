"""Earnings Call Analyzer — Streamlit dashboard UI.

A dark, finance-styled dashboard over the RAG + LangGraph pipeline:
  * sidebar controls (ticker input, company + quarter selectors, compare toggle),
  * a KPI header (sentiment, guidance, risks, groundedness, overall eval),
  * a grid of insight cards, each with inline source quotes and a grounding badge,
  * an Architecture view, and a password gate for deployment.

Run:  streamlit run app.py
"""
from __future__ import annotations

import hmac
import os

import streamlit as st

# set_page_config MUST be the first Streamlit command (before any st.* call,
# including reading st.secrets below).
st.set_page_config(page_title="Earnings Call Analyzer", page_icon="📊", layout="wide")

# On Streamlit Community Cloud, secrets live in st.secrets. Bridge them into
# os.environ *before* importing config (which reads env vars) so the same
# env-based configuration works both locally (.env) and on deploy (Secrets).
try:
    for _k in (
        "ANTHROPIC_API_KEY", "FMP_API_KEY", "SEC_USER_AGENT",
        "APP_PASSWORD", "ANALYSIS_MODEL", "JUDGE_MODEL",
    ):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass  # no secrets configured (e.g. local dev without a secrets.toml)

import config
from agent.graph import NODE_LABELS, NODE_ORDER, run_analysis, stream_analysis
from data.edgar_client import EdgarError
from data.source_router import SourceRouter
from ingest import vectorstore

# --- session state ----------------------------------------------------------
st.session_state.setdefault("ingest", {})     # ticker -> {source, source_label, note, quarters}
st.session_state.setdefault("results", {})    # (ticker, year, quarter) -> final state


# --- optional password gate -------------------------------------------------
def _app_password() -> str | None:
    """Resolve the gate password from Streamlit secrets (cloud) or env (.env)."""
    try:
        if "APP_PASSWORD" in st.secrets:
            return st.secrets["APP_PASSWORD"]
    except Exception:
        pass
    return os.getenv("APP_PASSWORD")


def require_auth() -> None:
    """Block the app behind a shared password when APP_PASSWORD is set."""
    password = _app_password()
    if not password:
        return
    if st.session_state.get("authed"):
        return
    st.markdown("## 🔒 Earnings Call Analyzer")
    st.caption("This app is password protected.")
    entered = st.text_input("Password", type="password", key="pw_input")
    if st.button("Enter"):
        if hmac.compare_digest(entered or "", password):
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


require_auth()


# --- small render helpers ---------------------------------------------------
def grounding_badge(frac) -> str:
    if frac is None:
        return "grounding n/a"
    pct = int(frac * 100)
    dot = "🟢" if frac >= 0.8 else "🟡" if frac >= 0.5 else "🔴"
    return f"{dot} {pct}% grounded"


def _quote(text: str) -> None:
    st.caption(f"❝ {text} ❞")


def render_guidance(g: dict, badge: str) -> None:
    st.markdown(f"#### 📈 Revenue Guidance")
    st.caption(badge)
    if not g:
        st.info("No guidance produced.")
        return
    st.markdown(f"**Revenue:** {g.get('revenue_guidance', '—')}")
    st.markdown(f"**EPS / margin:** {g.get('eps_guidance', '—')}")
    st.markdown(f"**Direction:** `{g.get('direction', 'unknown')}`")
    fs = g.get("forward_statements", []) or []
    if fs:
        st.markdown("**Forward-looking statements:**")
        for item in fs:
            st.markdown(f"- {item.get('statement', '')}")
            _quote(item.get("quote", ""))


def render_risks(r: dict, badge: str) -> None:
    st.markdown(f"#### ⚠️ Risk Factors")
    st.caption(badge)
    risks = (r or {}).get("risks", []) or []
    if not risks:
        st.info("No risks detected.")
        return
    sev_color = {"high": "🔴", "medium": "🟠", "low": "🟡"}
    for item in risks:
        flag = "🆕" if item.get("is_new_or_escalating") else ""
        sev = item.get("severity", "low")
        st.markdown(
            f"{sev_color.get(sev, '⚪')} **{item.get('risk', '')}** "
            f"· `{sev}` {flag}"
        )
        _quote(item.get("quote", ""))


def render_sentiment(s: dict, badge: str) -> None:
    st.markdown(f"#### 🎯 Management Sentiment")
    st.caption(badge)
    if not s:
        st.info("No sentiment produced.")
        return
    label = s.get("label", "unknown")
    icon = {"bullish": "🟢", "neutral": "⚪", "cautious": "🔴"}.get(label, "⚪")
    st.markdown(f"**Tone:** {icon} `{label}`  ·  **score:** {s.get('score', '—')}")
    st.markdown(f"_{s.get('rationale', '')}_")
    drivers = s.get("drivers", []) or []
    if drivers:
        st.markdown("**Drivers:**")
        for d in drivers:
            st.markdown(f"- {d.get('driver', '')}")
            _quote(d.get("quote", ""))


def render_qoq(q: dict, badge: str) -> None:
    st.markdown(f"#### 🔀 Quarter-over-Quarter")
    st.caption(badge)
    if not q:
        st.info("No comparison produced.")
        return
    if not q.get("comparison_available"):
        st.info(q.get("narrative_shift", "No prior quarter available for comparison."))
        return
    st.markdown(f"**{q.get('current_quarter', '')}** vs **{q.get('prior_quarter', '')}**")
    st.markdown(f"_{q.get('narrative_shift', '')}_")
    for mc in q.get("metric_changes", []) or []:
        st.markdown(f"- **{mc.get('metric', '')}** — {mc.get('change', '')}")
        _quote(f"current: {mc.get('quote_current', '')}")
        _quote(f"prior: {mc.get('quote_prior', '')}")


def render_reasoning(steps: list[dict]) -> None:
    with st.expander("🧠 Agent reasoning steps"):
        for s in steps:
            st.markdown(f"**{s.get('node', '')}** — {s.get('summary', '')}")
            st.caption(
                f"queries: {', '.join(s.get('queries', []))}  ·  "
                f"sections: {', '.join(s.get('sections_used', [])) or '—'}  ·  "
                f"{s.get('num_chunks', 0)} chunks"
            )


def render_eval_details(ev: dict) -> None:
    if not ev:
        st.info("No eval available.")
        return
    g = ev["groundedness"]
    st.write(f"**Groundedness:** {g['supported']}/{g['total']} quotes supported.")
    if g["unsupported"]:
        st.write("Unsupported quotes:")
        for q in g["unsupported"]:
            st.caption(f"• {q}")
    if ev["consistency"]["flags"]:
        st.write("**Consistency flags:**")
        for f in ev["consistency"]["flags"]:
            st.caption(f"• {f}")
    if ev["completeness"]["missing"]:
        st.write("Missing dimensions: " + ", ".join(ev["completeness"]["missing"]))
    if ev.get("judge"):
        j = ev["judge"]
        st.write(f"**LLM judge groundedness:** {j.get('groundedness', '—')}")
        st.caption(j.get("rationale", ""))


def render_kpis(state: dict) -> None:
    """A row of headline metric cards summarizing the quarter."""
    ev = state.get("eval") or {}
    s = state.get("sentiment") or {}
    g = state.get("guidance") or {}
    risks = (state.get("risks") or {}).get("risks", []) or []
    new_risks = sum(1 for x in risks if x.get("is_new_or_escalating"))

    c = st.columns(5)
    label = (s.get("label") or "—")
    c[0].metric("Sentiment", label.capitalize(), help=f"Tone score: {s.get('score', '—')}")
    c[1].metric("Guidance", (g.get("direction") or "—").capitalize())
    c[2].metric("Risks", len(risks), help=f"{new_risks} new/escalating")
    ground = ev.get("groundedness", {}).get("score") if ev else None
    c[3].metric("Groundedness", f"{ground:.0%}" if ground is not None else "—")
    overall = ev.get("overall") if ev else None
    c[4].metric("Overall eval", f"{overall:.2f}" if overall is not None else "—")


def _placeholder_card(title: str) -> None:
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.caption("Awaiting analysis…")


def _skeleton_grid(compact: bool) -> None:
    titles = [
        "📈 Revenue Guidance", "🎯 Management Sentiment",
        "⚠️ Risk Factors", "🔀 Quarter-over-Quarter",
    ]
    if compact:
        for t in titles:
            _placeholder_card(t)
    else:
        left, right = st.columns(2)
        with left:
            _placeholder_card(titles[0])
            _placeholder_card(titles[1])
        with right:
            _placeholder_card(titles[2])
            _placeholder_card(titles[3])


def render_dashboard(state: dict, quarter_label: str | None = None, compact: bool = False) -> None:
    """The main dashboard: KPI header + card grid + details.

    The layout is ALWAYS rendered — when no analysis has run yet it shows a
    skeleton (dashes + placeholder cards) and only the content fills in on search.
    """
    if state and quarter_label:
        st.markdown(f"### {state.get('ticker', '')} · {quarter_label}")
    else:
        st.markdown("### Dashboard")
        st.caption("Enter a ticker in the sidebar and click **Analyze** to populate the cards.")
    render_kpis(state or {})
    st.divider()

    if not state:
        _skeleton_grid(compact)
        with st.expander("🧪 Eval details"):
            st.caption("Awaiting analysis…")
        return

    if state.get("errors"):
        st.warning("Some steps had issues:\n" + "\n".join(f"- {e}" for e in state["errors"]))

    ev = state.get("eval") or {}
    per = ev.get("per_node_groundedness", {}) if ev else {}
    guidance = (state.get("guidance") or {}, grounding_badge(per.get("guidance")))
    risks = (state.get("risks") or {}, grounding_badge(per.get("risks")))
    sentiment = (state.get("sentiment") or {}, grounding_badge(per.get("sentiment")))
    qoq = (state.get("qoq") or {}, grounding_badge(per.get("qoq")))

    if compact:
        for fn, data in (
            (render_guidance, guidance), (render_sentiment, sentiment),
            (render_risks, risks), (render_qoq, qoq),
        ):
            with st.container(border=True):
                fn(*data)
    else:
        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                render_guidance(*guidance)
            with st.container(border=True):
                render_sentiment(*sentiment)
        with right:
            with st.container(border=True):
                render_risks(*risks)
            with st.container(border=True):
                render_qoq(*qoq)

    with st.expander("🧪 Eval details"):
        render_eval_details(ev)
    render_reasoning(state.get("reasoning") or [])


# --- architecture view ------------------------------------------------------
_ARCH_DOT = """
digraph G {
  rankdir=TB;
  bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#1a1f2b",
        fontcolor="#e6e9ef", color="#4c9be8", fontname="Helvetica", fontsize=11];
  edge [color="#7f8ea3", fontcolor="#9aa4b2", fontname="Helvetica", fontsize=9];

  ticker [label="Ticker (e.g. AAPL)", fillcolor="#22303f"];
  router [label="source_router\\n(tries real data, detects gating)"];
  fmp    [label="FMP transcript\\n(incl. analyst Q&A)"];
  edgar  [label="SEC EDGAR 8-K\\n(press release, no Q&A)"];
  chunk  [label="Section-aware chunker\\n(prepared / qa / press_release)"];
  chroma [label="ChromaDB\\n(local onnx MiniLM embeddings)", fillcolor="#22303f"];
  g [label="Guidance node"];
  r [label="Risk node"];
  s [label="Sentiment node"];
  q [label="QoQ node"];
  e [label="Eval scorer\\n(groundedness / completeness / consistency)"];
  ui [label="Streamlit dashboard\\n(KPIs + cards + inline quotes)", fillcolor="#22303f"];

  ticker -> router;
  router -> fmp   [label="ok"];
  router -> edgar [label="gated / no key"];
  fmp -> chunk;
  edgar -> chunk;
  chunk -> chroma;

  chroma -> g;
  chroma -> r [style=dashed, label="RAG"];
  chroma -> s [style=dashed];
  chroma -> q [style=dashed];
  g -> r -> s -> q -> e -> ui;
}
"""

_ARCH_ASCII = """Ticker -> source_router --(ok)------> FMP transcript (incl. Q&A) ---+
                       \\--(gated/no key)-> SEC EDGAR 8-K (no Q&A) ----+
                                                                       v
                                                        Section-aware chunker
                                                                       v
                                             ChromaDB (onnx MiniLM, local)
                                                                       v
   LangGraph agent (each node RAG-retrieves then calls Claude):
     Guidance -> Risk -> Sentiment -> QoQ -> Eval scorer
                                                                       v
                     Streamlit dashboard: KPIs + cards + inline quotes
"""


def render_architecture() -> None:
    st.markdown("### 🏗️ How this app is built")
    st.caption(
        "An agentic RAG pipeline: a multi-source ingestion router feeds a local "
        "vector store, a LangGraph agent runs four grounded analysis nodes, and an "
        "eval layer scores the output — all rendered in this dark dashboard."
    )
    try:
        st.graphviz_chart(_ARCH_DOT, use_container_width=True)
    except Exception:
        st.code(_ARCH_ASCII, language="text")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 1 · Data ingestion (`data/`)")
        st.markdown(
            "- **`source_router`** tries FMP for a real transcript, detects gating "
            "at runtime, and falls back to SEC EDGAR — **never mocks data**.\n"
            "- **`fmp_client`** fetches transcript bodies and classifies gating.\n"
            "- **`edgar_client`** resolves ticker→CIK, finds 8-K Item 2.02 filings, "
            "and extracts the Exhibit 99.1 earnings press release."
        )
        st.markdown("#### 2 · RAG (`ingest/`)")
        st.markdown(
            "- **`chunker`** splits on speaker/section boundaries and tags each chunk.\n"
            "- **`vectorstore`** is ChromaDB with the on-device **onnx MiniLM** "
            "embedder — no embedding API, no cloud."
        )
    with col_b:
        st.markdown("#### 3 · Agent (`agent/`)")
        st.markdown(
            "- A **LangGraph `StateGraph`** runs nodes in sequence: "
            "**guidance → risk → sentiment → QoQ → eval**.\n"
            "- Each node retrieves quarter-scoped chunks (RAG) and calls Claude "
            "(`claude-opus-4-8`) for **structured JSON**, backed by verbatim quotes."
        )
        st.markdown("#### 4 · Eval + UI")
        st.markdown(
            "- **Eval layer** scores *groundedness*, *completeness*, and *consistency*, "
            "plus an optional `claude-haiku-4-5` judge.\n"
            "- **Dashboard** streams live pipeline progress and renders KPI cards + "
            "insight cards with inline quotes and grounding badges."
        )
    st.markdown(
        "**Stack:** Python 3.11 · Streamlit · LangGraph · Anthropic Claude · "
        "ChromaDB (onnx MiniLM) · Financial Modeling Prep + SEC EDGAR."
    )


# --- analysis orchestration -------------------------------------------------
def analyze_quarter(ticker: str, year: int, quarter: str, live: bool = False) -> dict:
    """Run (or reuse cached) analysis for one quarter."""
    key = (ticker, year, quarter)
    if key in st.session_state.results:
        return st.session_state.results[key]

    meta = st.session_state.ingest[ticker]
    source, label = meta["source"], meta["source_label"]

    if live:
        placeholder = st.empty()
        done: list[str] = []
        final: dict = {}
        for node_name, state in stream_analysis(ticker, year, quarter, source, label):
            done.append(node_name)
            final = state
            lines = [f"{'✅' if n in done else '⏳'} {NODE_LABELS[n]}" for n in NODE_ORDER]
            placeholder.markdown("  \n".join(lines))
        st.session_state.results[key] = final
        return final

    with st.spinner(f"Analyzing {quarter} {year}…"):
        final = run_analysis(ticker, year, quarter, source, label)
    st.session_state.results[key] = final
    return final


def ingest_ticker(ticker: str) -> None:
    """Fetch + index earnings docs for a ticker, then analyze the latest quarter."""
    router = SourceRouter()
    with st.status(f"Analyzing {ticker}…", expanded=True) as status:
        try:
            result = router.fetch(ticker, config.NUM_QUARTERS)
        except EdgarError as exc:
            status.update(label="No data found", state="error")
            st.error(str(exc))
            return
        st.write(result.note)
        st.write(f"Retrieved {len(result.documents)} quarter(s). Indexing…")
        vectorstore.ingest_documents(result.documents)
        quarters = sorted({(d.year, d.quarter) for d in result.documents}, reverse=True)
        st.session_state.ingest[ticker] = {
            "source": result.source,
            "source_label": result.source_label,
            "note": result.note,
            "quarters": quarters,
        }
        st.session_state["sel_company"] = ticker  # make the new ticker active
        if quarters:
            st.write("Running analysis pipeline:")
            year, quarter = quarters[0]
            analyze_quarter(ticker, year, quarter, live=True)
        status.update(label=f"{ticker} ready", state="complete", expanded=False)


# --- sidebar + page ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 Earnings Call Analyzer")
    st.caption("Agentic RAG · LangGraph · Claude · ChromaDB")
    page = st.radio("View", ["Dashboard", "Architecture"], label_visibility="collapsed")
    st.divider()
    with st.form("ticker_form"):
        ticker_in = st.text_input("Stock ticker", placeholder="AAPL, MSFT, NVDA…")
        submitted = st.form_submit_button("▶  Analyze", use_container_width=True)

missing = config.missing_keys()

if page == "Architecture":
    render_architecture()
else:
    if missing:
        st.error(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". On Streamlit Cloud add them under Settings → Secrets; locally copy "
            "`.env.example` to `.env`."
        )

    if submitted and ticker_in.strip() and not missing:
        ingest_ticker(ticker_in.strip().upper())

    active_state: dict = {}
    active_label = None
    compare_pair = None
    if st.session_state.ingest:
        with st.sidebar:
            companies = sorted(st.session_state.ingest.keys())
            active = st.selectbox("Company", companies, key="sel_company")
            meta = st.session_state.ingest[active]
            quarters = meta["quarters"]
            labels = [f"{qq} {yy}" for (yy, qq) in quarters]
            pick = st.selectbox("Quarter", labels, key=f"q_{active}")
            compare = st.checkbox("Compare a 2nd quarter", key=f"cmp_{active}")
            pick2 = None
            if compare and len(quarters) > 1:
                others = [lbl for lbl in labels if lbl != pick]
                pick2 = st.selectbox("Compare against", others, key=f"q2_{active}")
            st.divider()
            badge = "🟩" if meta["source"] == "fmp_transcript" else "🟦"
            st.caption(f"{badge} **Source:** {meta['source_label']}")
            st.caption(meta["note"])

        y, q = quarters[labels.index(pick)]
        if compare and pick2:
            y2, q2 = quarters[labels.index(pick2)]
            compare_pair = (
                analyze_quarter(active, y, q), pick,
                analyze_quarter(active, y2, q2), pick2,
            )
        else:
            active_state = analyze_quarter(active, y, q)
            active_label = pick

    # The dashboard layout is always rendered — a skeleton until analysis fills it.
    if compare_pair:
        sa, la, sb, lb = compare_pair
        col_a, col_b = st.columns(2)
        with col_a:
            render_dashboard(sa, la, compact=True)
        with col_b:
            render_dashboard(sb, lb, compact=True)
    else:
        render_dashboard(active_state, active_label)
