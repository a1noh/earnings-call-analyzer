"""Earnings Call Analyzer — Streamlit UI.

A dark, finance-styled front end over the RAG + LangGraph pipeline:
  * ticker input at the top,
  * live step-by-step pipeline progress (not a spinner),
  * four collapsible analysis panels with inline source quotes,
  * eval quality badges on each panel + an overall scorecard,
  * a quarter selector to analyze any quarter and compare two side by side.

Run:  streamlit run app.py
"""
from __future__ import annotations

import hmac
import os

import streamlit as st

import config
from agent.graph import NODE_LABELS, NODE_ORDER, run_analysis, stream_analysis
from data.edgar_client import EdgarError
from data.source_router import SourceRouter
from ingest import vectorstore

st.set_page_config(page_title="Earnings Call Analyzer", page_icon="📊", layout="wide")


# --- optional password gate -------------------------------------------------
def _app_password() -> str | None:
    """Resolve the gate password from Streamlit secrets (cloud) or env (.env).

    Returns None when no password is configured, which leaves the app open — handy
    for local development. Set APP_PASSWORD to require a login before deploying.
    """
    try:
        if "APP_PASSWORD" in st.secrets:
            return st.secrets["APP_PASSWORD"]
    except Exception:
        pass  # no secrets.toml present
    return os.getenv("APP_PASSWORD")


def require_auth() -> None:
    """Block the app behind a shared password when APP_PASSWORD is set."""
    password = _app_password()
    if not password:
        return  # open (no gate configured)
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

# --- session state ----------------------------------------------------------
st.session_state.setdefault("ingest", {})     # ticker -> {source, source_label, note, quarters}
st.session_state.setdefault("results", {})    # (ticker, year, quarter) -> final state


# --- small render helpers ---------------------------------------------------
def grounding_badge(frac) -> str:
    if frac is None:
        return "grounding: n/a"
    pct = int(frac * 100)
    dot = "🟢" if frac >= 0.8 else "🟡" if frac >= 0.5 else "🔴"
    return f"{dot} {pct}% grounded"


def _quote(text: str) -> None:
    st.caption(f"❝ {text} ❞")


def render_guidance(g: dict, badge: str) -> None:
    with st.expander(f"📈 Revenue Guidance  ·  {badge}", expanded=True):
        if not g:
            st.info("No guidance produced.")
            return
        st.markdown(f"**Revenue guidance:** {g.get('revenue_guidance', '—')}")
        st.markdown(f"**EPS / margin guidance:** {g.get('eps_guidance', '—')}")
        st.markdown(f"**Direction:** `{g.get('direction', 'unknown')}`")
        fs = g.get("forward_statements", []) or []
        if fs:
            st.markdown("**Forward-looking statements:**")
            for item in fs:
                st.markdown(f"- {item.get('statement', '')}")
                _quote(item.get("quote", ""))


def render_risks(r: dict, badge: str) -> None:
    with st.expander(f"⚠️ Risk Factors  ·  {badge}", expanded=True):
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
                f"· severity: `{sev}` {flag}"
            )
            _quote(item.get("quote", ""))


def render_sentiment(s: dict, badge: str) -> None:
    with st.expander(f"🎯 Management Sentiment  ·  {badge}", expanded=True):
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
    with st.expander(f"🔀 Quarter-over-Quarter  ·  {badge}", expanded=True):
        if not q:
            st.info("No comparison produced.")
            return
        if not q.get("comparison_available"):
            st.info(q.get("narrative_shift", "No prior quarter available for comparison."))
            return
        st.markdown(
            f"**{q.get('current_quarter', '')}** vs **{q.get('prior_quarter', '')}**"
        )
        st.markdown(f"_{q.get('narrative_shift', '')}_")
        for mc in q.get("metric_changes", []) or []:
            st.markdown(f"- **{mc.get('metric', '')}** — {mc.get('change', '')}")
            _quote(f"current: {mc.get('quote_current', '')}")
            _quote(f"prior: {mc.get('quote_prior', '')}")


def render_scorecard(ev: dict) -> None:
    if not ev:
        return
    st.markdown("### 🧪 Output Quality Evals")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Groundedness", f"{ev['groundedness']['score']:.2f}",
              help="Fraction of quotes traceable to retrieved transcript text.")
    c2.metric("Completeness", f"{ev['completeness']['score']:.2f}",
              help="How many of the four analysis dimensions produced output.")
    c3.metric("Consistency", f"{ev['consistency']['score']:.2f}",
              help="QoQ references both quarters; sentiment agrees with guidance.")
    c4.metric("Overall", f"{ev['overall']:.2f}")

    with st.expander("Eval details"):
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


def render_reasoning(steps: list[dict]) -> None:
    with st.expander("🧠 Agent reasoning steps"):
        for s in steps:
            st.markdown(f"**{s.get('node', '')}** — {s.get('summary', '')}")
            st.caption(
                f"queries: {', '.join(s.get('queries', []))}  ·  "
                f"sections: {', '.join(s.get('sections_used', [])) or '—'}  ·  "
                f"{s.get('num_chunks', 0)} chunks"
            )


def render_results(state: dict) -> None:
    ev = state.get("eval") or {}
    per_node = ev.get("per_node_groundedness", {}) if ev else {}
    render_scorecard(ev)
    if state.get("errors"):
        st.warning("Some steps had issues:\n" + "\n".join(f"- {e}" for e in state["errors"]))
    render_guidance(state.get("guidance") or {}, grounding_badge(per_node.get("guidance")))
    render_risks(state.get("risks") or {}, grounding_badge(per_node.get("risks")))
    render_sentiment(state.get("sentiment") or {}, grounding_badge(per_node.get("sentiment")))
    render_qoq(state.get("qoq") or {}, grounding_badge(per_node.get("qoq")))
    render_reasoning(state.get("reasoning") or [])


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
            lines = [
                f"{'✅' if n in done else '⏳'} {NODE_LABELS[n]}" for n in NODE_ORDER
            ]
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
    with st.status(f"Fetching earnings data for {ticker}…", expanded=True) as status:
        try:
            result = router.fetch(ticker, config.NUM_QUARTERS)
        except EdgarError as exc:
            status.update(label="No data found", state="error")
            st.error(str(exc))
            return
        st.write(result.note)
        st.write(f"Retrieved {len(result.documents)} quarter(s). Indexing…")
        vectorstore.ingest_documents(result.documents)
        quarters = sorted(
            {(d.year, d.quarter) for d in result.documents}, reverse=True
        )
        st.session_state.ingest[ticker] = {
            "source": result.source,
            "source_label": result.source_label,
            "note": result.note,
            "quarters": quarters,
        }
        status.update(label=f"Indexed {len(quarters)} quarter(s) for {ticker}", state="complete")

    # Analyze the most recent quarter live.
    if quarters:
        year, quarter = quarters[0]
        st.markdown("#### Pipeline")
        analyze_quarter(ticker, year, quarter, live=True)


# --- page -------------------------------------------------------------------
st.title("📊 Earnings Call Analyzer")
st.caption(
    "Agentic RAG over earnings calls — LangGraph + Claude + ChromaDB. "
    "Extracts guidance, risks, sentiment, and quarter-over-quarter changes, "
    "each grounded in source quotes and scored by an eval layer."
)

missing = config.missing_keys()
if missing:
    st.error(
        "Missing required environment variable(s): "
        + ", ".join(missing)
        + ". Copy `.env.example` to `.env` and fill in your keys, then restart."
    )

with st.form("ticker_form"):
    col1, col2 = st.columns([3, 1])
    ticker_in = col1.text_input("Stock ticker", value="", placeholder="AAPL, MSFT, NVDA…")
    submitted = col2.form_submit_button("Analyze", use_container_width=True)

if submitted and ticker_in.strip():
    if missing:
        st.stop()
    ingest_ticker(ticker_in.strip().upper())

# --- results for any ingested ticker ---------------------------------------
if st.session_state.ingest:
    ticker = st.selectbox(
        "Company", sorted(st.session_state.ingest.keys()), key="active_ticker"
    )
    meta = st.session_state.ingest[ticker]
    quarters = meta["quarters"]

    badge_color = "🟩" if meta["source"] == "fmp_transcript" else "🟦"
    st.info(f"{badge_color} **Source:** {meta['source_label']}  \n{meta['note']}")

    tab_single, tab_compare = st.tabs(["📄 Single quarter", "🔀 Compare two quarters"])

    with tab_single:
        labels = [f"{q} {y}" for (y, q) in quarters]
        pick = st.selectbox("Quarter", labels, key=f"single_{ticker}")
        y, q = quarters[labels.index(pick)]
        state = analyze_quarter(ticker, y, q, live=False)
        render_results(state)

    with tab_compare:
        if len(quarters) < 2:
            st.info("Need at least two indexed quarters to compare.")
        else:
            labels = [f"{q} {y}" for (y, q) in quarters]
            ca, cb = st.columns(2)
            pa = ca.selectbox("Quarter A", labels, index=0, key=f"cmpA_{ticker}")
            pb = cb.selectbox("Quarter B", labels, index=1, key=f"cmpB_{ticker}")
            if st.button("Compare", use_container_width=True):
                ya, qa = quarters[labels.index(pa)]
                yb, qb = quarters[labels.index(pb)]
                state_a = analyze_quarter(ticker, ya, qa)
                state_b = analyze_quarter(ticker, yb, qb)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader(pa)
                    render_results(state_a)
                with col_b:
                    st.subheader(pb)
                    render_results(state_b)
