"""Streamlit web interface for the Multi-Agent Research Workflow."""

from __future__ import annotations

import asyncio
import logging
import uuid
import warnings

warnings.filterwarnings("ignore", category=ResourceWarning)
logging.getLogger("primp").setLevel(logging.ERROR)
logging.basicConfig(level=logging.WARNING)

import streamlit as st
from langgraph.checkpoint.memory import MemorySaver

from config import get_settings
from src.database.engine import init_db
from src.graph import build_graph

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍",
    layout="wide",
)

# ── Session state init ────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "stage": "idle",          # idle | awaiting_human | running | done
        "session_id": None,
        "query": "",
        "plan": [],
        "search_queries": [],
        "completed_nodes": [],
        "report": None,
        "errors": [],
        "checkpointer": None,
        "graph": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Build graph once per browser session
    if st.session_state.checkpointer is None:
        st.session_state.checkpointer = MemorySaver()
        st.session_state.graph = build_graph(
            checkpointer=st.session_state.checkpointer
        )


_init_state()

# ── Async helpers ─────────────────────────────────────────────────────────────
async def _phase1(query: str, session_id: str, graph) -> dict:
    """Run planner → pause at human_review. Returns state snapshot values."""
    await init_db()
    config = {"configurable": {"thread_id": session_id}}
    initial_state = {
        "query": query,
        "session_id": session_id,
        "plan": [],
        "search_queries": [],
        "search_results": [],
        "extracted_sources": [],
        "summaries": [],
        "final_report": None,
        "human_feedback": None,
        "retry_counts": {},
        "error_log": [],
        "node_costs": [],
        "status": "running",
    }
    nodes: list[str] = []
    async for event in graph.astream(initial_state, config, stream_mode="updates"):
        for node_name in event:
            if node_name != "__interrupt__":
                nodes.append(node_name)

    snapshot = await graph.aget_state(config)
    return {"snapshot": snapshot, "nodes": nodes}


async def _phase2(session_id: str, feedback: str, graph) -> dict:
    """Resume after human review → run until done or next interrupt.

    Returns 'next' so the caller can detect if the graph paused at
    human_review again (redirect path) instead of completing.
    """
    config = {"configurable": {"thread_id": session_id}}
    await graph.aupdate_state(config, {"human_feedback": feedback})

    nodes: list[str] = []
    async for event in graph.astream(None, config, stream_mode="updates"):
        for node_name in event:
            if node_name != "__interrupt__":
                nodes.append(node_name)

    final = await graph.aget_state(config)
    return {"values": final.values, "nodes": nodes, "next": list(final.next or [])}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("AI Research Assistant")
    st.caption("LangGraph · Groq · DuckDuckGo")
    st.divider()
    st.markdown("**Stack**")
    st.markdown("- LLM: `llama-3.3-70b-versatile` (Groq free)")
    st.markdown("- Search: DuckDuckGo (no API key)")
    st.markdown("- Orchestration: LangGraph")
    st.divider()
    if st.session_state.session_id:
        st.caption(f"Session: `{st.session_state.session_id[:8]}…`")
    if st.session_state.stage != "idle":
        if st.button("New Research", use_container_width=True):
            for key in ["stage", "session_id", "query", "plan", "search_queries",
                        "completed_nodes", "report", "errors"]:
                st.session_state[key] = [] if key in ("plan", "search_queries",
                                                        "completed_nodes", "errors") else \
                                         None if key in ("report", "session_id") else \
                                         "idle" if key == "stage" else ""
            st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
stage = st.session_state.stage

# ── IDLE ──────────────────────────────────────────────────────────────────────
if stage == "idle":
    st.header("What do you want to research?")
    query = st.text_area(
        "Research query",
        placeholder="e.g. Latest advances in quantum computing",
        height=100,
        label_visibility="collapsed",
    )
    auto_approve = st.checkbox("Auto-approve plan (skip review step)")

    if st.button("Start Research", type="primary", disabled=not query.strip()):
        st.session_state.query = query.strip()
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.completed_nodes = []

        with st.spinner("Planning your research…"):
            result = asyncio.run(
                _phase1(
                    st.session_state.query,
                    st.session_state.session_id,
                    st.session_state.graph,
                )
            )

        snapshot = result["snapshot"]
        st.session_state.completed_nodes.extend(result["nodes"])

        if snapshot.next and "human_review" in snapshot.next:
            st.session_state.plan = snapshot.values.get("plan", [])
            st.session_state.search_queries = snapshot.values.get("search_queries", [])
            if auto_approve:
                st.session_state.stage = "running"
                st.session_state._pending_feedback = "approved"
            else:
                st.session_state.stage = "awaiting_human"
        else:
            # Graph completed without interrupting (no checkpointer edge case)
            st.session_state.report = snapshot.values.get("final_report")
            st.session_state.errors = snapshot.values.get("error_log", [])
            st.session_state.stage = "done"

        st.rerun()

# ── AWAITING HUMAN ────────────────────────────────────────────────────────────
elif stage == "awaiting_human":
    st.header("Review the Research Plan")

    col_plan, col_queries = st.columns(2)
    with col_plan:
        st.subheader("Plan")
        for item in st.session_state.plan:
            st.markdown(f"- {item}")
    with col_queries:
        st.subheader("Search Queries")
        for q in st.session_state.search_queries:
            st.markdown(f"- `{q}`")

    st.divider()
    st.subheader("What would you like to do?")

    choice = st.radio(
        "Action",
        options=["Approve — run with this plan", "Redirect — re-plan with my feedback"],
        label_visibility="collapsed",
    )

    if choice.startswith("Redirect"):
        redirect_text = st.text_area(
            "Your feedback to the planner",
            placeholder="e.g. Focus only on open-source solutions",
            height=80,
        )
        btn_label = "Send Feedback & Re-plan"
        pending = redirect_text.strip() or "approved"
    else:
        redirect_text = ""
        btn_label = "Approve & Start Research"
        pending = "approved"

    if st.button(btn_label, type="primary", disabled=(choice.startswith("Redirect") and not redirect_text.strip())):
        st.session_state.stage = "running"
        st.session_state._pending_feedback = pending
        st.rerun()

# ── RUNNING ───────────────────────────────────────────────────────────────────
elif stage == "running":
    st.header("Running Research Pipeline")
    feedback = st.session_state.pop("_pending_feedback", "approved")

    node_display = {
        "human_review": "Human Review",
        "searcher": "Searching the web",
        "extractor": "Extracting sources",
        "summarizer": "Summarizing sources",
        "synthesizer": "Synthesizing report",
        "storage_agent": "Saving results",
    }

    with st.status("Running agents…", expanded=True) as status:
        for node in st.session_state.completed_nodes:
            label = node_display.get(node, node)
            st.write(f"✓ {label}")

        result = asyncio.run(
            _phase2(
                st.session_state.session_id,
                feedback,
                st.session_state.graph,
            )
        )

        for node in result["nodes"]:
            label = node_display.get(node, node)
            st.write(f"✓ {label}")

        status.update(label="Done!", state="complete")

    final_values = result["values"]

    if "human_review" in result["next"]:
        # Redirect path: planner re-ran and is waiting for human review again
        st.session_state.plan = final_values.get("plan", [])
        st.session_state.search_queries = final_values.get("search_queries", [])
        st.session_state.completed_nodes.extend(result["nodes"])
        st.session_state.stage = "awaiting_human"
    else:
        st.session_state.report = final_values.get("final_report")
        st.session_state.errors = final_values.get("error_log", [])
        st.session_state.stage = "done"
    st.rerun()

# ── DONE ──────────────────────────────────────────────────────────────────────
elif stage == "done":
    report = st.session_state.report

    if not report:
        st.error("Research completed but no report was generated.")
        if st.session_state.errors:
            for e in st.session_state.errors:
                st.warning(f"[{e.get('node', '?')}] {e.get('error', '')}")
    else:
        st.success(f"Research complete: **{report.get('query', '')}**")
        st.divider()

        # Executive summary
        st.subheader("Executive Summary")
        st.write(report.get("executive_summary", ""))

        # Key findings
        findings = report.get("key_findings", [])
        if findings:
            st.subheader("Key Findings")
            for i, f in enumerate(findings, 1):
                st.markdown(f"**{i}.** {f}")

        # Sources
        sources = report.get("sources", [])
        if sources:
            st.subheader("Sources")
            for s in sources:
                url = s.get("url", "")
                title = s.get("title", url)
                contribution = s.get("key_contribution", "")
                st.markdown(f"- [{title}]({url}) — {contribution}")

        # Metadata footer
        meta = report.get("metadata", {})
        if meta:
            st.divider()
            st.caption(
                f"Confidence: **{meta.get('confidence', 'N/A')}** · "
                f"Sources: **{meta.get('num_sources', 0)}** · "
                f"Gaps: {meta.get('gaps', 'None noted')}"
            )

    if st.session_state.errors:
        with st.expander("Errors encountered"):
            for e in st.session_state.errors:
                st.error(f"[{e.get('node', '?')}] {e.get('error', '')}")
