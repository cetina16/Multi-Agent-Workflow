"""LangGraph StateGraph builder for the AI Research Assistant.

Graph topology:
    START → planner → human_review → searcher → extractor
          → [fan-out: summarizer per source] → synthesizer → storage_agent → END

Conditional edges:
  - After human_review: if human_feedback is a redirect → back to planner
  - After searcher: if status == "failed" → END
  - After extractor: fan-out to summarizer via Send API
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agents.extractor import extractor_node
from src.agents.human_review import human_review_node
from src.agents.planner import planner_node
from src.agents.searcher import searcher_node
from src.agents.storage_agent import storage_agent_node
from src.agents.summarizer import summarizer_node
from src.agents.synthesizer import synthesizer_node
from src.state import ResearchState

logger = logging.getLogger(__name__)

# Keywords that mean "approved" — anything else is treated as a redirect
_APPROVAL_TOKENS = {"approved", "approve", "ok", "yes", "go", "continue", "proceed", ""}


def _route_after_human_review(state: ResearchState) -> str:
    """If human provided a redirect, loop back to planner; else continue."""
    feedback = (state.get("human_feedback") or "").strip().lower()
    if feedback in _APPROVAL_TOKENS:
        return "searcher"
    return "planner"  # redirect: re-plan with feedback


def _route_after_searcher(state: ResearchState) -> str:
    """Stop the pipeline if search completely failed."""
    if state.get("status") == "failed":
        logger.warning("Searcher returned no results. Terminating pipeline.")
        return END
    return "extractor"


def _fan_out_to_summarizer(state: ResearchState) -> list[Send]:
    """Fan-out: send each extracted source to a separate summarizer node."""
    sources = state.get("extracted_sources", [])
    query = state.get("query", "")
    if not sources:
        # If no sources, go directly to synthesizer
        return [Send("synthesizer", state)]
    return [Send("summarizer", {"source": src, "query": query}) for src in sources]


def build_graph(checkpointer=None) -> Any:
    """Build and compile the research assistant StateGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer (PostgresSaver / MemorySaver).
                      Required for human-in-the-loop (interrupt) support.

    Returns:
        Compiled LangGraph graph.
    """
    builder = StateGraph(ResearchState)

    # --- Register nodes ---
    builder.add_node("planner", planner_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("searcher", searcher_node)
    builder.add_node("extractor", extractor_node)
    builder.add_node("summarizer", summarizer_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("storage_agent", storage_agent_node)

    # --- Edges ---
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "human_review")

    # After human review: route conditionally
    builder.add_conditional_edges(
        "human_review",
        _route_after_human_review,
        {"planner": "planner", "searcher": "searcher"},
    )

    # After searcher: check for failure
    builder.add_conditional_edges(
        "searcher",
        _route_after_searcher,
        {"extractor": "extractor", END: END},
    )

    # After extractor: fan-out to summarizer (one per source)
    builder.add_conditional_edges("extractor", _fan_out_to_summarizer)

    # Summarizer results fan back into synthesizer
    builder.add_edge("summarizer", "synthesizer")

    builder.add_edge("synthesizer", "storage_agent")
    builder.add_edge("storage_agent", END)

    # Compile — interrupt BEFORE human_review so we can inject feedback
    compile_kwargs: dict = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
        compile_kwargs["interrupt_before"] = ["human_review"]

    return builder.compile(**compile_kwargs)
