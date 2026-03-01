"""Summarizer agent: summarizes a single source with Claude.

This node is designed for LangGraph's Send API (fan-out), receiving
one source at a time via a sub-state dict: {"source": Source, "query": str}.
"""

from __future__ import annotations

import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_settings
from src.cost_tracker import CostTracker
from src.state import NodeCost, Source, Summary

logger = logging.getLogger(__name__)

_SYSTEM = """You are a research summarizer. Given a web source and a research query,
write a concise, factual summary (3-5 sentences) focusing on information relevant
to the query. Do not add opinions or information not present in the source."""


async def summarizer_node(state: dict) -> dict:
    """Summarize one source. Receives: {source: Source, query: str}."""
    settings = get_settings()
    tracker = CostTracker()

    source: Source = state["source"]
    query: str = state.get("query", "")

    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        temperature=0,
        max_tokens=512,
    )

    user_msg = (
        f"Research query: {query}\n\n"
        f"Source URL: {source['url']}\n"
        f"Source title: {source.get('title', 'N/A')}\n\n"
        f"Content:\n{source['content'][:4000]}"
    )

    node_cost: NodeCost | None = None
    summary_text = ""

    try:
        response = await llm.ainvoke(
            [SystemMessage(content=_SYSTEM), HumanMessage(content=user_msg)]
        )
        summary_text = response.content
        cost = tracker.record_from_message("summarizer", response)
        node_cost = NodeCost(
            node=cost.node,
            tokens_in=cost.tokens_in,
            tokens_out=cost.tokens_out,
            cost_usd=cost.cost_usd,
        )
    except Exception as exc:
        logger.error("Summarizer failed for %s: %s", source["url"], exc)
        summary_text = f"[Summary unavailable: {exc}]"

    summary = Summary(
        url=source["url"],
        title=source.get("title", ""),
        summary=summary_text,
    )

    result: dict = {"summaries": [summary]}
    if node_cost is not None:
        result["node_costs"] = [node_cost]
    return result
