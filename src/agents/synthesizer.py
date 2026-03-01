"""Synthesizer agent: merges all summaries into a structured research report."""

from __future__ import annotations

import json
import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_settings
from src.cost_tracker import CostTracker
from src.state import NodeCost, Report, ResearchState

logger = logging.getLogger(__name__)

_SYSTEM = """You are a research analyst. Given a research query, a research plan,
and summaries of multiple web sources, synthesize a comprehensive research report.

Respond ONLY with valid JSON in this exact format:
{
  "executive_summary": "2-3 sentence overview",
  "key_findings": ["finding 1", "finding 2", "finding 3", ...],
  "sources": [
    {"url": "...", "title": "...", "key_contribution": "one sentence"}
  ],
  "metadata": {
    "num_sources": <int>,
    "confidence": "high|medium|low",
    "gaps": "any notable gaps in the research"
  }
}"""


async def synthesizer_node(state: ResearchState) -> dict:
    """Synthesize all source summaries into a structured final report."""
    settings = get_settings()
    tracker = CostTracker()

    query = state["query"]
    plan = state.get("plan", [])
    summaries = state.get("summaries", [])

    if not summaries:
        logger.warning("Synthesizer: no summaries available.")
        report = Report(
            query=query,
            executive_summary="No sources could be summarized.",
            key_findings=[],
            sources=[],
            metadata={"num_sources": 0, "confidence": "low", "gaps": "No data available"},
        )
        return {"final_report": report, "status": "complete"}

    summaries_text = "\n\n".join(
        f"[{i + 1}] {s['title']} ({s['url']})\n{s['summary']}"
        for i, s in enumerate(summaries)
    )

    user_msg = (
        f"Research query: {query}\n\n"
        f"Research plan:\n" + "\n".join(f"- {p}" for p in plan) + "\n\n"
        f"Source summaries:\n{summaries_text}"
    )

    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        temperature=0,
        max_tokens=2048,
    )

    node_cost: NodeCost | None = None
    report_data: dict = {}

    try:
        response = await llm.ainvoke(
            [SystemMessage(content=_SYSTEM), HumanMessage(content=user_msg)]
        )
        report_data = json.loads(response.content)
        cost = tracker.record_from_message("synthesizer", response)
        node_cost = NodeCost(
            node=cost.node,
            tokens_in=cost.tokens_in,
            tokens_out=cost.tokens_out,
            cost_usd=cost.cost_usd,
        )
    except Exception as exc:
        logger.error("Synthesizer failed: %s", exc)
        report_data = {
            "executive_summary": f"Synthesis failed: {exc}",
            "key_findings": [],
            "sources": [],
            "metadata": {"num_sources": len(summaries), "confidence": "low", "gaps": str(exc)},
        }

    report = Report(
        query=query,
        executive_summary=report_data.get("executive_summary", ""),
        key_findings=report_data.get("key_findings", []),
        sources=report_data.get("sources", []),
        metadata=report_data.get("metadata", {}),
    )

    result: dict = {"final_report": report, "status": "complete"}
    if node_cost is not None:
        result["node_costs"] = [node_cost]
    return result
