"""Synthesizer agent: merges all summaries into a structured research report."""

from __future__ import annotations

import json
import logging
import re

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_settings
from src.cost_tracker import CostTracker
from src.state import NodeCost, Report, ResearchState

logger = logging.getLogger(__name__)

_SYSTEM = """You are a research analyst. Synthesize the provided source summaries into a report.

Respond ONLY with valid JSON — no markdown, no code fences, no extra text:
{
  "executive_summary": "2 sentences max",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "sources": [
    {"url": "...", "title": "...", "key_contribution": "10 words max"}
  ],
  "metadata": {
    "num_sources": <int>,
    "confidence": "high|medium|low",
    "gaps": "one sentence"
  }
}"""


def _parse_json(text: str) -> dict:
    """Extract and parse JSON from LLM output robustly.

    Handles: raw JSON, ```json ... ``` fences, leading/trailing noise.
    """
    # 1. Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip ```json ... ``` or ``` ... ``` code fences
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Extract from first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON. Response: {text[:200]!r}")


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

    llm = ChatGroq(
        model=settings.model_name,
        api_key=settings.groq_api_key,
        temperature=0,
        max_tokens=4096,
    )

    node_cost: NodeCost | None = None
    report_data: dict = {}

    try:
        response = await llm.ainvoke(
            [SystemMessage(content=_SYSTEM), HumanMessage(content=user_msg)]
        )
        report_data = _parse_json(response.content)
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
