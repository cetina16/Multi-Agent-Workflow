"""Planner agent: decomposes a research query into sub-tasks and search queries."""

from __future__ import annotations

import json
import logging

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_settings
from src.cost_tracker import CostTracker
from src.state import NodeCost, ResearchState

logger = logging.getLogger(__name__)

_SYSTEM = """You are a research planning expert. Given a research query, decompose it into:
1. A list of focused sub-tasks (2-3 items)
2. A list of specific web search queries (exactly 3 items, no more) to gather information

Respond ONLY with valid JSON in this exact format:
{
  "plan": ["sub-task 1", "sub-task 2"],
  "search_queries": ["query 1", "query 2", "query 3"]
}"""


async def planner_node(state: ResearchState) -> dict:
    """Decompose the research query into a plan and search queries."""
    settings = get_settings()
    tracker = CostTracker()

    query = state["query"]
    human_feedback = state.get("human_feedback")

    # Incorporate human redirect feedback if provided
    user_content = query
    if human_feedback and human_feedback.lower() not in ("approved", "approve", "ok", "yes"):
        user_content = (
            f"Original query: {query}\n\n"
            f"Human reviewer feedback/redirect: {human_feedback}\n\n"
            "Please adjust the research plan based on this feedback."
        )

    llm = ChatGroq(
        model=settings.model_name,
        api_key=settings.groq_api_key,
        temperature=0,
    )

    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=user_content),
    ]

    try:
        response = await llm.ainvoke(messages)
        data = json.loads(response.content)
        plan: list[str] = data.get("plan", [])
        search_queries: list[str] = data.get("search_queries", [])
    except Exception as exc:
        logger.error("Planner failed: %s", exc)
        plan = [f"Research: {query}"]
        search_queries = [query]
        response = None

    node_cost: NodeCost | None = None
    if response is not None:
        cost = tracker.record_from_message("planner", response)
        node_cost = NodeCost(
            node=cost.node,
            tokens_in=cost.tokens_in,
            tokens_out=cost.tokens_out,
            cost_usd=cost.cost_usd,
        )

    result: dict = {
        "plan": plan,
        "search_queries": search_queries,
        "status": "running",
        # Clear previous human feedback so it doesn't re-trigger
        "human_feedback": None,
    }
    if node_cost is not None:
        result["node_costs"] = [node_cost]

    return result
