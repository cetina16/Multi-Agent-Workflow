"""Tests for the planner agent."""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.planner import planner_node


def _make_mock_response(plan: list, queries: list) -> MagicMock:
    msg = MagicMock()
    msg.content = json.dumps({"plan": plan, "search_queries": queries})
    msg.usage_metadata = {"input_tokens": 100, "output_tokens": 80}
    return msg


@pytest.mark.asyncio
async def test_planner_returns_plan_and_queries(base_state):
    mock_resp = _make_mock_response(
        plan=["Understand quantum basics", "Find recent advances"],
        queries=["quantum computing 2024", "quantum error correction breakthroughs"],
    )

    with patch("src.agents.planner.ChatGroq") as MockLLM:
        MockLLM.return_value.ainvoke = AsyncMock(return_value=mock_resp)
        result = await planner_node(base_state)

    assert len(result["plan"]) == 2
    assert len(result["search_queries"]) == 2
    assert result["status"] == "running"
    assert result["human_feedback"] is None


@pytest.mark.asyncio
async def test_planner_incorporates_feedback(base_state):
    base_state["human_feedback"] = "Focus more on practical applications"
    mock_resp = _make_mock_response(
        plan=["Find practical quantum use cases"],
        queries=["quantum computing practical applications industry"],
    )

    with patch("src.agents.planner.ChatGroq") as MockLLM:
        instance = MockLLM.return_value
        instance.ainvoke = AsyncMock(return_value=mock_resp)
        result = await planner_node(base_state)

    # Verify feedback was included in the call
    call_args = instance.ainvoke.call_args[0][0]
    human_msg = call_args[1]
    assert "Focus more on practical applications" in human_msg.content
    assert result["human_feedback"] is None  # cleared after use


@pytest.mark.asyncio
async def test_planner_fallback_on_llm_error(base_state):
    with patch("src.agents.planner.ChatGroq") as MockLLM:
        MockLLM.return_value.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
        result = await planner_node(base_state)

    # Should fallback gracefully
    assert len(result["plan"]) >= 1
    assert len(result["search_queries"]) >= 1
