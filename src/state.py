"""LangGraph state schema for the AI Research Assistant."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict


class SearchResult(TypedDict):
    url: str
    title: str
    content: str
    score: float


class Source(TypedDict):
    url: str
    title: str
    content: str
    relevance_score: float


class Summary(TypedDict):
    url: str
    title: str
    summary: str


class Report(TypedDict):
    query: str
    executive_summary: str
    key_findings: list[str]
    sources: list[dict]
    metadata: dict


class NodeCost(TypedDict):
    node: str
    tokens_in: int
    tokens_out: int
    cost_usd: float


class ErrorEntry(TypedDict):
    node: str
    error: str
    attempt: int


class ResearchState(TypedDict):
    # --- Core flow ---
    query: str
    session_id: str

    # Planner output
    plan: list[str]
    search_queries: list[str]

    # Search & extraction
    search_results: Annotated[list[SearchResult], operator.add]
    extracted_sources: list[Source]

    # Summarizer (fan-out fan-in — results are appended across parallel nodes)
    summaries: Annotated[list[Summary], operator.add]

    # Final output
    final_report: Report | None

    # --- Control & meta ---
    human_feedback: str | None
    retry_counts: dict[str, int]
    error_log: Annotated[list[ErrorEntry], operator.add]
    node_costs: Annotated[list[NodeCost], operator.add]
    status: Literal["running", "awaiting_human", "complete", "failed"]
