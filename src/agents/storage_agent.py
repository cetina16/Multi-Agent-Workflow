"""Storage agent: persists the final report and all metadata to PostgreSQL."""

from __future__ import annotations

import logging

from src.state import ErrorEntry, ResearchState
from src.tools.db_writer import persist_research_results

logger = logging.getLogger(__name__)


async def storage_agent_node(state: ResearchState) -> dict:
    """Write the complete research output to PostgreSQL."""
    session_id = state.get("session_id", "")
    final_report = state.get("final_report")
    extracted_sources = state.get("extracted_sources", [])
    summaries = state.get("summaries", [])
    node_costs = state.get("node_costs", [])

    if not session_id:
        logger.warning("Storage agent: no session_id in state, skipping DB write.")
        return {"status": "complete"}

    if final_report is None:
        logger.warning("Storage agent: no final_report to persist.")
        return {"status": "complete"}

    try:
        await persist_research_results(
            session_id=session_id,
            extracted_sources=extracted_sources,
            summaries=summaries,
            final_report=dict(final_report),
            node_costs=node_costs,
            status="complete",
        )
        logger.info("Storage agent: results persisted for session %s", session_id)
    except Exception as exc:
        logger.error("Storage agent: DB write failed: %s", exc)
        return {
            "status": "complete",  # Don't fail the pipeline on storage error
            "error_log": [
                ErrorEntry(node="storage_agent", error=str(exc), attempt=1)
            ],
        }

    return {"status": "complete"}
