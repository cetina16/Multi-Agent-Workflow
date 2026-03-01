"""High-level async database write helpers used by the storage agent."""

from __future__ import annotations

import uuid

from src.database.engine import get_session_factory
from src.database.repository import ResearchRepository
from src.state import NodeCost, Source, Summary


async def persist_research_results(
    session_id: str,
    extracted_sources: list[Source],
    summaries: list[Summary],
    final_report: dict,
    node_costs: list[NodeCost],
    status: str = "complete",
) -> None:
    """Write the full research output to PostgreSQL in a single transaction."""
    sid = uuid.UUID(session_id)
    factory = get_session_factory()

    # Build a summary lookup by URL for O(1) access
    summary_by_url = {s["url"]: s["summary"] for s in summaries}

    async with factory() as db:
        repo = ResearchRepository(db)

        # Update session with final report and status
        await repo.update_session_status(
            session_id=sid,
            status=status,
            final_report=final_report,
        )

        # Persist sources
        for src in extracted_sources:
            await repo.add_source(
                session_id=sid,
                url=src["url"],
                title=src.get("title"),
                relevance_score=src.get("relevance_score"),
                summary=summary_by_url.get(src["url"]),
            )

        # Persist per-node cost records
        for cost in node_costs:
            await repo.add_cost_record(
                session_id=sid,
                node_name=cost["node"],
                tokens_in=cost["tokens_in"],
                tokens_out=cost["tokens_out"],
                cost_usd=cost["cost_usd"],
            )

        await db.commit()
