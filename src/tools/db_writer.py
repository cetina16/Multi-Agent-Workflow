"""High-level async database write helpers used by the storage agent."""

from __future__ import annotations

from src.database.engine import get_session_factory
from src.database.models import ResearchSession
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
    """Write the full research output to SQLite in a single transaction."""
    factory = get_session_factory()

    # Build a summary lookup by URL for O(1) access
    summary_by_url = {s["url"]: s["summary"] for s in summaries}

    async with factory() as db:
        repo = ResearchRepository(db)

        # Upsert session: create if missing, update if already exists
        existing = await repo.get_session(session_id)
        if existing is None:
            record = ResearchSession(
                id=session_id,
                query=final_report.get("query", ""),
                status=status,
                final_report=final_report,
            )
            db.add(record)
            await db.flush()
        else:
            await repo.update_session_status(
                session_id=session_id,
                status=status,
                final_report=final_report,
            )

        # Persist sources
        for src in extracted_sources:
            await repo.add_source(
                session_id=session_id,
                url=src["url"],
                title=src.get("title"),
                relevance_score=src.get("relevance_score"),
                summary=summary_by_url.get(src["url"]),
            )

        # Persist per-node cost records
        for cost in node_costs:
            await repo.add_cost_record(
                session_id=session_id,
                node_name=cost["node"],
                tokens_in=cost["tokens_in"],
                tokens_out=cost["tokens_out"],
                cost_usd=cost["cost_usd"],
            )

        await db.commit()
