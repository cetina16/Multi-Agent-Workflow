from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import CostRecord, ResearchSession, Source


class ResearchRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_session(self, query: str, session_id: str | None = None) -> ResearchSession:
        record = ResearchSession(query=query, status="running")
        if session_id:
            record.id = session_id
        self.session.add(record)
        await self.session.flush()
        return record

    async def get_session(self, session_id: str) -> ResearchSession | None:
        result = await self.session.execute(
            select(ResearchSession)
            .options(selectinload(ResearchSession.sources))
            .options(selectinload(ResearchSession.cost_records))
            .where(ResearchSession.id == session_id)
        )
        return result.scalar_one_or_none()

    async def update_session_status(
        self,
        session_id: str,
        status: str,
        final_report: dict | None = None,
    ) -> None:
        values: dict = {
            "status": status,
            "updated_at": datetime.now(timezone.utc),
        }
        if final_report is not None:
            values["final_report"] = final_report
        await self.session.execute(
            update(ResearchSession)
            .where(ResearchSession.id == session_id)
            .values(**values)
        )

    async def add_source(
        self,
        session_id: str,
        url: str,
        title: str | None = None,
        relevance_score: float | None = None,
        summary: str | None = None,
    ) -> Source:
        source = Source(
            session_id=session_id,
            url=url,
            title=title,
            relevance_score=relevance_score,
            summary=summary,
        )
        self.session.add(source)
        await self.session.flush()
        return source

    async def add_cost_record(
        self,
        session_id: str,
        node_name: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
    ) -> CostRecord:
        record = CostRecord(
            session_id=session_id,
            node_name=node_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def get_cost_summary(self, session_id: str) -> list[CostRecord]:
        result = await self.session.execute(
            select(CostRecord).where(CostRecord.session_id == session_id)
        )
        return list(result.scalars().all())
