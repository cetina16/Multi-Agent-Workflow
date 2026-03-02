"""LangGraph checkpointing — uses AsyncSqliteSaver for free, file-based persistence.

This allows the graph to pause at interrupt nodes and resume in a new process
without any external database server.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


@asynccontextmanager
async def get_checkpointer(sqlite_path: str = "checkpoints.db") -> AsyncGenerator:
    """Async context manager that yields a LangGraph AsyncSqliteSaver checkpointer.

    Usage:
        async with get_checkpointer("checkpoints.db") as checkpointer:
            graph = build_graph(checkpointer=checkpointer)
            ...
    """
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(sqlite_path) as checkpointer:
            logger.info("AsyncSqliteSaver checkpointer initialized at '%s'.", sqlite_path)
            yield checkpointer
    except ImportError:
        logger.warning(
            "langgraph-checkpoint-sqlite not installed. "
            "Falling back to MemorySaver (state lost between processes)."
        )
        from langgraph.checkpoint.memory import MemorySaver
        yield MemorySaver()
    except Exception as exc:
        logger.warning("AsyncSqliteSaver unavailable (%s). Falling back to MemorySaver.", exc)
        from langgraph.checkpoint.memory import MemorySaver
        yield MemorySaver()
