"""LangGraph checkpointing setup for human-in-the-loop support.

Uses PostgresSaver (sync psycopg2-based) for durable state storage.
This allows the graph to pause at interrupt nodes and resume in a new process.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


@contextmanager
def get_checkpointer(sync_database_url: str) -> Generator:
    """Context manager that yields a LangGraph PostgresSaver checkpointer.

    Usage:
        with get_checkpointer(settings.sync_database_url) as checkpointer:
            graph = build_graph(checkpointer=checkpointer)
            ...
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        with PostgresSaver.from_conn_string(sync_database_url) as checkpointer:
            checkpointer.setup()  # Creates langgraph checkpoint tables if absent
            logger.info("PostgresSaver checkpointer initialized.")
            yield checkpointer
    except ImportError:
        logger.warning(
            "langgraph-checkpoint-postgres not installed. "
            "Falling back to MemorySaver (no persistence across processes)."
        )
        from langgraph.checkpoint.memory import MemorySaver

        yield MemorySaver()
    except Exception as exc:
        logger.warning(
            "PostgresSaver unavailable (%s). Falling back to MemorySaver.", exc
        )
        from langgraph.checkpoint.memory import MemorySaver

        yield MemorySaver()
