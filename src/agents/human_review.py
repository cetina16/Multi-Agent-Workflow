"""Human-in-the-loop review node.

This node is registered as an *interrupt* point. When the graph reaches it,
LangGraph pauses execution and checkpoints state. The CLI then prompts the user,
updates the state with `human_feedback`, and resumes.

The node itself is a pass-through — it just sets status to `awaiting_human`.
The actual pause/resume is handled by LangGraph's interrupt mechanism.
"""

from __future__ import annotations

from src.state import ResearchState


async def human_review_node(state: ResearchState) -> dict:
    """Signal that human review is pending.

    LangGraph will interrupt *before* this node runs (configured in graph.py).
    When resumed, human_feedback will already be set in state.
    """
    return {"status": "awaiting_human"}
