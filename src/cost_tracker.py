"""Token usage and cost tracking helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage

from config import get_settings


@dataclass
class NodeCostResult:
    node: str
    tokens_in: int
    tokens_out: int
    cost_usd: float

    def as_dict(self) -> dict:
        return {
            "node": self.node,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
        }


@dataclass
class CostSummary:
    records: list[NodeCostResult] = field(default_factory=list)

    @property
    def total_tokens_in(self) -> int:
        return sum(r.tokens_in for r in self.records)

    @property
    def total_tokens_out(self) -> int:
        return sum(r.tokens_out for r in self.records)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

    def as_dict(self) -> dict:
        return {
            "per_node": [r.as_dict() for r in self.records],
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_cost_usd": round(self.total_cost_usd, 6),
        }


class CostTracker:
    """Tracks LLM usage costs per graph node."""

    def __init__(self) -> None:
        settings = get_settings()
        self._cost_in = settings.cost_input_per_1k / 1000
        self._cost_out = settings.cost_output_per_1k / 1000

    def record_from_message(self, node: str, message: AIMessage) -> NodeCostResult:
        """Extract usage from an AIMessage and compute cost."""
        usage = message.usage_metadata or {}
        tokens_in = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)
        return self._compute(node, tokens_in, tokens_out)

    def record(self, node: str, tokens_in: int, tokens_out: int) -> NodeCostResult:
        return self._compute(node, tokens_in, tokens_out)

    def _compute(self, node: str, tokens_in: int, tokens_out: int) -> NodeCostResult:
        cost = (tokens_in * self._cost_in) + (tokens_out * self._cost_out)
        return NodeCostResult(
            node=node,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=round(cost, 6),
        )

    @staticmethod
    def summarize(node_costs: list[dict]) -> CostSummary:
        records = [
            NodeCostResult(
                node=c["node"],
                tokens_in=c["tokens_in"],
                tokens_out=c["tokens_out"],
                cost_usd=c["cost_usd"],
            )
            for c in node_costs
        ]
        return CostSummary(records=records)
