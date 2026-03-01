"""Tests for the CostTracker."""

import os

import pytest

os.environ["COST_INPUT_PER_1K"] = "0.003"
os.environ["COST_OUTPUT_PER_1K"] = "0.015"

from src.cost_tracker import CostSummary, CostTracker, NodeCostResult


def test_compute_cost():
    tracker = CostTracker()
    result = tracker.record("planner", tokens_in=1000, tokens_out=500)
    assert result.node == "planner"
    assert result.tokens_in == 1000
    assert result.tokens_out == 500
    # 1000 * 0.003/1000 + 500 * 0.015/1000 = 0.003 + 0.0075 = 0.0105
    assert abs(result.cost_usd - 0.0105) < 1e-6


def test_zero_tokens():
    tracker = CostTracker()
    result = tracker.record("extractor", tokens_in=0, tokens_out=0)
    assert result.cost_usd == 0.0


def test_summarize():
    records = [
        {"node": "planner", "tokens_in": 500, "tokens_out": 200, "cost_usd": 0.0045},
        {"node": "searcher", "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0},
        {"node": "synthesizer", "tokens_in": 2000, "tokens_out": 800, "cost_usd": 0.018},
    ]
    summary = CostTracker.summarize(records)
    assert summary.total_tokens_in == 2500
    assert summary.total_tokens_out == 1000
    assert abs(summary.total_cost_usd - 0.0225) < 1e-6


def test_summary_as_dict():
    records = [
        {"node": "planner", "tokens_in": 100, "tokens_out": 50, "cost_usd": 0.001}
    ]
    summary = CostTracker.summarize(records)
    d = summary.as_dict()
    assert "per_node" in d
    assert "total_cost_usd" in d
    assert len(d["per_node"]) == 1
