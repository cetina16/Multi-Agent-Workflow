"""CLI entry point for the Multi-Agent Research Workflow."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
import warnings

warnings.filterwarnings("ignore", category=ResourceWarning)

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import get_settings
from src.cost_tracker import CostTracker
from src.database.engine import init_db
from src.graph import build_graph
from src.memory import get_checkpointer

console = Console()
logging.basicConfig(level=logging.WARNING)


def _display_report(report: dict) -> None:
    """Pretty-print the final research report."""
    console.print()
    console.print(Panel(
        f"[bold cyan]{report.get('query', '')}[/bold cyan]",
        title="[bold green]Research Complete[/bold green]",
        border_style="green",
    ))

    console.print(f"\n[bold]Executive Summary[/bold]\n{report.get('executive_summary', '')}")

    findings = report.get("key_findings", [])
    if findings:
        console.print("\n[bold]Key Findings[/bold]")
        for i, f in enumerate(findings, 1):
            console.print(f"  {i}. {f}")

    sources = report.get("sources", [])
    if sources:
        table = Table(title="Sources", show_lines=True)
        table.add_column("Title", style="cyan", max_width=40)
        table.add_column("URL", style="dim", max_width=50)
        table.add_column("Contribution", max_width=60)
        for s in sources:
            table.add_row(
                s.get("title", ""),
                s.get("url", ""),
                s.get("key_contribution", ""),
            )
        console.print()
        console.print(table)

    meta = report.get("metadata", {})
    if meta:
        console.print(
            f"\n[dim]Confidence: {meta.get('confidence', 'N/A')} | "
            f"Sources: {meta.get('num_sources', 0)} | "
            f"Gaps: {meta.get('gaps', 'None noted')}[/dim]"
        )


def _display_costs(node_costs: list[dict]) -> None:
    summary = CostTracker.summarize(node_costs)
    table = Table(title="Cost Report", show_lines=True)
    table.add_column("Node", style="cyan")
    table.add_column("Tokens In", justify="right")
    table.add_column("Tokens Out", justify="right")
    table.add_column("Cost (USD)", justify="right", style="yellow")
    for r in summary.records:
        table.add_row(r.node, str(r.tokens_in), str(r.tokens_out), f"${r.cost_usd:.6f}")
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{summary.total_tokens_in}[/bold]",
        f"[bold]{summary.total_tokens_out}[/bold]",
        f"[bold yellow]${summary.total_cost_usd:.6f}[/bold yellow]",
    )
    console.print()
    console.print(table)


async def _run_research(query: str, session_id: str, auto_approve: bool) -> None:
    settings = get_settings()
    await init_db()  # Create SQLite tables if they don't exist yet

    async with get_checkpointer(settings.sqlite_path) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": session_id}}

        initial_state = {
            "query": query,
            "session_id": session_id,
            "plan": [],
            "search_queries": [],
            "search_results": [],
            "extracted_sources": [],
            "summaries": [],
            "final_report": None,
            "human_feedback": None,
            "retry_counts": {},
            "error_log": [],
            "node_costs": [],
            "status": "running",
        }

        console.print(f"\n[bold]Starting research:[/bold] {query}")
        console.print(f"[dim]Session ID: {session_id}[/dim]\n")

        # Run until the first interrupt (human_review)
        async for event in graph.astream(initial_state, config, stream_mode="updates"):
            for node_name in event:
                console.print(f"  [dim]→ {node_name}[/dim]")

        # Check if graph is waiting at human_review — use async aget_state
        current = await graph.aget_state(config)
        if current.next and "human_review" in current.next:
            state_vals = current.values
            plan = state_vals.get("plan", [])
            queries = state_vals.get("search_queries", [])

            console.print(Panel(
                "[bold yellow]Human Review Required[/bold yellow]\n\n"
                "[bold]Plan:[/bold]\n" + "\n".join(f"  • {p}" for p in plan) + "\n\n"
                "[bold]Search queries:[/bold]\n" + "\n".join(f"  • {q}" for q in queries),
                border_style="yellow",
            ))

            if auto_approve:
                feedback = "approved"
                console.print("[dim]Auto-approved.[/dim]")
            else:
                feedback = click.prompt(
                    "\nEnter feedback or type 'approved' to continue",
                    default="approved",
                )

            # Inject feedback and resume — use async aupdate_state
            await graph.aupdate_state(config, {"human_feedback": feedback})
            async for event in graph.astream(None, config, stream_mode="updates"):
                for node_name in event:
                    console.print(f"  [dim]→ {node_name}[/dim]")

        # Fetch final state — use async aget_state
        final = await graph.aget_state(config)
        final_state = final.values
        report = final_state.get("final_report")
        node_costs = final_state.get("node_costs", [])
        errors = final_state.get("error_log", [])

        if report:
            _display_report(dict(report))

        if node_costs:
            _display_costs(node_costs)

        if errors:
            console.print("\n[bold red]Errors encountered:[/bold red]")
            for e in errors:
                console.print(f"  [red]• [{e['node']}] {e['error']}[/red]")


@click.group()
def cli() -> None:
    """Multi-Agent AI Research Assistant powered by LangGraph + Groq (free)."""


@cli.command()
@click.argument("query")
@click.option("--session-id", default=None, help="Reuse an existing session ID.")
@click.option("--auto-approve", is_flag=True, default=False, help="Skip human review prompt.")
@click.option("--output-json", default=None, help="Save final report to a JSON file.")
def research(query: str, session_id: str | None, auto_approve: bool, output_json: str | None) -> None:
    """Run a research query through the multi-agent pipeline."""
    sid = session_id or str(uuid.uuid4())

    async def _main():
        await _run_research(query, sid, auto_approve)

        if output_json:
            settings = get_settings()
            async with get_checkpointer(settings.sqlite_path) as checkpointer:
                graph = build_graph(checkpointer=checkpointer)
                config = {"configurable": {"thread_id": sid}}
                state_snapshot = await graph.aget_state(config)
                report = state_snapshot.values.get("final_report")
                if report:
                    with open(output_json, "w") as f:
                        json.dump(dict(report), f, indent=2)
                    console.print(f"\n[green]Report saved to {output_json}[/green]")

    asyncio.run(_main())


@cli.command()
@click.option("--session-id", required=True, help="Session ID to resume.")
@click.option("--feedback", default="approved", help="Human feedback to inject.")
def resume(session_id: str, feedback: str) -> None:
    """Resume a paused session (after human-in-the-loop interrupt)."""
    settings = get_settings()

    async def _main():
        async with get_checkpointer(settings.sqlite_path) as checkpointer:
            graph = build_graph(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": session_id}}

            await graph.aupdate_state(config, {"human_feedback": feedback})
            console.print(f"[dim]Resuming session {session_id} with feedback: {feedback}[/dim]")

            async for event in graph.astream(None, config, stream_mode="updates"):
                for node_name in event:
                    console.print(f"  [dim]→ {node_name}[/dim]")

            final = await graph.aget_state(config)
            report = final.values.get("final_report")
            node_costs = final.values.get("node_costs", [])

            if report:
                _display_report(dict(report))
            if node_costs:
                _display_costs(node_costs)

    asyncio.run(_main())


@cli.command("cost-report")
@click.option("--session-id", required=True, help="Session ID to show costs for.")
def cost_report(session_id: str) -> None:
    """Display the cost breakdown for a completed session."""
    settings = get_settings()

    async def _main():
        async with get_checkpointer(settings.sqlite_path) as checkpointer:
            graph = build_graph(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": session_id}}
            state_snapshot = await graph.aget_state(config)

            if not state_snapshot or not state_snapshot.values:
                console.print(f"[red]No state found for session {session_id}[/red]")
                return

            node_costs = state_snapshot.values.get("node_costs", [])
            if not node_costs:
                console.print("[yellow]No cost data recorded for this session.[/yellow]")
                return

            _display_costs(node_costs)

    asyncio.run(_main())


if __name__ == "__main__":
    cli()
