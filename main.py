#!/usr/bin/env python3
"""AI Ensemble Platform — CLI Interface.

Usage:
    python main.py "Your question here"
    python main.py --strategy debate "Your question"
    python main.py --interactive
    python main.py --status
"""

from __future__ import annotations
import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner

from core.models import SynthesisStrategy, ModelRole
from core.pipeline import EnsemblePipeline


console = Console()


def print_status(pipeline: EnsemblePipeline):
    status = pipeline.get_status()

    table = Table(title="AI Ensemble Platform Status")
    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="green")
    table.add_column("Role", style="yellow")

    for m in status["active_models"]:
        table.add_row(m["name"], m["provider"], m["role"])

    console.print(table)
    console.print(f"\nSynthesis strategy: [bold]{status['synthesis_strategy']}[/bold]")
    console.print(f"Active providers: [bold]{', '.join(status['active_providers'])}[/bold]")


def print_result(result):
    # Individual model responses
    console.print("\n")
    console.print(Panel("[bold]Individual Model Responses[/bold]", style="blue"))

    for scored in result.individual_responses:
        r = scored.response
        status = "[red]ERROR[/red]" if r.error else "[green]OK[/green]"
        score_str = f"Score: {scored.total_score:.2f}" if scored.total_score > 0 else ""

        header = f"{r.model_name} ({r.provider}) | {status} | {r.latency_ms:.0f}ms | {score_str}"
        content = r.error if r.error else r.content[:500] + ("..." if len(r.content) > 500 else "")

        if r.reasoning_trace:
            content = f"[dim]Thinking: {r.reasoning_trace[:200]}...[/dim]\n\n{content}"

        console.print(Panel(content, title=header, border_style="dim"))

    # Consensus & Disagreements
    if result.consensus_points:
        console.print(Panel(
            "\n".join(f"  [green]+[/green] {p}" for p in result.consensus_points),
            title="Consensus Points", border_style="green"
        ))
    if result.disagreement_points:
        console.print(Panel(
            "\n".join(f"  [red]![/red] {p}" for p in result.disagreement_points),
            title="Disagreement Points", border_style="red"
        ))

    # Final synthesized answer
    console.print(Panel(
        Markdown(result.final_answer),
        title=f"[bold]Synthesized Answer[/bold] | Confidence: {result.confidence:.0%} | Strategy: {result.strategy.value}",
        border_style="bold green",
        padding=(1, 2),
    ))

    console.print(
        f"[dim]Total latency: {result.total_latency_ms:.0f}ms | "
        f"Models: {len(result.individual_responses)} | "
        f"ID: {result.id}[/dim]"
    )


async def run_query(pipeline: EnsemblePipeline, query: str, strategy: SynthesisStrategy, roles: list[ModelRole] | None):
    with console.status("[bold green]Querying all models in parallel...", spinner="dots"):
        result = await pipeline.run(
            query=query,
            strategy=strategy,
            roles=roles,
        )
    print_result(result)
    return result


async def interactive_mode(pipeline: EnsemblePipeline, strategy: SynthesisStrategy):
    console.print(Panel(
        "[bold]AI Ensemble Platform[/bold]\n"
        "Type your question and press Enter. All configured models will reason on it.\n"
        "Commands: /status, /strategy <name>, /quit",
        border_style="bold blue",
    ))

    current_strategy = strategy

    while True:
        try:
            query = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query == "/quit":
            break
        if query == "/status":
            print_status(pipeline)
            continue
        if query.startswith("/strategy"):
            parts = query.split(maxsplit=1)
            if len(parts) > 1:
                try:
                    current_strategy = SynthesisStrategy(parts[1])
                    console.print(f"Strategy set to: [bold]{current_strategy.value}[/bold]")
                except ValueError:
                    console.print(f"[red]Invalid strategy. Options: {[s.value for s in SynthesisStrategy]}[/red]")
            else:
                console.print(f"Current: [bold]{current_strategy.value}[/bold]")
            continue

        await run_query(pipeline, query, current_strategy, None)

    console.print("[dim]Goodbye![/dim]")


def main():
    parser = argparse.ArgumentParser(description="AI Ensemble Reasoning Platform")
    parser.add_argument("query", nargs="?", help="Question to send to all models")
    parser.add_argument("--strategy", "-s", default=None,
                        choices=[s.value for s in SynthesisStrategy],
                        help="Synthesis strategy")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--status", action="store_true",
                        help="Show platform status")
    parser.add_argument("--config", "-c", default=None,
                        help="Path to config.yaml")
    parser.add_argument("--role", "-r", default=None,
                        choices=[r.value for r in ModelRole],
                        help="Only use models with this role")

    args = parser.parse_args()

    pipeline = EnsemblePipeline(args.config)

    if args.status:
        print_status(pipeline)
        return

    strategy = SynthesisStrategy(args.strategy) if args.strategy else SynthesisStrategy.WEIGHTED_MERGE

    if args.interactive:
        asyncio.run(interactive_mode(pipeline, strategy))
    elif args.query:
        roles = [ModelRole(args.role)] if args.role else None
        asyncio.run(run_query(pipeline, args.query, strategy, roles))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
