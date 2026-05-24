import argparse
import asyncio
import csv
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.agents.state import derive_collection_mode
from src.config.loader import default_config_path, load_config
from src.config.models import WildConfig

load_dotenv()
console = Console()

CSV_FIELDS = [
    "content", "url", "title", "similarity_score", "relevance_score",
    "matched_reference_index", "themes", "scraped_at",
]


def export_to_csv(samples: list[dict], output_path: Path) -> None:
    if not samples:
        console.print("[yellow]No samples to export.[/yellow]")
        return
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                "content": sample["content"][:1000],
                "url": sample["url"],
                "title": sample.get("title") or "",
                "similarity_score": (
                    f"{sample['similarity_score']:.4f}"
                    if sample.get("similarity_score") is not None else ""
                ),
                "relevance_score": sample.get("relevance_score") or "",
                "matched_reference_index": sample.get("matched_reference_index") or "",
                "themes": "|".join(sample.get("themes", [])),
                "scraped_at": sample["scraped_at"],
            })
    console.print(f"[green]Exported {len(samples)} samples to {output_path}[/green]")


def _summary_text(config: WildConfig) -> str:
    coll = config.collection
    mode = derive_collection_mode(coll.theme, coll.examples)
    lines = [f"Config: {config.config_path}"]
    if coll.theme:
        lines.append(f"Theme: {coll.theme}")
    if coll.examples:
        lines.append(f"Examples: {len(coll.examples)}")
    lines.append(f"Target: {coll.target_count} | Mode: {mode}")
    return "\n".join(lines)


async def main_async(config: WildConfig) -> int:
    from src.agents.graph import run_collection

    console.print(Panel(_summary_text(config), title="Wild Agent", border_style="green"))

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        progress.add_task("Collecting...", total=None)
        try:
            final_state = await run_collection(config)
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            return 1

    accepted = len(final_state.get("samples", []))
    rejected = final_state.get("rejected_count", 0)
    cycles = final_state.get("iteration", 0)
    max_cycles = final_state.get("max_iterations", 0)
    console.print(
        f"Accepted: [green]{accepted}[/green] | Rejected: [red]{rejected}[/red] "
        f"| Cycles: {cycles}/{max_cycles}"
    )
    if config.collection.output and accepted:
        export_to_csv(final_state["samples"], Path(config.collection.output))
    elif config.collection.output:
        console.print("[yellow]No samples to export.[/yellow]")
    if err := final_state.get("error"):
        console.print(f"[red]Last error: {err}[/red]")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Wild Agent — config-driven data collection")
    parser.add_argument("config", nargs="?", help="YAML config (default: config/default.yaml)")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path()
    if not config_path.exists():
        parser.error(f"Config not found: {config_path}")

    try:
        config = load_config(config_path)
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]Config error: {e}[/red]")
        return 1

    return asyncio.run(main_async(config))


if __name__ == "__main__":
    sys.exit(main())
