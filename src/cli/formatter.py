"""CLI output formatter for displaying results."""

import json
from pathlib import Path
from typing import Any

import click

from src.models import AnalysisResult, CollectionResult, RankingResult


def display_analysis_start(sample_count: int):
    """Display analysis start message.
    
    Args:
        sample_count: Number of samples being analyzed
    """
    click.echo()
    click.echo(click.style("🔍 Analyzing Samples", fg="cyan", bold=True))
    click.echo(f"  Extracting themes from {sample_count} sample(s)...")


def display_analysis_result(result: AnalysisResult, verbose: bool):
    """Display analysis result.
    
    Args:
        result: Analysis result from LLM
        verbose: Whether to show detailed output
    """
    click.echo(click.style("  ✓ Analysis complete", fg="green"))
    
    if verbose:
        click.echo(f"  Query context: {result.query_context[:100]}...")
        click.echo(f"  Theme IDs: {len(result.theme_ids)} found")
        click.echo(f"  Processing time: {result.metadata.get('processing_time_ms', 0)}ms")


def display_collection_start(enable_online: bool, enable_crawl: bool, url_count: int):
    """Display collection start message.
    
    Args:
        enable_online: Whether online search is enabled
        enable_crawl: Whether URL crawling is enabled
        url_count: Number of URLs to crawl
    """
    click.echo()
    click.echo(click.style("🌐 Collecting Samples", fg="cyan", bold=True))
    
    methods = []
    if enable_online:
        methods.append("online search")
    if enable_crawl:
        methods.append(f"URL crawling ({url_count} URLs)")
    
    click.echo(f"  Using: {', '.join(methods)}")


def display_collection_result(result: CollectionResult, verbose: bool):
    """Display collection result.
    
    Args:
        result: Collection result from orchestrator
        verbose: Whether to show detailed output
    """
    click.echo(click.style("  ✓ Collection complete", fg="green"))
    click.echo(f"  Total samples: {result.metadata.get('total_samples', len(result.sample_ids))}")
    click.echo(f"  Duplicates removed: {result.metadata.get('duplicates_removed', 0)}")
    
    failed_url_count = result.metadata.get('failed_url_count', 0)
    if failed_url_count > 0:
        click.echo(
            click.style(
                f"  ⚠ Failed URLs: {failed_url_count}",
                fg="yellow",
            )
        )
    
    if verbose:
        collection_ids = result.metadata.get('collection_ids', [])
        click.echo(f"  Collections: {len(collection_ids)}")
        click.echo(f"  Processing time: {result.metadata.get('processing_time_ms', 0)}ms")


def display_ranking_start(candidate_count: int, threshold: float, top_n: int):
    """Display ranking start message.
    
    Args:
        candidate_count: Number of candidates to rank
        threshold: Similarity threshold
        top_n: Maximum results to return
    """
    click.echo()
    click.echo(click.style("📊 Ranking Candidates", fg="cyan", bold=True))
    click.echo(f"  Ranking {candidate_count} candidates...")
    click.echo(f"  Threshold: {threshold:.2f} | Top-N: {top_n}")


def display_ranking_result(result: RankingResult, verbose: bool):
    """Display ranking result with top results.
    
    Args:
        result: Ranking result from ranker
        verbose: Whether to show detailed output
    """
    click.echo(click.style("  ✓ Ranking complete", fg="green"))
    click.echo(f"  Ranked results: {result.returned_count}")
    
    if not result.results:
        click.echo(click.style("  No results met the similarity threshold", fg="yellow"))
        return
    
    # Display top results
    click.echo()
    click.echo(click.style("🏆 Top Results", fg="green", bold=True))
    click.echo()

    for ranked in result.results[:10]:  # Show top 10
        # Similarity bar
        bar_length = 20
        filled = int(ranked.similarity_details.max_score * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        click.echo(
            click.style(f"#{ranked.rank} ", fg="cyan", bold=True)
            + f"[{bar}] "
            + click.style(f"{ranked.score:.3f}", fg="green")
        )
        
        # Sample preview
        content_preview = ranked.sample.content[:100].replace("\n", " ")
        click.echo(f"  {content_preview}...")
        
        # Source info
        source = ranked.sample.source.value
        source_url = ranked.sample.metadata.get("source_url", "unknown")
        click.echo(
            click.style(f"  Source: ", fg="bright_black")
            + click.style(source, fg="blue")
            + click.style(f" | {source_url}", fg="bright_black")
        )
        
        if verbose:
            click.echo(
                click.style(
                    f"  Avg: {ranked.similarity_details.avg_score:.3f} | "
                    f"Max Score: {ranked.similarity_details.max_score} | "
                    f"Closest Ref ID: {ranked.similarity_details.closest_reference_id}",
                    fg="bright_black",
                )
            )
        
        click.echo()


def display_summary(
    reference_count: int,
    candidate_count: int,
    ranked_count: int,
    processing_time_ms: int,
):
    """Display final summary statistics.
    
    Args:
        reference_count: Number of reference samples
        candidate_count: Number of candidate samples collected
        ranked_count: Number of results after ranking
        processing_time_ms: Total processing time in milliseconds
    """
    click.echo(click.style("📈 Summary", fg="cyan", bold=True))
    click.echo(f"  Reference samples: {reference_count}")
    click.echo(f"  Candidates collected: {candidate_count}")
    click.echo(f"  Results ranked: {ranked_count}")
    click.echo(f"  Total time: {processing_time_ms / 1000:.2f}s")
    click.echo()


def export_results_to_json(
    ranking_result: RankingResult,
    analysis_result: AnalysisResult,
    collection_result: CollectionResult,
    export_path: Path,
):
    """Export results to JSON file.
    
    Args:
        ranking_result: Ranking result to export
        analysis_result: Analysis result to export
        collection_result: Collection result to export
        export_path: Path to export file
    """
    try:
        # Build export data
        export_data: dict[str, Any] = {
            "analysis": {
                "query_context": analysis_result.query_context,
                "theme_ids": analysis_result.theme_ids,
                "url_target_ids": analysis_result.url_target_ids,
                "metadata": analysis_result.metadata,
            },
            "collection": {
                "collection_id": collection_result.collection_id,
                "total_samples": len(collection_result.sample_ids),
                "duplicates_removed": collection_result.metadata.get("duplicates_removed", 0),
                "metadata": collection_result.metadata,
            },
            "ranking": {
                "total_candidates": ranking_result.total_candidates,
                "returned_count": ranking_result.returned_count,
                "threshold": ranking_result.threshold,
                "below_threshold_count": ranking_result.below_threshold_count,
                "processing_time_ms": ranking_result.processing_time_ms,
                "embeddings_cached": ranking_result.embeddings_cached,
                "results": [
                    {
                        "rank": r.rank,
                        "sample_id": r.sample.id,
                        "content": r.sample.content,
                        "source": r.sample.source.value,
                        "score": r.score,
                        "max_similarity": r.similarity_details.max_score if r.similarity_details else None,
                        "average_similarity": r.similarity_details.avg_score if r.similarity_details else None,
                        "closest_reference_id": r.similarity_details.closest_reference_id if r.similarity_details else None,
                        "metadata": r.sample.metadata,
                    }
                    for r in ranking_result.results
                ],
            },
        }
        
        # Write to file
        with export_path.open("w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        click.echo(
            click.style(f"✓ Results exported to {export_path}", fg="green")
        )
        
    except Exception as e:
        click.echo(
            click.style(f"✗ Export failed: {e}", fg="red"),
            err=True,
        )
