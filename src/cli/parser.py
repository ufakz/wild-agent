"""CLI argument parser using click."""

from pathlib import Path
from typing import Optional

import click
import structlog

logger = structlog.get_logger()


@click.group()
@click.version_option(version="0.1.0", prog_name="wild-agent")
def cli():
    """Wild Agent - Sample collection and similarity ranking using LLMs.
    
    Analyze text samples, collect similar content from the web,
    and rank results by semantic similarity.
    """
    pass


@cli.command()
@click.option(
    "--sample",
    "-s",
    "samples",
    multiple=True,
    help="Text sample to analyze (can be used multiple times)",
)
@click.option(
    "--sample-file",
    "-f",
    type=click.Path(exists=True, path_type=Path),
    help="File containing samples (one per line)",
)
@click.option(
    "--url",
    "-u",
    "urls",
    multiple=True,
    help="URL to crawl for samples (can be used multiple times)",
)
@click.option(
    "--urls-file",
    type=click.Path(exists=True, path_type=Path),
    help="File containing URLs (one per line)",
)
@click.option(
    "--top-n",
    "-n",
    type=int,
    default=100,
    help="Maximum number of results to return (default: 10)",
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=0.3,
    help="Minimum similarity threshold 0.0-1.0 (default: 0.3)",
)
@click.option(
    "--no-crawl",
    is_flag=True,
    help="Disable URL crawling (only use online search)",
)
@click.option(
    "--no-online-search",
    is_flag=True,
    help="Disable online search (only crawl provided URLs)",
)
@click.option(
    "--export",
    "-e",
    type=click.Path(path_type=Path),
    help="Export results to JSON file",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def collect(
    samples: tuple[str, ...],
    sample_file: Optional[Path],
    urls: tuple[str, ...],
    urls_file: Optional[Path],
    top_n: int,
    threshold: float,
    no_crawl: bool,
    no_online_search: bool,
    export: Optional[Path],
    verbose: bool,
):
    """Collect and rank similar samples based on reference text.
    
    Provide one or more text samples as references. The tool will:
    1. Analyze samples to extract common themes
    2. Collect similar content from online search and/or URL crawling
    3. Rank collected samples by similarity to your references
    4. Return the top-N most similar results
    
    Examples:
    
        # Basic usage with one sample
        wild-agent collect --sample "Your reference text here"
        
        # Multiple samples with URLs to crawl
        wild-agent collect -s "Sample 1" -s "Sample 2" -u https://example.com
        
        # Load samples from file, export to JSON
        wild-agent collect -f samples.txt -e results.json
        
        # Only crawl URLs (no online search)
        wild-agent collect -f samples.txt -u https://example.com --no-online-search
        
        # Adjust similarity threshold and result count
        wild-agent collect -s "Sample" --threshold 0.8 --top-n 5
    """
    from src.cli.orchestrator import run_collection
    
    # Validate inputs
    validation_errors = validate_collect_args(
        samples=samples,
        sample_file=sample_file,
        urls=urls,
        urls_file=urls_file,
        top_n=top_n,
        threshold=threshold,
        no_crawl=no_crawl,
        no_online_search=no_online_search,
    )
    
    if validation_errors:
        for error in validation_errors:
            click.echo(click.style(f"Error: {error}", fg="red"), err=True)
        raise click.Abort()
    
    # Load samples from file if provided
    all_samples = list(samples)
    if sample_file:
        try:
            file_samples = load_samples_from_file(sample_file)
            all_samples.extend(file_samples)
            if verbose:
                click.echo(f"Loaded {len(file_samples)} samples from {sample_file}")
        except Exception as e:
            click.echo(click.style(f"Error loading samples: {e}", fg="red"), err=True)
            raise click.Abort()
    
    # Load URLs from file if provided
    all_urls = list(urls)
    if urls_file:
        try:
            file_urls = load_urls_from_file(urls_file)
            all_urls.extend(file_urls)
            if verbose:
                click.echo(f"Loaded {len(file_urls)} URLs from {urls_file}")
        except Exception as e:
            click.echo(click.style(f"Error loading URLs: {e}", fg="red"), err=True)
            raise click.Abort()
    
    # Run the collection pipeline
    try:
        run_collection(
            samples=all_samples,
            urls=all_urls if all_urls else None,
            top_n=top_n,
            threshold=threshold,
            enable_crawl=not no_crawl,
            enable_online=not no_online_search,
            export_path=export,
            verbose=verbose,
        )
    except KeyboardInterrupt:
        click.echo("\n" + click.style("Collection interrupted by user", fg="yellow"))
        raise click.Abort()
    except Exception as e:
        click.echo(click.style(f"Collection failed: {e}", fg="red"), err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        raise click.Abort()


def validate_collect_args(
    samples: tuple[str, ...],
    sample_file: Optional[Path],
    urls: tuple[str, ...],
    urls_file: Optional[Path],
    top_n: int,
    threshold: float,
    no_crawl: bool,
    no_online_search: bool,
) -> list[str]:
    """Validate collect command arguments.
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Must provide at least one sample
    if not samples and not sample_file:
        errors.append("Must provide at least one sample (--sample or --sample-file)")
    
    # Can't disable both collection methods
    if no_crawl and no_online_search:
        errors.append("Cannot disable both crawling and online search")
    
    # If no URLs and crawl is enabled, we need online search
    if not urls and not urls_file and not no_crawl and no_online_search:
        errors.append("No URLs provided and online search is disabled")
    
    # Validate top_n
    if top_n < 1:
        errors.append(f"--top-n must be >= 1, got {top_n}")
    if top_n > 100:
        errors.append(f"--top-n must be <= 100, got {top_n}")
    
    # Validate threshold
    if not 0.0 <= threshold <= 1.0:
        errors.append(f"--threshold must be between 0.0 and 1.0, got {threshold}")
    
    return errors


def load_samples_from_file(path: Path) -> list[str]:
    """Load samples from file (one per line).
    
    Args:
        path: Path to samples file
        
    Returns:
        List of sample texts
    """
    samples = []
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):  
                samples.append(line)
    
    return samples


def load_urls_from_file(path: Path) -> list[str]:
    """Load URLs from file (one per line).
    
    Args:
        path: Path to URLs file
        
    Returns:
        List of URLs
    """
    urls = []
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if line.startswith(("http://", "https://")):
                    urls.append(line)
                else:
                    logger.warning("invalid_url_skipped", url=line)
    
    return urls


if __name__ == "__main__":
    cli()
