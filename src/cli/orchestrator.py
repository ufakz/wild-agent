"""CLI orchestration logic for the main pipeline."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import structlog

from src.analyzers.llm_analyzer import LLMAnalyzer
from src.collectors.orchestrator import SampleCollector
from src.models import Sample, SampleSource
from src.rankers.similarity_ranker import SimilarityRanker
from src.cli.formatter import (
    display_analysis_start,
    display_analysis_result,
    display_collection_start,
    display_collection_result,
    display_ranking_start,
    display_ranking_result,
    export_results_to_json,
    display_summary,
)

logger = structlog.get_logger()


def run_collection(
    samples: list[str],
    urls: Optional[list[str]],
    top_n: int,
    threshold: float,
    enable_crawl: bool,
    enable_online: bool,
    export_path: Optional[Path],
    verbose: bool,
):
    """Run the complete collection and ranking pipeline.
    
    Args:
        samples: List of reference sample texts
        urls: Optional list of URLs to crawl
        top_n: Maximum number of results to return
        threshold: Minimum similarity threshold
        enable_crawl: Whether to enable URL crawling
        enable_online: Whether to enable online search
        export_path: Optional path to export JSON results
        verbose: Whether to enable verbose output
    """
    # Configure logging
    if verbose:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        )
    
    # Run async pipeline
    asyncio.run(
        _run_pipeline_async(
            samples=samples,
            urls=urls,
            top_n=top_n,
            threshold=threshold,
            enable_crawl=enable_crawl,
            enable_online=enable_online,
            export_path=export_path,
            verbose=verbose,
        )
    )


async def _run_pipeline_async(
    samples: list[str],
    urls: Optional[list[str]],
    top_n: int,
    threshold: float,
    enable_crawl: bool,
    enable_online: bool,
    export_path: Optional[Path],
    verbose: bool,
):
    """Async implementation of the pipeline.
    
    Pipeline stages:
    1. Create Sample objects from input text
    2. Analyze samples to extract themes (LLM Analyzer)
    3. Collect similar samples (Sample Collector)
    4. Rank candidates by similarity (Similarity Ranker)
    5. Display/export results
    """
    try:
        # Stage 1: Create Sample objects
        logger.info("pipeline_start", sample_count=len(samples))
        
        reference_samples = [
            Sample(
                content=text,
                source=SampleSource.USER_INPUT,
                metadata={"index": i},
            )
            for i, text in enumerate(samples)
        ]
        
        # Stage 2: Analyze samples for themes
        display_analysis_start(len(reference_samples))
        
        analyzer = LLMAnalyzer()
        try:
            analysis_result = await analyzer.analyze(
                samples=reference_samples,
                max_themes=5,
                min_confidence=0.8,
            )
            display_analysis_result(analysis_result, verbose)
        except Exception as e:
            logger.error("analysis_failed", error=str(e))
            raise
        
        # Stage 3: Collect similar samples
        query_context = analysis_result.query_context
        display_collection_start(enable_online, enable_crawl, len(urls) if urls else 0)
        
        collector = SampleCollector()
        try:
            collection_result, candidate_samples = await collector.collect(
                query_context=query_context,
                urls=urls,
                enable_online=enable_online,
                enable_crawler=enable_crawl,
                max_samples_per_source=100,
            )
            display_collection_result(collection_result, verbose)
        except Exception as e:
            logger.error("collection_failed", error=str(e))
            raise
        
        # Check if we have collected any candidates
        if not candidate_samples:
            logger.warning("no_candidates_collected")
            display_summary(
                reference_count=len(reference_samples),
                candidate_count=0,
                ranked_count=0,
                processing_time_ms=0,
            )
            return
        
        # Stage 4: Rank candidates by similarity
        display_ranking_start(len(candidate_samples), threshold, top_n)
        
        ranker = SimilarityRanker()
        ranking_result = await ranker.rank(
            reference_samples=reference_samples,
            candidate_samples=candidate_samples,
            threshold=threshold,
            top_n=top_n,
        )
        
        display_ranking_result(ranking_result, verbose)
        
        # Stage 5: Export if requested
        if export_path:
            export_results_to_json(
                ranking_result=ranking_result,
                analysis_result=analysis_result,
                collection_result=collection_result,
                export_path=export_path,
            )
        
        # Display final summary
        total_time = (
            analysis_result.metadata.get("processing_time_ms", 0)
            + collection_result.metadata.get("processing_time_ms", 0)
            + ranking_result.processing_time_ms
        )
        
        display_summary(
            reference_count=len(reference_samples),
            candidate_count=collection_result.metadata.get("total_samples", 0),
            ranked_count=len(ranking_result.results),
            processing_time_ms=total_time,
        )
        
        logger.info("pipeline_complete")
        
    except Exception as e:
        logger.error("pipeline_failed", error=str(e))
        raise
