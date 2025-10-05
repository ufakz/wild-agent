"""Sample collector orchestrator."""

import asyncio
import time
from typing import Optional

import structlog

from src.collectors.crawler import CrawlerCollector
from src.collectors.online import OnlineCollector
from src.lib.deduplicator import Deduplicator
from src.models import Collection, CollectionMethod, CollectionResult, Sample, URLTarget

logger = structlog.get_logger()


class SampleCollector:
    """Orchestrates sample collection from multiple sources."""

    def __init__(
        self,
        online_collector: Optional[OnlineCollector] = None,
        crawler_collector: Optional[CrawlerCollector] = None,
        deduplicator: Optional[Deduplicator] = None,
    ):
        """Initialize sample collector.
        
        Args:
            online_collector: Collector for online search (default: new instance)
            crawler_collector: Collector for URL crawling (default: new instance)
            deduplicator: Deduplicator for removing duplicates (default: new instance)
        """
        self.online_collector = online_collector or OnlineCollector()
        self.crawler_collector = crawler_collector or CrawlerCollector()
        self.deduplicator = deduplicator or Deduplicator()

    async def collect(
        self,
        query_context: str,
        urls: Optional[list[str]] = None,
        enable_online: bool = True,
        enable_crawler: bool = True,
        max_samples_per_source: int = 10,
    ) -> tuple[CollectionResult, list[Sample]]:
        """Collect samples from multiple sources.
        
        Args:
            query_context: Search context/themes
            urls: Optional list of URLs to crawl
            enable_online: Whether to use online search
            enable_crawler: Whether to crawl URLs
            max_samples_per_source: Maximum samples per collection method
            
        Returns:
            Tuple of (CollectionResult with metadata, list of collected Sample objects)
            
        Raises:
            ValueError: If both online and crawler are disabled
        """
        start_time = time.time()
        
        if not enable_online and not enable_crawler:
            raise ValueError("At least one collection method must be enabled")
        
        if not query_context or len(query_context) < 10:
            raise ValueError("query_context must be at least 10 characters")
        
        logger.info(
            "collection_start",
            query_length=len(query_context),
            urls_count=len(urls) if urls else 0,
            enable_online=enable_online,
            enable_crawler=enable_crawler,
        )
        
        # Collect from both sources in parallel
        collections: list[Collection] = []
        all_samples: list[Sample] = []
        url_targets: list[URLTarget] = []
        
        tasks = []
        
        # Online search task
        if enable_online:
            tasks.append(
                self._collect_online(query_context, max_samples_per_source)
            )
        
        # URL crawling task
        if enable_crawler and urls:
            tasks.append(
                self._collect_from_urls(urls, query_context)
            )
        
        # Run tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error("collection_failed", error=str(result))
                continue
            
            if isinstance(result, tuple):
                if len(result) == 3:
                    # Crawler returns (Collection, list[Sample], list[URLTarget])
                    collection, samples, targets = result
                    collections.append(collection)
                    all_samples.extend(samples)
                    url_targets.extend(targets)
                elif len(result) == 2:
                    # Online returns (Collection, list[Sample])
                    collection, samples = result
                    collections.append(collection)
                    all_samples.extend(samples)
        
        logger.info(
            "collections_complete",
            collection_count=len(collections),
            total_samples_before_dedup=len(all_samples),
        )
        
        # Deduplicate samples
        samples_before_dedup = len(all_samples)
        if all_samples:
            all_samples = await self.deduplicator.deduplicate(all_samples)
            logger.info(
                "deduplication_complete",
                final_sample_count=len(all_samples),
            )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Count failed URLs
        failed_urls = [t for t in url_targets if t.status.value in ["FAILED", "TIMEOUT"]]
        
        # Create a master Collection that aggregates all samples
        # Use ONLINE_SEARCH if only online, URL_CRAWLING if only crawler, or ONLINE_SEARCH as default for mixed
        if len(collections) == 1:
            method = collections[0].method
        elif enable_online:
            method = CollectionMethod.ONLINE_SEARCH
        else:
            method = CollectionMethod.URL_CRAWLING
        
        master_collection = Collection(
            method=method,
            query_context=query_context,
            sample_ids=[s.id for s in all_samples],
            url_target_ids=[t.id for t in url_targets],
            metadata={
                "sub_collection_ids": [c.id for c in collections],
                "sub_collections_count": len(collections),
                "processing_time_ms": processing_time,
                "online_enabled": enable_online,
                "crawler_enabled": enable_crawler,
            },
        )
        
        # Create CollectionResult with correct fields
        result = CollectionResult(
            collection_id=master_collection.id,
            sample_ids=[s.id for s in all_samples],
            metadata={
                "processing_time_ms": processing_time,
                "total_samples": len(all_samples),
                "samples_before_dedup": samples_before_dedup,
                "duplicates_removed": samples_before_dedup - len(all_samples),
                "failed_url_count": len(failed_urls),
                "collections": len(collections),
                "collection_ids": [c.id for c in collections],
                "online_enabled": enable_online,
                "crawler_enabled": enable_crawler,
                "urls_crawled": len(urls) if urls else 0,
            },
        )
        
        logger.info(
            "collection_result",
            total_samples=len(all_samples),
            duplicates_removed=samples_before_dedup - len(all_samples),
            failed_urls=len(failed_urls),
            processing_time_ms=processing_time,
        )
        
        return result, all_samples

    async def _collect_online(
        self,
        query_context: str,
        max_samples: int,
    ) -> tuple[Collection, list[Sample]]:
        """Collect samples from online search.
        
        Args:
            query_context: Search context
            max_samples: Maximum samples to collect
            
        Returns:
            Tuple of (Collection, list of Sample objects)
        """
        try:
            return await self.online_collector.collect(
                query_context=query_context,
                max_samples=max_samples,
            )
        except Exception as e:
            logger.error("online_collection_failed", error=str(e))
            raise

    async def _collect_from_urls(
        self,
        urls: list[str],
        query_context: str,
    ) -> tuple[Collection, list[Sample], list[URLTarget]]:
        """Collect samples from URL crawling.
        
        Args:
            urls: List of URLs to crawl
            query_context: Context for extraction
            
        Returns:
            Tuple of (Collection, list of Sample objects, list of URLTargets)
        """
        try:
            return await self.crawler_collector.crawl(
                urls=urls,
                query_context=query_context,
            )
        except Exception as e:
            logger.error("crawler_collection_failed", error=str(e))
            raise