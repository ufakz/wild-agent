"""Web crawler collector using crawl4ai."""

import asyncio
import time
from typing import Any

import structlog
from crawl4ai import AsyncWebCrawler

from src.models import (
    Collection,
    CollectionMethod,
    CrawlStatus,
    Sample,
    SampleSource,
    URLTarget,
)

logger = structlog.get_logger()


class CrawlerCollector:
    """Collects samples by crawling URLs."""

    def __init__(
        self,
        max_concurrent: int = 5,
        timeout: float = 30.0,
        min_content_length: int = 50,
        max_content_length: int = 50000,
    ):
        """Initialize crawler collector.
        
        Args:
            max_concurrent: Maximum parallel crawls
            timeout: Timeout per URL in seconds
            min_content_length: Minimum content length for valid sample
            max_content_length: Maximum content length for sample
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.min_content_length = min_content_length
        self.max_content_length = max_content_length

    async def crawl(
        self,
        urls: list[str],
        query_context: str = "",
    ) -> tuple[Collection, list[Sample], list[URLTarget]]:
        """Crawl URLs and extract samples.
        
        Args:
            urls: List of URLs to crawl
            query_context: Optional context for sample extraction
            
        Returns:
            Tuple of (Collection, list of Sample objects, list of URLTarget status)
            
        Raises:
            ValueError: If URLs list is invalid
        """
        start_time = time.time()
        
        # Validate input
        if not urls:
            raise ValueError("URLs list cannot be empty")
        
        logger.info(
            "crawling_start",
            url_count=len(urls),
            max_concurrent=self.max_concurrent,
        )
        
        # Initialize URL targets
        url_targets = [
            URLTarget(url=url, status=CrawlStatus.PENDING)
            for url in urls
        ]
        
        # Crawl URLs with concurrency limit
        all_samples: list[Sample] = []
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def crawl_one(target: URLTarget) -> None:
            async with semaphore:
                await self._crawl_url(target, all_samples)
        
        # Run all crawls concurrently
        await asyncio.gather(
            *[crawl_one(target) for target in url_targets],
            return_exceptions=True,
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Count results
        success_count = sum(1 for t in url_targets if t.status == CrawlStatus.SUCCESS)
        failed_count = sum(1 for t in url_targets if t.status == CrawlStatus.FAILED)
        
        logger.info(
            "crawling_complete",
            total_urls=len(urls),
            success=success_count,
            failed=failed_count,
            samples_found=len(all_samples),
            processing_time_ms=processing_time,
        )
        
        collection = Collection(
            method=CollectionMethod.URL_CRAWLING,
            sample_ids=[s.id for s in all_samples],
            query_context=query_context or "URL crawling",
            metadata={
                "processing_time_ms": processing_time,
                "urls_crawled": len(urls),
                "urls_success": success_count,
                "urls_failed": failed_count,
                "samples_found": len(all_samples),
            },
        )
        
        return collection, all_samples, url_targets

    async def _crawl_url(self, target: URLTarget, samples: list[Sample]) -> None:
        """Crawl a single URL and extract samples.
        
        Args:
            target: URLTarget to update with status
            samples: List to append extracted samples
        """
        url_start = time.time()
        target.status = CrawlStatus.CRAWLING
        
        try:
            logger.debug("crawling_url", url=target.url)
            
            # Use crawl4ai to fetch content
            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await asyncio.wait_for(
                    crawler.arun(url=target.url),
                    timeout=self.timeout,
                )
                
                if not result.success:
                    raise RuntimeError(f"Crawl failed: {result.error_message}")
                
                # Extract text content
                content = result.markdown or result.cleaned_html or ""
                content = content.strip()
                
                # Validate content length
                if len(content) < self.min_content_length:
                    logger.warning(
                        "content_too_short",
                        url=target.url,
                        length=len(content),
                    )
                    target.status = CrawlStatus.SUCCESS
                    target.metadata["samples_found"] = 0
                    target.metadata["response_time_ms"] = int((time.time() - url_start) * 1000)
                    return
                
                # Truncate if too long
                if len(content) > self.max_content_length:
                    content = content[:self.max_content_length]
                
                # Create sample
                sample = Sample(
                    content=content,
                    source=SampleSource.URL_CRAWL,
                    metadata={
                        "source_url": target.url,
                        "crawled_at": time.time(),
                    },
                )
                
                samples.append(sample)
                
                # Update target status
                target.status = CrawlStatus.SUCCESS
                target.metadata["samples_found"] = 1
                target.metadata["response_time_ms"] = int((time.time() - url_start) * 1000)
                
                logger.debug(
                    "crawl_success",
                    url=target.url,
                    content_length=len(content),
                    response_time_ms=target.metadata["response_time_ms"],
                )
                
        except asyncio.TimeoutError:
            target.status = CrawlStatus.TIMEOUT
            target.error_message = "Crawl timed out"
            target.metadata["response_time_ms"] = int((time.time() - url_start) * 1000)
            logger.warning("crawl_timeout", url=target.url)
            
        except Exception as e:
            target.status = CrawlStatus.FAILED
            target.error_message = str(e)
            target.metadata["response_time_ms"] = int((time.time() - url_start) * 1000)
            logger.warning("crawl_failed", url=target.url, error=str(e))
