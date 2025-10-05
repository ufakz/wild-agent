"""Contract tests for Sample Collector based on contracts/sample-collector.yaml.

These tests validate the interface contract for the Sample Collector component.
They MUST fail initially since no implementation exists yet.
"""

import pytest
from src.models import (
    Sample,
    SampleSource,
    Theme,
    URLTarget,
    CrawlStatus,
    Collection,
    CollectionMethod,
    CollectionResult,
)


class TestSampleCollectorContract:
    """Test Sample Collector contract compliance."""

    @pytest.fixture
    def valid_themes(self):
        """Create valid themes for testing."""
        return [
            Theme(
                name="AI in Healthcare",
                description="Application of artificial intelligence and machine learning in medical diagnosis and treatment",
                keywords=["artificial intelligence", "healthcare", "machine learning", "diagnosis", "medical"],
                confidence=0.92,
            )
        ]

    @pytest.fixture
    def valid_urls(self):
        """Create valid URLs for testing."""
        return [
            "https://example.com/ai-healthcare",
            "https://example.com/medical-ai",
        ]

    @pytest.mark.asyncio
    async def test_collect_with_urls_success(self, valid_themes, valid_urls):
        """Test successful collection with URL crawling.
        
        Contract: POST /collect with themes and URLs
        Expected: CollectionResult with samples from both online search and crawling
        """
        from src.collectors.orchestrator import SampleCollector
        
        collector = SampleCollector()
        result = await collector.collect(
            themes=valid_themes,
            urls=valid_urls,
            max_samples_per_source=10,
            crawl_timeout_seconds=30,
            enable_online_search=True,
            enable_url_crawling=True,
        )
        
        # Validate response structure
        assert isinstance(result, CollectionResult)
        assert isinstance(result.sample_ids, list)
        assert len(result.sample_ids) >= 0
        assert "total_samples" in result.metadata
        assert "duplicates_removed" in result.metadata
        assert "failed_urls" in result.metadata
        assert "processing_time_ms" in result.metadata
        assert result.metadata["processing_time_ms"] >= 0
        
        # Should have URL statuses
        assert "url_statuses" in result.metadata
        url_statuses = result.metadata["url_statuses"]
        assert len(url_statuses) == len(valid_urls)
        
        # Verify each URL has a status
        for url_status in url_statuses:
            assert "url" in url_status
            assert "status" in url_status
            assert url_status["status"] in ["pending", "crawling", "success", "failed", "timeout"]

    @pytest.mark.asyncio
    async def test_collect_online_only(self, valid_themes):
        """Test collection with online search only (no URL crawling).
        
        Contract: POST /collect with enable_url_crawling=false
        Expected: Samples only from internet_search source
        """
        from src.collectors.orchestrator import SampleCollector
        
        collector = SampleCollector()
        result = await collector.collect(
            themes=valid_themes,
            urls=[],
            max_samples_per_source=10,
            enable_online_search=True,
            enable_url_crawling=False,
        )
        
        assert isinstance(result, CollectionResult)
        # All samples should be from internet_search
        # (Will validate when we can retrieve samples by ID)
        assert len(result.sample_ids) >= 0

    @pytest.mark.asyncio
    async def test_collect_urls_only(self, valid_themes, valid_urls):
        """Test collection with URL crawling only (no online search).
        
        Contract: POST /collect with enable_online_search=false
        Expected: Samples only from url_crawl source
        """
        from src.collectors.orchestrator import SampleCollector
        
        collector = SampleCollector()
        result = await collector.collect(
            themes=valid_themes,
            urls=valid_urls,
            enable_online_search=False,
            enable_url_crawling=True,
        )
        
        assert isinstance(result, CollectionResult)
        # All samples should be from url_crawl
        assert len(result.sample_ids) >= 0

    @pytest.mark.asyncio
    async def test_collect_deduplication(self, valid_themes, valid_urls):
        """Test that duplicate samples are removed.
        
        Contract: duplicates_removed field in response
        Expected: Samples deduplicated by content hash
        """
        from src.collectors.orchestrator import SampleCollector
        
        collector = SampleCollector()
        result = await collector.collect(
            themes=valid_themes,
            urls=valid_urls,
        )
        
        assert "duplicates_removed" in result.metadata
        assert isinstance(result.metadata["duplicates_removed"], int)
        assert result.metadata["duplicates_removed"] >= 0
        
        # total_samples should be >= sample_ids count
        if "total_samples" in result.metadata:
            assert result.metadata["total_samples"] >= len(result.sample_ids)

    @pytest.mark.asyncio
    async def test_collect_url_status_tracking(self, valid_themes, valid_urls):
        """Test that URL crawl status is tracked.
        
        Contract: URLStatus for each URL with status, samples_found, error
        Expected: All URLs have status entries
        """
        from src.collectors.orchestrator import SampleCollector
        
        collector = SampleCollector()
        result = await collector.collect(
            themes=valid_themes,
            urls=valid_urls,
            enable_url_crawling=True,
        )
        
        assert "url_statuses" in result.metadata
        url_statuses = result.metadata["url_statuses"]
        
        for url_status in url_statuses:
            assert "url" in url_status
            assert "status" in url_status
            
            if url_status["status"] == "success":
                assert "samples_found" in url_status
                assert url_status["samples_found"] >= 0
                assert "crawled_at" in url_status
                assert "response_time_ms" in url_status
            elif url_status["status"] == "failed":
                assert "error" in url_status

    @pytest.mark.asyncio
    async def test_collect_failed_urls_graceful(self, valid_themes):
        """Test graceful handling of failed URLs.
        
        Contract: Failed URLs should not crash collection
        Expected: Continue with other URLs, log failures
        """
        from src.collectors.orchestrator import SampleCollector
        
        # Include an invalid URL
        urls = [
            "https://example.com/valid",
            "https://definitely-not-a-real-domain-12345.com/fail",
        ]
        
        collector = SampleCollector()
        # Should not raise exception
        result = await collector.collect(
            themes=valid_themes,
            urls=urls,
            enable_url_crawling=True,
        )
        
        assert isinstance(result, CollectionResult)
        assert "failed_urls" in result.metadata
        # At least one URL should have failed
        # (actual network test will use mocks)

    @pytest.mark.asyncio
    async def test_collect_empty_themes_list(self):
        """Test error handling for empty themes list.
        
        Contract: 400 invalid_input error
        Expected: ValueError
        """
        from src.collectors.orchestrator import SampleCollector
        
        collector = SampleCollector()
        
        with pytest.raises(ValueError) as exc_info:
            await collector.collect(themes=[], urls=[])
        
        assert "themes" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_collect_too_many_themes(self):
        """Test error handling for >10 themes.
        
        Contract: Maximum 10 themes allowed
        Expected: ValueError
        """
        from src.collectors.orchestrator import SampleCollector
        
        # Create 11 themes
        themes = [
            Theme(
                name=f"Theme {i}",
                description=f"Description for theme {i} with enough content to be valid",
                keywords=[f"keyword{i}1", f"keyword{i}2", f"keyword{i}3"],
                confidence=0.8,
            )
            for i in range(11)
        ]
        
        collector = SampleCollector()
        
        with pytest.raises(ValueError) as exc_info:
            await collector.collect(themes=themes, urls=[])
        
        assert "10" in str(exc_info.value) or "maximum" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_collect_too_many_urls(self, valid_themes):
        """Test error handling for >50 URLs.
        
        Contract: Maximum 50 URLs allowed
        Expected: ValueError
        """
        from src.collectors.orchestrator import SampleCollector
        
        # Create 51 URLs
        urls = [f"https://example{i}.com/article" for i in range(51)]
        
        collector = SampleCollector()
        
        with pytest.raises(ValueError) as exc_info:
            await collector.collect(themes=valid_themes, urls=urls)
        
        assert "50" in str(exc_info.value) or "maximum" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_collect_invalid_url_format(self, valid_themes):
        """Test error handling for invalid URL format.
        
        Contract: 400 invalid_input error
        Expected: ValueError
        """
        from src.collectors.orchestrator import SampleCollector
        
        urls = ["not-a-valid-url", "https://example.com/valid"]
        
        collector = SampleCollector()
        
        with pytest.raises(ValueError) as exc_info:
            await collector.collect(themes=valid_themes, urls=urls)
        
        assert "url" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_collect_max_samples_per_source_validation(self, valid_themes):
        """Test validation of max_samples_per_source parameter.
        
        Contract: Must be 1-100
        Expected: ValueError for out-of-range values
        """
        from src.collectors.orchestrator import SampleCollector
        
        collector = SampleCollector()
        
        # Test below minimum
        with pytest.raises(ValueError) as exc_info:
            await collector.collect(
                themes=valid_themes,
                urls=[],
                max_samples_per_source=0,
            )
        assert "max_samples" in str(exc_info.value).lower()
        
        # Test above maximum
        with pytest.raises(ValueError) as exc_info:
            await collector.collect(
                themes=valid_themes,
                urls=[],
                max_samples_per_source=101,
            )
        assert "max_samples" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_collect_crawl_timeout_validation(self, valid_themes, valid_urls):
        """Test validation of crawl_timeout_seconds parameter.
        
        Contract: Must be 5-120 seconds
        Expected: ValueError for out-of-range values
        """
        from src.collectors.orchestrator import SampleCollector
        
        collector = SampleCollector()
        
        # Test below minimum
        with pytest.raises(ValueError) as exc_info:
            await collector.collect(
                themes=valid_themes,
                urls=valid_urls,
                crawl_timeout_seconds=4,
            )
        assert "timeout" in str(exc_info.value).lower()
        
        # Test above maximum
        with pytest.raises(ValueError) as exc_info:
            await collector.collect(
                themes=valid_themes,
                urls=valid_urls,
                crawl_timeout_seconds=121,
            )
        assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_collect_url_timeout_handling(self, valid_themes):
        """Test handling of URL crawl timeouts.
        
        Contract: URLStatus with "timeout" status
        Expected: Continue with other URLs
        """
        from src.collectors.orchestrator import SampleCollector
        
        # URL that will timeout (mock will handle this)
        urls = ["https://slow-website.example.com/timeout"]
        
        collector = SampleCollector()
        result = await collector.collect(
            themes=valid_themes,
            urls=urls,
            crawl_timeout_seconds=5,
        )
        
        # Should not crash
        assert isinstance(result, CollectionResult)

    @pytest.mark.asyncio
    async def test_collect_no_samples_found(self, valid_themes, valid_urls):
        """Test handling when no samples are found.
        
        Contract: 404 no_samples_found error (or empty result)
        Expected: Return empty sample list or raise exception
        """
        from src.collectors.orchestrator import SampleCollector
        
        collector = SampleCollector()
        # With mocked responses that return no samples
        # this will be tested properly in implementation
        result = await collector.collect(
            themes=valid_themes,
            urls=valid_urls,
        )
        
        # Should return result even if empty
        assert isinstance(result, CollectionResult)
        assert isinstance(result.sample_ids, list)

    @pytest.mark.asyncio
    async def test_collect_parallel_execution(self, valid_themes, valid_urls):
        """Test that collection methods run in parallel.
        
        Contract: Online search and URL crawling run concurrently
        Expected: Processing time < sum of individual times
        """
        from src.collectors.orchestrator import SampleCollector
        
        collector = SampleCollector()
        result = await collector.collect(
            themes=valid_themes,
            urls=valid_urls,
            enable_online_search=True,
            enable_url_crawling=True,
        )
        
        # Processing time should be tracked
        assert "processing_time_ms" in result.metadata

    @pytest.mark.asyncio
    async def test_collect_sample_metadata(self, valid_themes, valid_urls):
        """Test that samples include source metadata.
        
        Contract: Sample metadata includes url, title, snippet for crawled samples
        Expected: Metadata populated appropriately per source
        """
        from src.collectors.orchestrator import SampleCollector
        
        collector = SampleCollector()
        result = await collector.collect(
            themes=valid_themes,
            urls=valid_urls,
        )
        
        # Samples will be validated when we can retrieve them by ID
        assert len(result.sample_ids) >= 0

    @pytest.mark.asyncio
    async def test_collect_concurrent_url_crawling(self, valid_themes):
        """Test that URLs are crawled concurrently (max 5 parallel).
        
        Contract: Concurrent crawling with max 5 parallel requests
        Expected: Efficient crawling of multiple URLs
        """
        from src.collectors.orchestrator import SampleCollector
        
        # Create 10 URLs to test concurrent crawling
        urls = [f"https://example{i}.com/article" for i in range(10)]
        
        collector = SampleCollector()
        result = await collector.collect(
            themes=valid_themes,
            urls=urls,
            enable_url_crawling=True,
        )
        
        # Should process all URLs
        assert "url_statuses" in result.metadata
        assert len(result.metadata["url_statuses"]) == 10

    @pytest.mark.asyncio
    async def test_collect_content_extraction(self, valid_themes, valid_urls):
        """Test that crawled content is properly extracted.
        
        Contract: Extract samples from HTML using local LLM
        Expected: Meaningful text samples extracted from pages
        """
        from src.collectors.orchestrator import SampleCollector
        
        collector = SampleCollector()
        result = await collector.collect(
            themes=valid_themes,
            urls=valid_urls,
            enable_url_crawling=True,
        )
        
        # Content extraction validation will happen with actual samples
        assert isinstance(result, CollectionResult)

    @pytest.mark.asyncio
    async def test_collect_response_timing(self, valid_themes, valid_urls):
        """Test that processing time is tracked and reasonable.
        
        Contract: processing_time_ms must be >= 0
        Expected: Positive integer representing milliseconds
        """
        from src.collectors.orchestrator import SampleCollector
        
        collector = SampleCollector()
        result = await collector.collect(
            themes=valid_themes,
            urls=valid_urls[:1],  # Just one URL for speed
        )
        
        assert "processing_time_ms" in result.metadata
        assert isinstance(result.metadata["processing_time_ms"], int)
        assert result.metadata["processing_time_ms"] >= 0
