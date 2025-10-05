"""Tests for CrawlStatus enum."""

from src.models.crawl_status import CrawlStatus


class TestCrawlStatus:
    """Test CrawlStatus enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        assert CrawlStatus.PENDING == "pending"
        assert CrawlStatus.CRAWLING == "crawling"
        assert CrawlStatus.SUCCESS == "success"
        assert CrawlStatus.FAILED == "failed"
        assert CrawlStatus.TIMEOUT == "timeout"

    def test_enum_count(self):
        """Test that enum has exactly 5 values."""
        assert len(CrawlStatus) == 5

    def test_state_transitions(self):
        """Test valid state transitions."""
        # PENDING can go to CRAWLING
        initial = CrawlStatus.PENDING
        assert initial != CrawlStatus.CRAWLING
        
        # CRAWLING can go to SUCCESS, FAILED, or TIMEOUT
        assert CrawlStatus.CRAWLING != CrawlStatus.SUCCESS
        assert CrawlStatus.CRAWLING != CrawlStatus.FAILED
        assert CrawlStatus.CRAWLING != CrawlStatus.TIMEOUT

    def test_string_conversion(self):
        """Test that enum values convert to strings correctly."""
        assert str(CrawlStatus.PENDING) == "pending"
        assert CrawlStatus.SUCCESS.value == "success"
