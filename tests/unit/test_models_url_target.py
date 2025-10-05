"""Tests for URLTarget model."""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError
from src.models.url_target import URLTarget
from src.models.crawl_status import CrawlStatus


class TestURLTarget:
    """Test URLTarget model."""

    def test_create_valid_url_target(self):
        """Test creating a valid URL target."""
        url = "https://example.com/article"
        target = URLTarget(url=url, priority=5)
        assert target.id is not None
        assert target.url == url
        assert target.priority == 5
        assert target.status == CrawlStatus.PENDING
        assert target.discovered_at is not None
        assert target.crawled_at is None
        assert target.error_message is None
        assert target.metadata == {}

    def test_valid_http_url(self):
        """Test that HTTP URLs are valid."""
        target = URLTarget(url="http://example.com", priority=1)
        assert target.url == "http://example.com"

    def test_valid_https_url(self):
        """Test that HTTPS URLs are valid."""
        target = URLTarget(url="https://example.com", priority=1)
        assert target.url == "https://example.com"

    def test_invalid_url_scheme(self):
        """Test that non-HTTP(S) URLs fail."""
        with pytest.raises(ValidationError) as exc_info:
            URLTarget(url="ftp://example.com", priority=1)
        assert "url" in str(exc_info.value).lower()

    def test_invalid_url_format(self):
        """Test that malformed URLs fail."""
        with pytest.raises(ValidationError) as exc_info:
            URLTarget(url="not a url", priority=1)
        assert "url" in str(exc_info.value).lower()

    def test_priority_below_minimum(self):
        """Test that priority below 1 fails."""
        with pytest.raises(ValidationError) as exc_info:
            URLTarget(url="https://example.com", priority=0)
        assert "priority" in str(exc_info.value).lower()

    def test_priority_above_maximum(self):
        """Test that priority above 10 fails."""
        with pytest.raises(ValidationError) as exc_info:
            URLTarget(url="https://example.com", priority=11)
        assert "priority" in str(exc_info.value).lower()

    def test_priority_boundaries(self):
        """Test priority boundary values."""
        target1 = URLTarget(url="https://example.com", priority=1)
        assert target1.priority == 1

        target2 = URLTarget(url="https://example.com", priority=10)
        assert target2.priority == 10

    def test_status_defaults_to_pending(self):
        """Test that status defaults to PENDING."""
        target = URLTarget(url="https://example.com", priority=1)
        assert target.status == CrawlStatus.PENDING

    def test_status_can_be_set(self):
        """Test that status can be set during creation."""
        target = URLTarget(
            url="https://example.com",
            priority=1,
            status=CrawlStatus.CRAWLING,
        )
        assert target.status == CrawlStatus.CRAWLING

    def test_discovered_at_auto_set(self):
        """Test that discovered_at is automatically set."""
        target = URLTarget(url="https://example.com", priority=1)
        assert target.discovered_at is not None
        assert isinstance(target.discovered_at, datetime)

    def test_crawled_at_defaults_to_none(self):
        """Test that crawled_at defaults to None."""
        target = URLTarget(url="https://example.com", priority=1)
        assert target.crawled_at is None

    def test_crawled_at_can_be_set(self):
        """Test that crawled_at can be set."""
        now = datetime.now(timezone.utc)
        target = URLTarget(
            url="https://example.com",
            priority=1,
            crawled_at=now,
        )
        assert target.crawled_at == now

    def test_error_message_defaults_to_none(self):
        """Test that error_message defaults to None."""
        target = URLTarget(url="https://example.com", priority=1)
        assert target.error_message is None

    def test_error_message_can_be_set(self):
        """Test that error_message can be set."""
        error = "Connection timeout"
        target = URLTarget(
            url="https://example.com",
            priority=1,
            error_message=error,
        )
        assert target.error_message == error

    def test_metadata_defaults_to_empty(self):
        """Test that metadata defaults to empty dict."""
        target = URLTarget(url="https://example.com", priority=1)
        assert target.metadata == {}

    def test_metadata_can_be_provided(self):
        """Test that metadata can be provided."""
        metadata = {"depth": 2, "referrer": "https://search.example.com"}
        target = URLTarget(
            url="https://example.com",
            priority=1,
            metadata=metadata,
        )
        assert target.metadata == metadata

    def test_id_auto_generated(self):
        """Test that ID is automatically generated."""
        target = URLTarget(url="https://example.com", priority=1)
        assert target.id is not None
        assert len(target.id) == 36  # UUID4 format

    def test_different_targets_have_different_ids(self):
        """Test that different targets get different IDs."""
        target1 = URLTarget(url="https://example.com", priority=1)
        target2 = URLTarget(url="https://example.org", priority=1)
        assert target1.id != target2.id

    def test_all_statuses_valid(self):
        """Test that all status types are valid."""
        url = "https://example.com"
        for status in CrawlStatus:
            target = URLTarget(url=url, priority=1, status=status)
            assert target.status == status

    def test_url_with_path(self):
        """Test URLs with paths."""
        url = "https://example.com/articles/2024/test"
        target = URLTarget(url=url, priority=1)
        assert target.url == url

    def test_url_with_query_params(self):
        """Test URLs with query parameters."""
        url = "https://example.com/search?q=test&page=2"
        target = URLTarget(url=url, priority=1)
        assert target.url == url

    def test_url_with_fragment(self):
        """Test URLs with fragments."""
        url = "https://example.com/page#section"
        target = URLTarget(url=url, priority=1)
        assert target.url == url

    def test_model_serialization(self):
        """Test that model can be serialized to dict."""
        target = URLTarget(
            url="https://example.com",
            priority=5,
            status=CrawlStatus.SUCCESS,
            metadata={"key": "value"},
        )
        data = target.model_dump()
        assert "id" in data
        assert "url" in data
        assert "priority" in data
        assert "status" in data
        assert "discovered_at" in data
        assert "crawled_at" in data
        assert "error_message" in data
        assert "metadata" in data
