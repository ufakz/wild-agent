"""Tests for SampleSource enum."""

import pytest
from src.models.sample_source import SampleSource


class TestSampleSource:
    """Test SampleSource enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        assert SampleSource.USER_INPUT == "user_input"
        assert SampleSource.INTERNET_SEARCH == "internet_search"
        assert SampleSource.URL_CRAWL == "url_crawl"
        assert SampleSource.FILE_IMPORT == "file_import"

    def test_enum_count(self):
        """Test that enum has exactly 4 values."""
        assert len(SampleSource) == 4

    def test_string_conversion(self):
        """Test that enum values convert to strings correctly."""
        assert str(SampleSource.USER_INPUT) == "user_input"
        assert SampleSource.USER_INPUT.value == "user_input"

    def test_enum_comparison(self):
        """Test that enum values can be compared."""
        assert SampleSource.USER_INPUT == SampleSource.USER_INPUT
        assert SampleSource.USER_INPUT != SampleSource.INTERNET_SEARCH

    def test_enum_iteration(self):
        """Test that enum can be iterated."""
        values = [source.value for source in SampleSource]
        assert "user_input" in values
        assert "internet_search" in values
        assert "url_crawl" in values
        assert "file_import" in values
