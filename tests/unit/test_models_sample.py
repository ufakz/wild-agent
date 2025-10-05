"""Tests for Sample model."""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError
from src.models.sample import Sample
from src.models.sample_source import SampleSource


class TestSample:
    """Test Sample model."""

    def test_create_valid_sample(self):
        """Test creating a valid sample."""
        sample = Sample(
            content="A" * 100,  # Valid length
            source=SampleSource.USER_INPUT,
        )
        assert sample.id is not None
        assert len(sample.content) == 100
        assert sample.source == SampleSource.USER_INPUT
        assert sample.content_hash is not None
        assert isinstance(sample.created_at, datetime)
        assert isinstance(sample.metadata, dict)

    def test_content_too_short(self):
        """Test that content shorter than 50 chars fails."""
        with pytest.raises(ValidationError) as exc_info:
            Sample(
                content="Short",  # Only 5 chars
                source=SampleSource.USER_INPUT,
            )
        assert "content" in str(exc_info.value).lower()

    def test_content_too_long(self):
        """Test that content longer than 50000 chars fails."""
        with pytest.raises(ValidationError) as exc_info:
            Sample(
                content="A" * 50001,
                source=SampleSource.USER_INPUT,
            )
        assert "content" in str(exc_info.value).lower()

    def test_content_empty(self):
        """Test that empty content fails."""
        with pytest.raises(ValidationError):
            Sample(
                content="",
                source=SampleSource.USER_INPUT,
            )

    def test_content_whitespace_only(self):
        """Test that whitespace-only content fails."""
        with pytest.raises(ValidationError):
            Sample(
                content="   " * 20,  # 60 spaces but no actual content
                source=SampleSource.USER_INPUT,
            )

    def test_content_hash_computed(self):
        """Test that content hash is automatically computed."""
        content = "Test content that is long enough to be valid for our sample model"
        sample = Sample(
            content=content,
            source=SampleSource.USER_INPUT,
        )
        assert sample.content_hash is not None
        assert len(sample.content_hash) == 64  # SHA-256 hex digest

    def test_content_hash_same_for_same_content(self):
        """Test that same content produces same hash."""
        content = "Test content that is long enough to be valid for our sample model"
        sample1 = Sample(content=content, source=SampleSource.USER_INPUT)
        sample2 = Sample(content=content, source=SampleSource.USER_INPUT)
        assert sample1.content_hash == sample2.content_hash

    def test_content_hash_different_for_different_content(self):
        """Test that different content produces different hashes."""
        sample1 = Sample(
            content="First test content that is long enough to be valid",
            source=SampleSource.USER_INPUT,
        )
        sample2 = Sample(
            content="Second test content that is long enough to be valid",
            source=SampleSource.USER_INPUT,
        )
        assert sample1.content_hash != sample2.content_hash

    def test_metadata_optional(self):
        """Test that metadata is optional."""
        sample = Sample(
            content="A" * 100,
            source=SampleSource.USER_INPUT,
        )
        assert sample.metadata == {}

    def test_metadata_can_be_provided(self):
        """Test that metadata can be provided."""
        metadata = {"url": "https://example.com", "title": "Test"}
        sample = Sample(
            content="A" * 100,
            source=SampleSource.USER_INPUT,
            metadata=metadata,
        )
        assert sample.metadata == metadata

    def test_created_at_auto_set(self):
        """Test that created_at is automatically set."""
        sample = Sample(
            content="A" * 100,
            source=SampleSource.USER_INPUT,
        )
        assert sample.created_at is not None
        assert isinstance(sample.created_at, datetime)

    def test_id_auto_generated(self):
        """Test that ID is automatically generated."""
        sample = Sample(
            content="A" * 100,
            source=SampleSource.USER_INPUT,
        )
        assert sample.id is not None
        assert len(sample.id) == 36  # UUID4 format

    def test_different_samples_have_different_ids(self):
        """Test that different samples get different IDs."""
        sample1 = Sample(content="A" * 100, source=SampleSource.USER_INPUT)
        sample2 = Sample(content="B" * 100, source=SampleSource.USER_INPUT)
        assert sample1.id != sample2.id

    def test_all_sources_valid(self):
        """Test that all source types are valid."""
        content = "A" * 100
        for source in SampleSource:
            sample = Sample(content=content, source=source)
            assert sample.source == source

    def test_model_serialization(self):
        """Test that model can be serialized to dict."""
        sample = Sample(
            content="A" * 100,
            source=SampleSource.USER_INPUT,
            metadata={"key": "value"},
        )
        data = sample.model_dump()
        assert "id" in data
        assert "content" in data
        assert "content_hash" in data
        assert "source" in data
        assert "metadata" in data
        assert "created_at" in data
