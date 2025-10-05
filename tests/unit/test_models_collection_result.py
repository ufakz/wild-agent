"""Tests for CollectionResult model."""

import pytest
from pydantic import ValidationError
from src.models.collection_result import CollectionResult


class TestCollectionResult:
    """Test CollectionResult model."""

    def test_create_valid_collection_result(self):
        """Test creating a valid collection result."""
        result = CollectionResult(
            collection_id="col123",
            sample_ids=["sample1", "sample2", "sample3"],
        )
        assert result.collection_id == "col123"
        assert result.sample_ids == ["sample1", "sample2", "sample3"]
        assert result.metadata == {}

    def test_collection_id_required(self):
        """Test that collection_id is required."""
        with pytest.raises(ValidationError):
            CollectionResult(
                sample_ids=["sample1"],
            )

    def test_sample_ids_required(self):
        """Test that sample_ids is required."""
        with pytest.raises(ValidationError):
            CollectionResult(
                collection_id="col123",
            )

    def test_sample_ids_can_be_empty(self):
        """Test that sample_ids can be empty (failed collection)."""
        result = CollectionResult(
            collection_id="col123",
            sample_ids=[],
        )
        assert result.sample_ids == []

    def test_single_sample(self):
        """Test collection with single sample."""
        result = CollectionResult(
            collection_id="col123",
            sample_ids=["sample1"],
        )
        assert len(result.sample_ids) == 1

    def test_many_samples(self):
        """Test collection with many samples."""
        sample_ids = [f"sample{i}" for i in range(100)]
        result = CollectionResult(
            collection_id="col123",
            sample_ids=sample_ids,
        )
        assert len(result.sample_ids) == 100

    def test_metadata_default_empty(self):
        """Test that metadata defaults to empty dict."""
        result = CollectionResult(
            collection_id="col123",
            sample_ids=["sample1"],
        )
        assert result.metadata == {}

    def test_metadata_can_be_provided(self):
        """Test that metadata can be provided."""
        metadata = {
            "method": "online_search",
            "total_found": 100,
            "collected": 50,
            "duplicates_removed": 10,
            "processing_time_ms": 5000,
        }
        result = CollectionResult(
            collection_id="col123",
            sample_ids=["sample1", "sample2"],
            metadata=metadata,
        )
        assert result.metadata == metadata

    def test_model_serialization(self):
        """Test that model can be serialized to dict."""
        result = CollectionResult(
            collection_id="col123",
            sample_ids=["sample1", "sample2"],
            metadata={"key": "value"},
        )
        data = result.model_dump()
        assert "collection_id" in data
        assert "sample_ids" in data
        assert "metadata" in data

    def test_metadata_with_stats(self):
        """Test metadata can store collection statistics."""
        metadata = {
            "urls_crawled": 25,
            "urls_failed": 3,
            "search_results": 50,
            "avg_content_length": 2500,
        }
        result = CollectionResult(
            collection_id="col123",
            sample_ids=[f"sample{i}" for i in range(22)],
            metadata=metadata,
        )
        assert result.metadata["urls_crawled"] == 25
        assert result.metadata["urls_failed"] == 3
        assert len(result.sample_ids) == 22
