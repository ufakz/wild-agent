"""Tests for Collection model."""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError
from src.models.collection import Collection
from src.models.collection_method import CollectionMethod


class TestCollection:
    """Test Collection model."""

    def test_create_valid_collection(self):
        """Test creating a valid collection."""
        query_context = "Find articles about machine learning best practices" * 5  # Make it long enough
        collection = Collection(
            query_context=query_context,
            method=CollectionMethod.ONLINE_SEARCH,
        )
        assert collection.id is not None
        assert collection.query_context == query_context
        assert collection.method == CollectionMethod.ONLINE_SEARCH
        assert collection.created_at is not None
        assert collection.sample_ids == []
        assert collection.url_target_ids == []
        assert collection.theme_ids == []
        assert collection.metadata == {}

    def test_query_context_too_short(self):
        """Test that query_context shorter than 10 chars fails."""
        with pytest.raises(ValidationError) as exc_info:
            Collection(
                query_context="Short",  # Only 5 chars
                method=CollectionMethod.ONLINE_SEARCH,
            )
        assert "query_context" in str(exc_info.value).lower()

    def test_query_context_too_long(self):
        """Test that query_context longer than 5000 chars fails."""
        with pytest.raises(ValidationError) as exc_info:
            Collection(
                query_context="A" * 5001,
                method=CollectionMethod.ONLINE_SEARCH,
            )
        assert "query_context" in str(exc_info.value).lower()

    def test_query_context_boundaries(self):
        """Test query_context boundary values."""
        # Exactly 10 chars (minimum)
        collection1 = Collection(
            query_context="A" * 10,
            method=CollectionMethod.ONLINE_SEARCH,
        )
        assert len(collection1.query_context) == 10

        # Exactly 5000 chars (maximum)
        collection2 = Collection(
            query_context="A" * 5000,
            method=CollectionMethod.ONLINE_SEARCH,
        )
        assert len(collection2.query_context) == 5000

    def test_all_methods_valid(self):
        """Test that all collection methods are valid."""
        query = "Find articles about testing" * 5  # Make it long enough
        for method in CollectionMethod:
            collection = Collection(query_context=query, method=method)
            assert collection.method == method

    def test_sample_ids_default_empty(self):
        """Test that sample_ids defaults to empty list."""
        collection = Collection(
            query_context="Find articles about testing" * 5,
            method=CollectionMethod.ONLINE_SEARCH,
        )
        assert collection.sample_ids == []

    def test_sample_ids_can_be_provided(self):
        """Test that sample_ids can be provided."""
        sample_ids = ["id1", "id2", "id3"]
        collection = Collection(
            query_context="Find articles about testing" * 5,
            method=CollectionMethod.ONLINE_SEARCH,
            sample_ids=sample_ids,
        )
        assert collection.sample_ids == sample_ids

    def test_url_target_ids_default_empty(self):
        """Test that url_target_ids defaults to empty list."""
        collection = Collection(
            query_context="Find articles about testing" * 5,
            method=CollectionMethod.ONLINE_SEARCH,
        )
        assert collection.url_target_ids == []

    def test_url_target_ids_can_be_provided(self):
        """Test that url_target_ids can be provided."""
        url_ids = ["url1", "url2"]
        collection = Collection(
            query_context="Find articles about testing" * 5,
            method=CollectionMethod.URL_CRAWLING,
            url_target_ids=url_ids,
        )
        assert collection.url_target_ids == url_ids

    def test_theme_ids_default_empty(self):
        """Test that theme_ids defaults to empty list."""
        collection = Collection(
            query_context="Find articles about testing" * 5,
            method=CollectionMethod.ONLINE_SEARCH,
        )
        assert collection.theme_ids == []

    def test_theme_ids_can_be_provided(self):
        """Test that theme_ids can be provided."""
        theme_ids = ["theme1", "theme2"]
        collection = Collection(
            query_context="Find articles about testing" * 5,
            method=CollectionMethod.ONLINE_SEARCH,
            theme_ids=theme_ids,
        )
        assert collection.theme_ids == theme_ids

    def test_metadata_default_empty(self):
        """Test that metadata defaults to empty dict."""
        collection = Collection(
            query_context="Find articles about testing" * 5,
            method=CollectionMethod.ONLINE_SEARCH,
        )
        assert collection.metadata == {}

    def test_metadata_can_be_provided(self):
        """Test that metadata can be provided."""
        metadata = {"max_results": 50, "language": "en"}
        collection = Collection(
            query_context="Find articles about testing" * 5,
            method=CollectionMethod.ONLINE_SEARCH,
            metadata=metadata,
        )
        assert collection.metadata == metadata

    def test_created_at_auto_set(self):
        """Test that created_at is automatically set."""
        collection = Collection(
            query_context="Find articles about testing" * 5,
            method=CollectionMethod.ONLINE_SEARCH,
        )
        assert collection.created_at is not None
        assert isinstance(collection.created_at, datetime)

    def test_id_auto_generated(self):
        """Test that ID is automatically generated."""
        collection = Collection(
            query_context="Find articles about testing" * 5,
            method=CollectionMethod.ONLINE_SEARCH,
        )
        assert collection.id is not None
        assert len(collection.id) == 36  # UUID4 format

    def test_different_collections_have_different_ids(self):
        """Test that different collections get different IDs."""
        query = "Find articles about testing" * 5
        collection1 = Collection(
            query_context=query,
            method=CollectionMethod.ONLINE_SEARCH,
        )
        collection2 = Collection(
            query_context=query,
            method=CollectionMethod.URL_CRAWLING,
        )
        assert collection1.id != collection2.id

    def test_online_search_collection(self):
        """Test collection with ONLINE_SEARCH method."""
        collection = Collection(
            query_context="Find Python tutorials for beginners" * 5,
            method=CollectionMethod.ONLINE_SEARCH,
            theme_ids=["theme1", "theme2"],
        )
        assert collection.method == CollectionMethod.ONLINE_SEARCH
        assert len(collection.theme_ids) == 2

    def test_url_crawling_collection(self):
        """Test collection with URL_CRAWLING method."""
        collection = Collection(
            query_context="Crawl documentation sites for API examples" * 3,
            method=CollectionMethod.URL_CRAWLING,
            url_target_ids=["url1", "url2", "url3"],
        )
        assert collection.method == CollectionMethod.URL_CRAWLING
        assert len(collection.url_target_ids) == 3

    def test_manual_import_collection(self):
        """Test collection with MANUAL_IMPORT method."""
        collection = Collection(
            query_context="Import samples from local files" * 5,
            method=CollectionMethod.MANUAL_IMPORT,
            sample_ids=["sample1", "sample2"],
        )
        assert collection.method == CollectionMethod.MANUAL_IMPORT
        assert len(collection.sample_ids) == 2

    def test_model_serialization(self):
        """Test that model can be serialized to dict."""
        collection = Collection(
            query_context="Find articles about testing" * 5,
            method=CollectionMethod.ONLINE_SEARCH,
            sample_ids=["s1"],
            url_target_ids=["u1"],
            theme_ids=["t1"],
            metadata={"key": "value"},
        )
        data = collection.model_dump()
        assert "id" in data
        assert "query_context" in data
        assert "method" in data
        assert "sample_ids" in data
        assert "url_target_ids" in data
        assert "theme_ids" in data
        assert "metadata" in data
        assert "created_at" in data
