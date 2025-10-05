"""Tests for CollectionMethod enum."""

from src.models.collection_method import CollectionMethod


class TestCollectionMethod:
    """Test CollectionMethod enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        assert CollectionMethod.ONLINE_SEARCH == "online_search"
        assert CollectionMethod.URL_CRAWLING == "url_crawling"
        assert CollectionMethod.MANUAL_IMPORT == "manual_import"

    def test_enum_count(self):
        """Test that enum has exactly 3 values."""
        assert len(CollectionMethod) == 3

    def test_string_conversion(self):
        """Test that enum values convert to strings correctly."""
        assert str(CollectionMethod.ONLINE_SEARCH) == "online_search"
        assert CollectionMethod.URL_CRAWLING.value == "url_crawling"

    def test_enum_usage(self):
        """Test enum can be used for comparison."""
        method = CollectionMethod.ONLINE_SEARCH
        assert method == CollectionMethod.ONLINE_SEARCH
        assert method != CollectionMethod.URL_CRAWLING
