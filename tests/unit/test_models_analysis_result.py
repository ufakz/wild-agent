"""Tests for AnalysisResult model."""

import pytest
from pydantic import ValidationError
from src.models.analysis_result import AnalysisResult


class TestAnalysisResult:
    """Test AnalysisResult model."""

    def test_create_valid_analysis_result(self):
        """Test creating a valid analysis result."""
        result = AnalysisResult(
            query_context="Find articles about machine learning",
            theme_ids=["theme1", "theme2"],
            url_target_ids=["url1", "url2", "url3"],
        )
        assert result.query_context == "Find articles about machine learning"
        assert result.theme_ids == ["theme1", "theme2"]
        assert result.url_target_ids == ["url1", "url2", "url3"]
        assert result.metadata == {}

    def test_query_context_required(self):
        """Test that query_context is required."""
        with pytest.raises(ValidationError):
            AnalysisResult(
                theme_ids=["theme1"],
                url_target_ids=["url1"],
            )

    def test_theme_ids_required(self):
        """Test that theme_ids is required."""
        with pytest.raises(ValidationError):
            AnalysisResult(
                query_context="Find articles",
                url_target_ids=["url1"],
            )

    def test_url_target_ids_required(self):
        """Test that url_target_ids is required."""
        with pytest.raises(ValidationError):
            AnalysisResult(
                query_context="Find articles",
                theme_ids=["theme1"],
            )

    def test_theme_ids_can_be_empty(self):
        """Test that theme_ids can be empty list."""
        result = AnalysisResult(
            query_context="Find articles about machine learning",
            theme_ids=[],
            url_target_ids=["url1"],
        )
        assert result.theme_ids == []

    def test_url_target_ids_can_be_empty(self):
        """Test that url_target_ids can be empty list."""
        result = AnalysisResult(
            query_context="Find articles about machine learning",
            theme_ids=["theme1"],
            url_target_ids=[],
        )
        assert result.url_target_ids == []

    def test_both_can_be_empty(self):
        """Test that both lists can be empty (failed analysis)."""
        result = AnalysisResult(
            query_context="Find articles about machine learning",
            theme_ids=[],
            url_target_ids=[],
        )
        assert result.theme_ids == []
        assert result.url_target_ids == []

    def test_metadata_default_empty(self):
        """Test that metadata defaults to empty dict."""
        result = AnalysisResult(
            query_context="Find articles",
            theme_ids=["theme1"],
            url_target_ids=["url1"],
        )
        assert result.metadata == {}

    def test_metadata_can_be_provided(self):
        """Test that metadata can be provided."""
        metadata = {
            "llm_model": "grok-beta",
            "confidence": 0.85,
            "processing_time_ms": 250,
        }
        result = AnalysisResult(
            query_context="Find articles",
            theme_ids=["theme1"],
            url_target_ids=["url1"],
            metadata=metadata,
        )
        assert result.metadata == metadata

    def test_many_themes(self):
        """Test result with many themes."""
        theme_ids = [f"theme{i}" for i in range(20)]
        result = AnalysisResult(
            query_context="Find articles",
            theme_ids=theme_ids,
            url_target_ids=["url1"],
        )
        assert len(result.theme_ids) == 20

    def test_many_urls(self):
        """Test result with many URLs."""
        url_ids = [f"url{i}" for i in range(50)]
        result = AnalysisResult(
            query_context="Find articles",
            theme_ids=["theme1"],
            url_target_ids=url_ids,
        )
        assert len(result.url_target_ids) == 50

    def test_model_serialization(self):
        """Test that model can be serialized to dict."""
        result = AnalysisResult(
            query_context="Find articles about testing",
            theme_ids=["theme1", "theme2"],
            url_target_ids=["url1", "url2"],
            metadata={"key": "value"},
        )
        data = result.model_dump()
        assert "query_context" in data
        assert "theme_ids" in data
        assert "url_target_ids" in data
        assert "metadata" in data
