"""Tests for RankingResult model."""

import pytest
from pydantic import ValidationError
from src.models.ranking_result import RankingResult
from src.models.ranked_result import RankedResult, SimilarityDetails
from src.models.sample import Sample
from src.models.sample_source import SampleSource


class TestRankingResult:
    """Test RankingResult model."""

    def create_sample(self, sample_id: str = "sample123") -> Sample:
        """Helper to create a sample for testing."""
        return Sample(
            id=sample_id,
            content="This is a test sample with enough content to meet the minimum length requirement for validation.",
            source=SampleSource.INTERNET_SEARCH,
        )

    def create_ranked_result(self, rank: int, sample_id: str, score: float) -> RankedResult:
        """Helper to create a ranked result for testing."""
        sample = self.create_sample(sample_id)
        return RankedResult(
            rank=rank,
            sample=sample,
            score=score,
        )

    def test_create_valid_ranking_result(self):
        """Test creating a valid ranking result."""
        results = [
            self.create_ranked_result(1, "sample1", 0.95),
            self.create_ranked_result(2, "sample2", 0.85),
        ]
        result = RankingResult(
            results=results,
            total_candidates=10,
            returned_count=2,
            threshold=0.7,
            below_threshold_count=8,
            processing_time_ms=150,
        )
        assert len(result.results) == 2
        assert result.total_candidates == 10
        assert result.returned_count == 2
        assert result.threshold == 0.7
        assert result.below_threshold_count == 8
        assert result.processing_time_ms == 150

    def test_results_can_be_empty(self):
        """Test that results can be empty (no matches)."""
        result = RankingResult(
            results=[],
            total_candidates=10,
            returned_count=0,
            threshold=0.9,
            below_threshold_count=10,
        )
        assert result.results == []
        assert result.total_candidates == 10

    def test_total_candidates_required(self):
        """Test that total_candidates is required."""
        with pytest.raises(ValidationError):
            RankingResult(
                results=[],
                returned_count=0,
                threshold=0.7,
            )

    def test_returned_count_required(self):
        """Test that returned_count is required."""
        with pytest.raises(ValidationError):
            RankingResult(
                results=[],
                total_candidates=10,
                threshold=0.7,
            )

    def test_threshold_required(self):
        """Test that threshold is required."""
        with pytest.raises(ValidationError):
            RankingResult(
                results=[],
                total_candidates=10,
                returned_count=0,
            )

    def test_threshold_validation(self):
        """Test threshold is validated to be between 0 and 1."""
        with pytest.raises(ValidationError):
            RankingResult(
                results=[],
                total_candidates=10,
                returned_count=0,
                threshold=1.5,
            )

    def test_below_threshold_count_defaults(self):
        """Test that below_threshold_count defaults to 0."""
        result = RankingResult(
            results=[],
            total_candidates=10,
            returned_count=0,
            threshold=0.7,
        )
        assert result.below_threshold_count == 0

    def test_processing_time_ms_defaults(self):
        """Test that processing_time_ms defaults to 0."""
        result = RankingResult(
            results=[],
            total_candidates=10,
            returned_count=0,
            threshold=0.7,
        )
        assert result.processing_time_ms == 0

    def test_embeddings_cached_optional(self):
        """Test that embeddings_cached is optional."""
        result = RankingResult(
            results=[],
            total_candidates=10,
            returned_count=0,
            threshold=0.7,
        )
        assert result.embeddings_cached is None

    def test_embeddings_cached_can_be_provided(self):
        """Test that embeddings_cached can be provided."""
        result = RankingResult(
            results=[],
            total_candidates=10,
            returned_count=0,
            threshold=0.7,
            embeddings_cached=5,
        )
        assert result.embeddings_cached == 5

    def test_non_negative_counts(self):
        """Test that counts cannot be negative."""
        with pytest.raises(ValidationError):
            RankingResult(
                results=[],
                total_candidates=-1,
                returned_count=0,
                threshold=0.7,
            )

    def test_model_serialization(self):
        """Test that model can be serialized to dict."""
        results = [self.create_ranked_result(1, "sample1", 0.95)]
        result = RankingResult(
            results=results,
            total_candidates=10,
            returned_count=1,
            threshold=0.7,
            below_threshold_count=9,
            processing_time_ms=150,
            embeddings_cached=3,
        )
        data = result.model_dump()
        assert "results" in data
        assert "total_candidates" in data
        assert "returned_count" in data
        assert "threshold" in data
        assert "below_threshold_count" in data
        assert "processing_time_ms" in data
        assert "embeddings_cached" in data

    def test_many_ranked_results(self):
        """Test ranking with many results."""
        results = [
            self.create_ranked_result(i, f"sample{i}", 1.0 - (i * 0.01))
            for i in range(1, 101)
        ]
        result = RankingResult(
            results=results,
            total_candidates=200,
            returned_count=100,
            threshold=0.5,
            below_threshold_count=100,
            processing_time_ms=500,
        )
        assert len(result.results) == 100
        assert result.total_candidates == 200

    def test_ordering_preserved(self):
        """Test that ranking order is preserved."""
        results = [
            self.create_ranked_result(i, f"sample{i}", 1.0 - (i * 0.1))
            for i in range(1, 11)
        ]
        result = RankingResult(
            results=results,
            total_candidates=20,
            returned_count=10,
            threshold=0.5,
        )
        # Verify order is maintained
        for i, ranked in enumerate(result.results, 1):
            assert ranked.rank == i

