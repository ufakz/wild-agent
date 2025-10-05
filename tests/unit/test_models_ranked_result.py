"""Tests for RankedResult model."""

import pytest
from pydantic import ValidationError
from src.models.ranked_result import RankedResult, SimilarityDetails
from src.models.sample import Sample
from src.models.sample_source import SampleSource


class TestSimilarityDetails:
    """Test SimilarityDetails nested model."""

    def test_create_valid_similarity_details(self):
        """Test creating valid similarity details."""
        details = SimilarityDetails(
            max_score=0.95,
            avg_score=0.85,
            closest_reference_id="ref123",
        )
        assert details.max_score == 0.95
        assert details.avg_score == 0.85
        assert details.closest_reference_id == "ref123"

    def test_max_score_boundaries(self):
        """Test max_score boundary values."""
        details1 = SimilarityDetails(
            max_score=0.0,
            avg_score=0.0,
            closest_reference_id="ref123",
        )
        assert details1.max_score == 0.0

        details2 = SimilarityDetails(
            max_score=1.0,
            avg_score=1.0,
            closest_reference_id="ref123",
        )
        assert details2.max_score == 1.0

    def test_max_score_validation(self):
        """Test that max_score is validated."""
        with pytest.raises(ValidationError):
            SimilarityDetails(
                max_score=1.5,
                avg_score=0.85,
                closest_reference_id="ref123",
            )


class TestRankedResult:
    """Test RankedResult model."""

    def create_sample(self, sample_id: str = "sample123") -> Sample:
        """Helper to create a sample for testing."""
        return Sample(
            id=sample_id,
            content="This is a test sample with enough content to meet the minimum length requirement for validation.",
            source=SampleSource.INTERNET_SEARCH,
        )

    def test_create_valid_ranked_result(self):
        """Test creating a valid ranked result."""
        sample = self.create_sample()
        details = SimilarityDetails(
            max_score=0.95,
            avg_score=0.90,
            closest_reference_id="ref1",
        )
        result = RankedResult(
            rank=1,
            sample=sample,
            score=0.95,
            similarity_details=details,
        )
        assert result.rank == 1
        assert result.sample.id == "sample123"
        assert result.score == 0.95
        assert result.similarity_details.max_score == 0.95

    def test_rank_below_minimum(self):
        """Test that rank below 1 fails."""
        sample = self.create_sample()
        with pytest.raises(ValidationError) as exc_info:
            RankedResult(
                rank=0,
                sample=sample,
                score=0.95,
            )
        assert "rank" in str(exc_info.value).lower()

    def test_rank_valid_values(self):
        """Test that rank can be any positive integer."""
        for rank in [1, 10, 100, 1000]:
            sample = self.create_sample(f"sample{rank}")
            result = RankedResult(
                rank=rank,
                sample=sample,
                score=0.95,
            )
            assert result.rank == rank

    def test_score_below_zero(self):
        """Test that score below 0.0 fails."""
        sample = self.create_sample()
        with pytest.raises(ValidationError) as exc_info:
            RankedResult(
                rank=1,
                sample=sample,
                score=-0.1,
            )
        assert "score" in str(exc_info.value).lower()

    def test_score_above_one(self):
        """Test that score above 1.0 fails."""
        sample = self.create_sample()
        with pytest.raises(ValidationError) as exc_info:
            RankedResult(
                rank=1,
                sample=sample,
                score=1.1,
            )
        assert "score" in str(exc_info.value).lower()

    def test_score_boundaries(self):
        """Test score boundary values."""
        sample1 = self.create_sample("sample1")
        result1 = RankedResult(
            rank=1,
            sample=sample1,
            score=0.0,
        )
        assert result1.score == 0.0

        sample2 = self.create_sample("sample2")
        result2 = RankedResult(
            rank=1,
            sample=sample2,
            score=1.0,
        )
        assert result2.score == 1.0

    def test_similarity_details_optional(self):
        """Test that similarity_details is optional."""
        sample = self.create_sample()
        result = RankedResult(
            rank=1,
            sample=sample,
            score=0.95,
        )
        assert result.similarity_details is None

    def test_sample_required(self):
        """Test that sample is required."""
        with pytest.raises(ValidationError):
            RankedResult(
                rank=1,
                score=0.95,
            )

    def test_rank_required(self):
        """Test that rank is required."""
        sample = self.create_sample()
        with pytest.raises(ValidationError):
            RankedResult(
                sample=sample,
                score=0.95,
            )

    def test_score_required(self):
        """Test that score is required."""
        sample = self.create_sample()
        with pytest.raises(ValidationError):
            RankedResult(
                rank=1,
                sample=sample,
            )

    def test_high_rank_low_score(self):
        """Test that high rank can have low score (outlier case)."""
        sample = self.create_sample()
        result = RankedResult(
            rank=100,
            sample=sample,
            score=0.1,
        )
        assert result.rank == 100
        assert result.score == 0.1

    def test_low_rank_high_score(self):
        """Test typical case: low rank with high score."""
        sample = self.create_sample()
        result = RankedResult(
            rank=1,
            sample=sample,
            score=0.99,
        )
        assert result.rank == 1
        assert result.score == 0.99

    def test_model_serialization(self):
        """Test that model can be serialized to dict."""
        sample = self.create_sample()
        details = SimilarityDetails(
            max_score=0.95,
            avg_score=0.85,
            closest_reference_id="ref1",
        )
        result = RankedResult(
            rank=5,
            sample=sample,
            score=0.85,
            similarity_details=details,
        )
        data = result.model_dump()
        assert "rank" in data
        assert "sample" in data
        assert "score" in data
        assert "similarity_details" in data

    def test_multiple_results_different_ranks(self):
        """Test creating multiple results with different ranks."""
        results = []
        for i in range(1, 11):
            sample = self.create_sample(f"sample{i}")
            result = RankedResult(
                rank=i,
                sample=sample,
                score=1.0 - (i * 0.05),
            )
            results.append(result)
        
        # Verify ranks are unique and ordered
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, 11))
        
        # Verify scores decrease with rank
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
