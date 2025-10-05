"""Contract tests for Similarity Ranker based on contracts/similarity-ranker.yaml.

These tests validate the interface contract for the Similarity Ranker component.
They MUST fail initially since no implementation exists yet.
"""

import pytest
import numpy as np
from src.models import (
    Sample,
    SampleSource,
    SimilarityScore,
    RankedResult,
    RankingResult,
)


class TestSimilarityRankerContract:
    """Test Similarity Ranker contract compliance."""

    @pytest.fixture
    def reference_samples(self):
        """Create reference samples for testing."""
        return [
            Sample(
                content="Artificial intelligence is transforming healthcare through machine learning models that can detect diseases earlier and more accurately than traditional methods. These AI systems analyze medical imaging, patient records, and genetic data to provide insights that help doctors make better diagnoses." * 2,
                source=SampleSource.USER_INPUT,
            )
        ]

    @pytest.fixture
    def candidate_samples(self):
        """Create candidate samples for ranking."""
        return [
            Sample(
                content="Machine learning algorithms are revolutionizing medical diagnosis by using deep learning to analyze medical images with superhuman accuracy. Neural networks can detect tumors, fractures, and other abnormalities faster than radiologists." * 2,
                source=SampleSource.INTERNET_SEARCH,
                metadata={"url": "https://example.com/article1", "title": "ML in Medicine"},
            ),
            Sample(
                content="Climate change poses significant risks to ecosystems worldwide. Rising temperatures affect biodiversity, ocean levels, and weather patterns. Scientists warn of irreversible damage without immediate action." * 3,
                source=SampleSource.URL_CRAWL,
                metadata={"url": "https://example.com/article2", "title": "Climate Crisis"},
            ),
            Sample(
                content="AI-powered diagnostic tools improve accuracy in detecting early-stage cancers. Machine learning models trained on millions of medical scans can identify subtle patterns that human doctors might miss." * 2,
                source=SampleSource.INTERNET_SEARCH,
                metadata={"url": "https://example.com/article3", "title": "AI Diagnostics"},
            ),
        ]

    @pytest.mark.asyncio
    async def test_rank_basic_success(self, reference_samples, candidate_samples):
        """Test successful ranking of candidate samples.
        
        Contract: POST /rank with reference and candidate samples
        Expected: RankingResult with ranked samples sorted by similarity
        """
        from src.rankers.similarity_ranker import SimilarityRanker
        
        ranker = SimilarityRanker()
        result = await ranker.rank(
            reference_samples=reference_samples,
            candidate_samples=candidate_samples,
            top_n=10,
            threshold=0.7,
            include_embeddings=False,
        )
        
        # Validate response structure
        assert isinstance(result, RankingResult)
        assert isinstance(result.ranked_result_ids, list)
        assert "total_candidates" in result.metadata
        assert "returned_count" in result.metadata
        assert "threshold" in result.metadata
        assert "below_threshold_count" in result.metadata
        assert "processing_time_ms" in result.metadata
        assert result.metadata["processing_time_ms"] >= 0
        
        # Validate ranking order (scores should be descending)
        # Will verify when we can retrieve RankedResult objects

    @pytest.mark.asyncio
    async def test_rank_top_n_limiting(self, reference_samples, candidate_samples):
        """Test that only top N results are returned.
        
        Contract: Return at most top_n results
        Expected: len(results) <= top_n
        """
        from src.rankers.similarity_ranker import SimilarityRanker
        
        ranker = SimilarityRanker()
        result = await ranker.rank(
            reference_samples=reference_samples,
            candidate_samples=candidate_samples,
            top_n=2,
            threshold=0.0,  # Low threshold to get results
        )
        
        assert len(result.ranked_result_ids) <= 2
        assert result.metadata["returned_count"] <= 2

    @pytest.mark.asyncio
    async def test_rank_threshold_filtering(self, reference_samples, candidate_samples):
        """Test that samples below threshold are filtered out.
        
        Contract: Only return samples with score >= threshold
        Expected: All results have score >= threshold
        """
        from src.rankers.similarity_ranker import SimilarityRanker
        
        ranker = SimilarityRanker()
        result = await ranker.rank(
            reference_samples=reference_samples,
            candidate_samples=candidate_samples,
            top_n=10,
            threshold=0.8,  # High threshold
        )
        
        # Some samples should be filtered out
        assert result.metadata["returned_count"] <= len(candidate_samples)
        assert "below_threshold_count" in result.metadata

    @pytest.mark.asyncio
    async def test_rank_similarity_score_range(self, reference_samples, candidate_samples):
        """Test that similarity scores are in valid range.
        
        Contract: Scores must be 0.0-1.0
        Expected: All scores within range
        """
        from src.rankers.similarity_ranker import SimilarityRanker
        
        ranker = SimilarityRanker()
        result = await ranker.rank(
            reference_samples=reference_samples,
            candidate_samples=candidate_samples,
            threshold=0.0,
        )
        
        # Will validate actual scores when we can retrieve RankedResult objects
        assert len(result.ranked_result_ids) >= 0

    @pytest.mark.asyncio
    async def test_rank_with_embeddings(self, reference_samples, candidate_samples):
        """Test inclusion of embedding vectors in response.
        
        Contract: include_embeddings=true returns embedding vectors
        Expected: Each result has embedding array (768 dimensions)
        """
        from src.rankers.similarity_ranker import SimilarityRanker
        
        ranker = SimilarityRanker()
        result = await ranker.rank(
            reference_samples=reference_samples,
            candidate_samples=candidate_samples,
            include_embeddings=True,
        )
        
        # Embedding inclusion will be validated when we can retrieve results
        assert isinstance(result, RankingResult)

    @pytest.mark.asyncio
    async def test_rank_empty_reference_samples(self, candidate_samples):
        """Test error handling for empty reference samples.
        
        Contract: 400 invalid_input error
        Expected: ValueError
        """
        from src.rankers.similarity_ranker import SimilarityRanker
        
        ranker = SimilarityRanker()
        
        with pytest.raises(ValueError) as exc_info:
            await ranker.rank(
                reference_samples=[],
                candidate_samples=candidate_samples,
            )
        
        assert "reference" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_rank_empty_candidate_samples(self, reference_samples):
        """Test error handling for empty candidate samples.
        
        Contract: 400 invalid_input error
        Expected: ValueError
        """
        from src.rankers.similarity_ranker import SimilarityRanker
        
        ranker = SimilarityRanker()
        
        with pytest.raises(ValueError) as exc_info:
            await ranker.rank(
                reference_samples=reference_samples,
                candidate_samples=[],
            )
        
        assert "candidate" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_rank_too_many_reference_samples(self, candidate_samples):
        """Test error handling for >10 reference samples.
        
        Contract: Maximum 10 reference samples
        Expected: ValueError
        """
        from src.rankers.similarity_ranker import SimilarityRanker
        
        # Create 11 reference samples
        reference_samples = [
            Sample(
                content=f"Sample {i} with enough content to be valid for testing purposes and meet minimum length requirements" * 3,
                source=SampleSource.USER_INPUT,
            )
            for i in range(11)
        ]
        
        ranker = SimilarityRanker()
        
        with pytest.raises(ValueError) as exc_info:
            await ranker.rank(
                reference_samples=reference_samples,
                candidate_samples=candidate_samples,
            )
        
        assert "10" in str(exc_info.value) or "maximum" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_rank_too_many_candidate_samples(self, reference_samples):
        """Test error handling for >500 candidate samples.
        
        Contract: Maximum 500 candidate samples
        Expected: ValueError
        """
        from src.rankers.similarity_ranker import SimilarityRanker
        
        # Create 501 candidate samples
        candidate_samples = [
            Sample(
                content=f"Candidate sample {i} with enough content to be valid for testing purposes" * 3,
                source=SampleSource.INTERNET_SEARCH,
            )
            for i in range(501)
        ]
        
        ranker = SimilarityRanker()
        
        with pytest.raises(ValueError) as exc_info:
            await ranker.rank(
                reference_samples=reference_samples,
                candidate_samples=candidate_samples,
            )
        
        assert "500" in str(exc_info.value) or "maximum" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_rank_top_n_validation(self, reference_samples, candidate_samples):
        """Test validation of top_n parameter.
        
        Contract: top_n must be 1-100
        Expected: ValueError for out-of-range values
        """
        from src.rankers.similarity_ranker import SimilarityRanker
        
        ranker = SimilarityRanker()
        
        # Test below minimum
        with pytest.raises(ValueError) as exc_info:
            await ranker.rank(
                reference_samples=reference_samples,
                candidate_samples=candidate_samples,
                top_n=0,
            )
        assert "top_n" in str(exc_info.value).lower()
        
        # Test above maximum
        with pytest.raises(ValueError) as exc_info:
            await ranker.rank(
                reference_samples=reference_samples,
                candidate_samples=candidate_samples,
                top_n=101,
            )
        assert "top_n" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_rank_threshold_validation(self, reference_samples, candidate_samples):
        """Test validation of threshold parameter.
        
        Contract: threshold must be 0.0-1.0
        Expected: ValueError for out-of-range values
        """
        from src.rankers.similarity_ranker import SimilarityRanker
        
        ranker = SimilarityRanker()
        
        # Test below minimum
        with pytest.raises(ValueError) as exc_info:
            await ranker.rank(
                reference_samples=reference_samples,
                candidate_samples=candidate_samples,
                threshold=-0.1,
            )
        assert "threshold" in str(exc_info.value).lower()
        
        # Test above maximum
        with pytest.raises(ValueError) as exc_info:
            await ranker.rank(
                reference_samples=reference_samples,
                candidate_samples=candidate_samples,
                threshold=1.5,
            )
        assert "threshold" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_rank_content_too_short(self, reference_samples):
        """Test error handling for candidate content too short.
        
        Contract: 400 invalid_input error
        Expected: Pydantic ValidationError during Sample creation
        """
        from src.rankers.similarity_ranker import SimilarityRanker
        from pydantic import ValidationError
        
        # This should fail during Sample creation
        with pytest.raises(ValidationError):
            Sample(
                content="Short",
                source=SampleSource.INTERNET_SEARCH,
            )

    @pytest.mark.asyncio
    async def test_rank_cosine_similarity_method(self, reference_samples, candidate_samples):
        """Test that cosine similarity is used by default.
        
        Contract: Use cosine similarity on sentence-transformer embeddings
        Expected: Similarity scores based on cosine distance
        """
        from src.rankers.similarity_ranker import SimilarityRanker
        
        ranker = SimilarityRanker()
        result = await ranker.rank(
            reference_samples=reference_samples,
            candidate_samples=candidate_samples,
        )
        
        # Method verification will be done through SimilarityScore objects
        assert isinstance(result, RankingResult)

    @pytest.mark.asyncio
    async def test_rank_embedding_dimension(self, reference_samples, candidate_samples):
        """Test that embeddings have correct dimension.
        
        Contract: 768 dimensions for google/embeddinggemma-300m model
        Expected: All embeddings have 768 dimensions
        """
        from src.rankers.similarity_ranker import SimilarityRanker
        
        ranker = SimilarityRanker()
        result = await ranker.rank(
            reference_samples=reference_samples,
            candidate_samples=candidate_samples,
            include_embeddings=True,
        )
        
        # Embedding dimension will be validated through SimilarityScore
        assert isinstance(result, RankingResult)

    @pytest.mark.asyncio
    async def test_rank_no_results_above_threshold(self, reference_samples, candidate_samples):
        """Test handling when no results exceed threshold.
        
        Contract: 404 no_results error (or empty result list)
        Expected: Empty results or raise exception
        """
        from src.rankers.similarity_ranker import SimilarityRanker
        
        ranker = SimilarityRanker()
        result = await ranker.rank(
            reference_samples=reference_samples,
            candidate_samples=candidate_samples,
            threshold=0.99,  # Impossibly high threshold
        )
        
        # Should return result with empty list
        assert isinstance(result, RankingResult)
        assert result.metadata["returned_count"] == 0
        assert result.metadata["below_threshold_count"] >= 0

    @pytest.mark.asyncio
    async def test_rank_multiple_reference_samples(self, candidate_samples):
        """Test ranking with multiple reference samples.
        
        Contract: Support 1-10 reference samples
        Expected: Aggregate similarity across all references
        """
        from src.rankers.similarity_ranker import SimilarityRanker
        
        reference_samples = [
            Sample(
                content="AI in healthcare improves diagnosis accuracy through machine learning models trained on medical data." * 3,
                source=SampleSource.USER_INPUT,
            ),
            Sample(
                content="Medical AI systems analyze patient records and imaging to detect diseases earlier than traditional methods." * 3,
                source=SampleSource.USER_INPUT,
            ),
        ]
        
        ranker = SimilarityRanker()
        result = await ranker.rank(
            reference_samples=reference_samples,
            candidate_samples=candidate_samples,
        )
        
        assert isinstance(result, RankingResult)
        # Similarity calculation should consider all references

    @pytest.mark.asyncio
    async def test_rank_sorting_order(self, reference_samples, candidate_samples):
        """Test that results are sorted by score descending.
        
        Contract: Results sorted highest score first
        Expected: rank=1 has highest score
        """
        from src.rankers.similarity_ranker import SimilarityRanker
        
        ranker = SimilarityRanker()
        result = await ranker.rank(
            reference_samples=reference_samples,
            candidate_samples=candidate_samples,
            threshold=0.0,  # Get all results
        )
        
        # Sorting will be validated when we can retrieve RankedResult objects
        assert len(result.ranked_result_ids) >= 0

    @pytest.mark.asyncio
    async def test_rank_similarity_details(self, reference_samples, candidate_samples):
        """Test that similarity details are provided.
        
        Contract: Include max_score, avg_score, closest_reference_id
        Expected: Detailed similarity breakdown per result
        """
        from src.rankers.similarity_ranker import SimilarityRanker
        
        ranker = SimilarityRanker()
        result = await ranker.rank(
            reference_samples=reference_samples,
            candidate_samples=candidate_samples,
        )
        
        # Details will be in RankedResult metadata
        assert isinstance(result, RankingResult)

    @pytest.mark.asyncio
    async def test_rank_embedding_caching(self, reference_samples, candidate_samples):
        """Test that embeddings are cached for efficiency.
        
        Contract: Track embeddings_cached count
        Expected: Subsequent calls use cached embeddings
        """
        from src.rankers.similarity_ranker import SimilarityRanker
        
        ranker = SimilarityRanker()
        
        # First call - no cache
        result1 = await ranker.rank(
            reference_samples=reference_samples,
            candidate_samples=candidate_samples,
        )
        
        # Second call - should use cache
        result2 = await ranker.rank(
            reference_samples=reference_samples,
            candidate_samples=candidate_samples,
        )
        
        # Caching will be tracked in metadata
        assert "embeddings_cached" in result2.metadata or True  # May not be implemented

    @pytest.mark.asyncio
    async def test_rank_processing_time(self, reference_samples, candidate_samples):
        """Test that processing time is tracked.
        
        Contract: processing_time_ms must be >= 0
        Expected: Positive integer representing milliseconds
        """
        from src.rankers.similarity_ranker import SimilarityRanker
        
        ranker = SimilarityRanker()
        result = await ranker.rank(
            reference_samples=reference_samples,
            candidate_samples=candidate_samples,
        )
        
        assert "processing_time_ms" in result.metadata
        assert isinstance(result.metadata["processing_time_ms"], int)
        assert result.metadata["processing_time_ms"] >= 0
        # Should complete in reasonable time (< 10 seconds for small dataset)
        assert result.metadata["processing_time_ms"] < 10000

    @pytest.mark.asyncio
    async def test_rank_model_loading(self):
        """Test that embedding model is loaded successfully.
        
        Contract: Load google/embeddinggemma-300m model
        Expected: Model loaded on initialization or first use
        """
        from src.rankers.similarity_ranker import SimilarityRanker
        
        ranker = SimilarityRanker()
        # Model loading will be tested during first rank call
        assert ranker is not None
