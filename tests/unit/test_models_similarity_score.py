"""Tests for SimilarityScore model."""

import pytest
import numpy as np
from datetime import datetime
from pydantic import ValidationError
from src.models.similarity_score import SimilarityScore


class TestSimilarityScore:
    """Test SimilarityScore model."""

    def test_create_valid_similarity_score(self):
        """Test creating a valid similarity score."""
        embedding = np.random.rand(384).tolist()  # MiniLM dimension
        score = SimilarityScore(
            sample_id="sample123",
            reference_ids=["ref1", "ref2"],
            score=0.85,
            embedding=embedding,
        )
        assert score.sample_id == "sample123"
        assert score.reference_ids == ["ref1", "ref2"]
        assert score.score == 0.85
        assert len(score.embedding) == 384
        assert isinstance(score.calculated_at, datetime)

    def test_score_below_zero(self):
        """Test that score below 0.0 fails."""
        embedding = np.random.rand(384).tolist()
        with pytest.raises(ValidationError) as exc_info:
            SimilarityScore(
                sample_id="sample123",
                reference_ids=["ref1"],
                score=-0.1,
                embedding=embedding,
            )
        assert "score" in str(exc_info.value).lower()

    def test_score_above_one(self):
        """Test that score above 1.0 fails."""
        embedding = np.random.rand(384).tolist()
        with pytest.raises(ValidationError) as exc_info:
            SimilarityScore(
                sample_id="sample123",
                reference_ids=["ref1"],
                score=1.1,
                embedding=embedding,
            )
        assert "score" in str(exc_info.value).lower()

    def test_score_boundaries(self):
        """Test score boundary values."""
        embedding = np.random.rand(384).tolist()
        
        score1 = SimilarityScore(
            sample_id="sample123",
            reference_ids=["ref1"],
            score=0.0,
            embedding=embedding,
        )
        assert score1.score == 0.0

        score2 = SimilarityScore(
            sample_id="sample123",
            reference_ids=["ref1"],
            score=1.0,
            embedding=embedding,
        )
        assert score2.score == 1.0

    def test_embedding_wrong_dimension(self):
        """Test that embedding with wrong dimension fails."""
        embedding = np.random.rand(256).tolist()  # Wrong dimension
        with pytest.raises(ValidationError) as exc_info:
            SimilarityScore(
                sample_id="sample123",
                reference_ids=["ref1"],
                score=0.85,
                embedding=embedding,
            )
        assert "embedding" in str(exc_info.value).lower()
        assert "384" in str(exc_info.value)

    def test_embedding_correct_dimension(self):
        """Test that embedding with correct dimension succeeds."""
        embedding = np.random.rand(384).tolist()
        score = SimilarityScore(
            sample_id="sample123",
            reference_ids=["ref1"],
            score=0.85,
            embedding=embedding,
        )
        assert len(score.embedding) == 384

    def test_embedding_optional(self):
        """Test that embedding is optional."""
        score = SimilarityScore(
            sample_id="sample123",
            reference_ids=["ref1"],
            score=0.85,
        )
        assert score.embedding is None

    def test_sample_id_required(self):
        """Test that sample_id is required."""
        with pytest.raises(ValidationError):
            SimilarityScore(
                reference_ids=["ref1"],
                score=0.85,
            )

    def test_reference_ids_required(self):
        """Test that reference_ids is required."""
        with pytest.raises(ValidationError):
            SimilarityScore(
                sample_id="sample123",
                score=0.85,
            )

    def test_reference_ids_must_not_be_empty(self):
        """Test that reference_ids cannot be empty."""
        with pytest.raises(ValidationError) as exc_info:
            SimilarityScore(
                sample_id="sample123",
                reference_ids=[],
                score=0.85,
            )
        assert "reference_ids" in str(exc_info.value).lower()

    def test_reference_ids_multiple_references(self):
        """Test multiple reference IDs."""
        score = SimilarityScore(
            sample_id="sample123",
            reference_ids=["ref1", "ref2", "ref3"],
            score=0.85,
        )
        assert len(score.reference_ids) == 3

    def test_score_required(self):
        """Test that score is required."""
        with pytest.raises(ValidationError):
            SimilarityScore(
                sample_id="sample123",
                reference_ids=["ref1"],
            )

    def test_calculated_at_auto_populated(self):
        """Test that calculated_at is automatically populated."""
        score = SimilarityScore(
            sample_id="sample123",
            reference_ids=["ref1"],
            score=0.85,
        )
        assert isinstance(score.calculated_at, datetime)

    def test_calculated_at_can_be_provided(self):
        """Test that calculated_at can be provided."""
        now = datetime.now()
        score = SimilarityScore(
            sample_id="sample123",
            reference_ids=["ref1"],
            score=0.85,
            calculated_at=now,
        )
        assert score.calculated_at == now

    def test_embedding_with_integers(self):
        """Test that embedding can contain integer values."""
        embedding = [0] * 384
        score = SimilarityScore(
            sample_id="sample123",
            reference_ids=["ref1"],
            score=0.85,
            embedding=embedding,
        )
        assert len(score.embedding) == 384

    def test_embedding_with_mixed_types(self):
        """Test that embedding with mixed int/float works."""
        embedding = [0.5] * 192 + [1] * 192
        score = SimilarityScore(
            sample_id="sample123",
            reference_ids=["ref1"],
            score=0.85,
            embedding=embedding,
        )
        assert len(score.embedding) == 384

    def test_model_serialization(self):
        """Test that model can be serialized to dict."""
        embedding = np.random.rand(384).tolist()
        score = SimilarityScore(
            sample_id="sample123",
            reference_ids=["ref1", "ref2"],
            score=0.85,
            embedding=embedding,
        )
        data = score.model_dump()
        assert "sample_id" in data
        assert "reference_ids" in data
        assert "score" in data
        assert "embedding" in data
        assert "calculated_at" in data
        assert len(data["embedding"]) == 384

    def test_different_samples_different_embeddings(self):
        """Test that different samples can have different embeddings."""
        embedding1 = np.random.rand(384).tolist()
        embedding2 = np.random.rand(384).tolist()
        
        score1 = SimilarityScore(
            sample_id="sample1",
            reference_ids=["ref1"],
            score=0.8,
            embedding=embedding1,
        )
        score2 = SimilarityScore(
            sample_id="sample2",
            reference_ids=["ref1"],
            score=0.9,
            embedding=embedding2,
        )
        
        assert score1.sample_id != score2.sample_id
        assert score1.embedding != score2.embedding
