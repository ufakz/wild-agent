"""Tests for Theme model."""

import pytest
from pydantic import ValidationError
from src.models.theme import Theme


class TestTheme:
    """Test Theme model."""

    def test_create_valid_theme(self):
        """Test creating a valid theme."""
        theme = Theme(
            name="Machine Learning",
            description="Topics related to machine learning algorithms and applications",
            keywords=["neural networks", "deep learning", "supervised learning"],
            confidence=0.85,
        )
        assert theme.id is not None
        assert theme.name == "Machine Learning"
        assert len(theme.keywords) == 3
        assert theme.confidence == 0.85
        assert theme.sample_ids == []

    def test_name_too_short(self):
        """Test that name shorter than 5 chars fails."""
        with pytest.raises(ValidationError) as exc_info:
            Theme(
                name="AI",  # Only 2 chars
                description="Topics related to artificial intelligence",
                keywords=["ai", "ml", "algorithms"],
                confidence=0.8,
            )
        assert "name" in str(exc_info.value).lower()

    def test_name_too_long(self):
        """Test that name longer than 100 chars fails."""
        with pytest.raises(ValidationError) as exc_info:
            Theme(
                name="A" * 101,
                description="Topics related to artificial intelligence",
                keywords=["ai", "ml", "algorithms"],
                confidence=0.8,
            )
        assert "name" in str(exc_info.value).lower()

    def test_description_too_short(self):
        """Test that description shorter than 20 chars fails."""
        with pytest.raises(ValidationError) as exc_info:
            Theme(
                name="Machine Learning",
                description="Short",  # Only 5 chars
                keywords=["ai", "ml", "algorithms"],
                confidence=0.8,
            )
        assert "description" in str(exc_info.value).lower()

    def test_description_too_long(self):
        """Test that description longer than 1000 chars fails."""
        with pytest.raises(ValidationError) as exc_info:
            Theme(
                name="Machine Learning",
                description="A" * 1001,
                keywords=["ai", "ml", "algorithms"],
                confidence=0.8,
            )
        assert "description" in str(exc_info.value).lower()

    def test_keywords_too_few(self):
        """Test that fewer than 3 keywords fails."""
        with pytest.raises(ValidationError) as exc_info:
            Theme(
                name="Machine Learning",
                description="Topics related to machine learning",
                keywords=["ai", "ml"],  # Only 2 keywords
                confidence=0.8,
            )
        assert "keywords" in str(exc_info.value).lower()

    def test_keywords_too_many(self):
        """Test that more than 20 keywords fails."""
        with pytest.raises(ValidationError) as exc_info:
            Theme(
                name="Machine Learning",
                description="Topics related to machine learning",
                keywords=[f"keyword{i}" for i in range(21)],  # 21 keywords
                confidence=0.8,
            )
        assert "keywords" in str(exc_info.value).lower()

    def test_keyword_too_short(self):
        """Test that keyword shorter than 2 chars fails."""
        with pytest.raises(ValidationError) as exc_info:
            Theme(
                name="Machine Learning",
                description="Topics related to machine learning",
                keywords=["a", "ml", "algorithms"],  # First keyword only 1 char
                confidence=0.8,
            )
        assert "keyword" in str(exc_info.value).lower()

    def test_keyword_too_long(self):
        """Test that keyword longer than 50 chars fails."""
        with pytest.raises(ValidationError) as exc_info:
            Theme(
                name="Machine Learning",
                description="Topics related to machine learning",
                keywords=["A" * 51, "ml", "algorithms"],  # First keyword 51 chars
                confidence=0.8,
            )
        assert "keyword" in str(exc_info.value).lower()

    def test_confidence_below_zero(self):
        """Test that confidence below 0.0 fails."""
        with pytest.raises(ValidationError) as exc_info:
            Theme(
                name="Machine Learning",
                description="Topics related to machine learning",
                keywords=["ai", "ml", "algorithms"],
                confidence=-0.1,
            )
        assert "confidence" in str(exc_info.value).lower()

    def test_confidence_above_one(self):
        """Test that confidence above 1.0 fails."""
        with pytest.raises(ValidationError) as exc_info:
            Theme(
                name="Machine Learning",
                description="Topics related to machine learning",
                keywords=["ai", "ml", "algorithms"],
                confidence=1.1,
            )
        assert "confidence" in str(exc_info.value).lower()

    def test_confidence_boundaries(self):
        """Test that confidence at 0.0 and 1.0 is valid."""
        theme1 = Theme(
            name="Machine Learning",
            description="Topics related to machine learning",
            keywords=["ai", "ml", "algorithms"],
            confidence=0.0,
        )
        assert theme1.confidence == 0.0

        theme2 = Theme(
            name="Machine Learning",
            description="Topics related to machine learning",
            keywords=["ai", "ml", "algorithms"],
            confidence=1.0,
        )
        assert theme2.confidence == 1.0

    def test_sample_ids_default_empty(self):
        """Test that sample_ids defaults to empty list."""
        theme = Theme(
            name="Machine Learning",
            description="Topics related to machine learning",
            keywords=["ai", "ml", "algorithms"],
            confidence=0.8,
        )
        assert theme.sample_ids == []

    def test_sample_ids_can_be_provided(self):
        """Test that sample_ids can be provided."""
        sample_ids = ["id1", "id2", "id3"]
        theme = Theme(
            name="Machine Learning",
            description="Topics related to machine learning",
            keywords=["ai", "ml", "algorithms"],
            confidence=0.8,
            sample_ids=sample_ids,
        )
        assert theme.sample_ids == sample_ids

    def test_id_auto_generated(self):
        """Test that ID is automatically generated."""
        theme = Theme(
            name="Machine Learning",
            description="Topics related to machine learning",
            keywords=["ai", "ml", "algorithms"],
            confidence=0.8,
        )
        assert theme.id is not None
        assert len(theme.id) == 36  # UUID4 format

    def test_different_themes_have_different_ids(self):
        """Test that different themes get different IDs."""
        theme1 = Theme(
            name="Machine Learning",
            description="Topics related to machine learning",
            keywords=["ai", "ml", "algorithms"],
            confidence=0.8,
        )
        theme2 = Theme(
            name="Web Development",
            description="Topics related to web development",
            keywords=["html", "css", "javascript"],
            confidence=0.9,
        )
        assert theme1.id != theme2.id

    def test_keywords_boundary_count(self):
        """Test boundary cases for keyword count."""
        # Exactly 3 keywords (minimum)
        theme1 = Theme(
            name="Machine Learning",
            description="Topics related to machine learning",
            keywords=["ai", "ml", "algorithms"],
            confidence=0.8,
        )
        assert len(theme1.keywords) == 3

        # Exactly 20 keywords (maximum)
        theme2 = Theme(
            name="Machine Learning",
            description="Topics related to machine learning",
            keywords=[f"keyword{i}" for i in range(20)],
            confidence=0.8,
        )
        assert len(theme2.keywords) == 20

    def test_model_serialization(self):
        """Test that model can be serialized to dict."""
        theme = Theme(
            name="Machine Learning",
            description="Topics related to machine learning",
            keywords=["ai", "ml", "algorithms"],
            confidence=0.8,
            sample_ids=["id1", "id2"],
        )
        data = theme.model_dump()
        assert "id" in data
        assert "name" in data
        assert "description" in data
        assert "keywords" in data
        assert "confidence" in data
        assert "sample_ids" in data
