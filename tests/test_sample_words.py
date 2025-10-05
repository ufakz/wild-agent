import json

import pytest
from pydantic import ValidationError

from src.agents.harvester import _samples_from_result
from src.config.models import CollectionConfig
from src.text.words import count_words, max_chars_for_words


def test_count_words():
    assert count_words("one two three") == 3
    assert count_words("") == 0


def test_max_chars_for_words():
    assert max_chars_for_words(500) == 4096
    assert max_chars_for_words(10) == 150


def test_samples_from_result_word_bounds():
    words = " ".join(["word"] * 25)
    result = type("R", (), {
        "success": True,
        "url": "https://example.com",
        "extracted_content": json.dumps({
            "samples": [{"content": words, "category": "x"}],
        }),
    })()
    accepted = _samples_from_result(result, (20, 40), max_chars=4096)
    assert len(accepted) == 1

    short = " ".join(["word"] * 5)
    result.extracted_content = json.dumps({
        "samples": [{"content": short, "category": "x"}],
    })
    assert _samples_from_result(result, (20, 40), max_chars=4096) == []


def test_collection_min_words_gt_max_words_raises():
    with pytest.raises(ValidationError, match="min_words"):
        CollectionConfig(theme="x", min_words=100, max_words=50)


def test_collection_words_defaults():
    coll = CollectionConfig(theme="test")
    assert coll.min_words == 20
    assert coll.max_words == 100
