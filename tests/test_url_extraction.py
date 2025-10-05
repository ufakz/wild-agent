"""Unit tests for URL extraction (no network)."""

from langchain_core.messages import AIMessage

from src.search.url_extraction import (
    dedupe_urls,
    extract_urls_from_ai_message,
    extract_urls_from_ddg_results,
    normalize_url,
)


def test_normalize_url_strips_trailing_punctuation():
    assert normalize_url("https://example.com/path).") == "https://example.com/path"


def test_extract_urls_from_citation_annotations():
    message = AIMessage(
        content=[
            {
                "type": "text",
                "text": "See the guide for details.",
                "annotations": [
                    {
                        "type": "url_citation",
                        "url": "https://docs.python.org/3/library/asyncio.html",
                        "title": "asyncio",
                    }
                ],
            }
        ]
    )
    urls = extract_urls_from_ai_message(message)
    assert "https://docs.python.org/3/library/asyncio.html" in urls


def test_extract_urls_from_plain_text_content():
    message = AIMessage(
        content="Read https://realpython.com/async-io-python/ for more."
    )
    urls = extract_urls_from_ai_message(message)
    assert "https://realpython.com/async-io-python/" in urls


def test_extract_urls_from_ddg_results():
    results = [
        {"title": "A", "link": "https://a.example.com", "snippet": "..."},
        {"title": "B", "link": "https://b.example.com", "snippet": "..."},
    ]
    urls = extract_urls_from_ddg_results(results)
    assert urls == ["https://a.example.com", "https://b.example.com"]


def test_dedupe_urls():
    assert dedupe_urls([
        "https://x.com",
        "https://x.com",
        "https://y.com",
    ]) == ["https://x.com", "https://y.com"]
