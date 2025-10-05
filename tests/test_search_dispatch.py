"""Tests for explorer search dispatch (mocked, no network)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.config.llm_factory import search_urls
from src.config.models import ExplorerLLMConfig


@pytest.mark.asyncio
async def test_search_urls_uses_web_search_for_openai():
    cfg = ExplorerLLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        max_results=5,
    )
    with patch(
        "src.config.llm_factory._search_via_web_search_tool",
        new_callable=AsyncMock,
        return_value=["https://example.com"],
    ) as mock_primary:
        urls = await search_urls("python asyncio", ["asyncio"], cfg)
    mock_primary.assert_awaited_once()
    assert urls == ["https://example.com"]


@pytest.mark.asyncio
async def test_search_urls_uses_duckduckgo_fallback_for_anthropic():
    cfg = ExplorerLLMConfig(
        provider="anthropic",
        model="claude-3-5-haiku-latest",
        api_key_env="ANTHROPIC_API_KEY",
        max_results=3,
    )
    with patch(
        "src.config.llm_factory._search_via_duckduckgo",
        new_callable=AsyncMock,
        return_value=["https://fallback.example.com"],
    ) as mock_fallback:
        urls = await search_urls("python asyncio", [], cfg)
    mock_fallback.assert_awaited_once()
    assert urls == ["https://fallback.example.com"]
