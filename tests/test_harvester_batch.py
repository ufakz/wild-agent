"""Unit tests for harvester batching and parallel crawl (mocked crawler)."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.harvester import _harvest_seed_batch
from src.config.models import (
    CollectionConfig,
    EmbeddingsConfig,
    HarvestConfig,
    WildConfig,
    default_llms,
)


def _mock_result(url: str, word_count: int = 60):
    content = " ".join(["word"] * word_count)
    result = MagicMock()
    result.success = True
    result.url = url
    result.extracted_content = json.dumps({
        "samples": [{"content": content, "category": "test"}],
    })
    return result


@pytest.mark.asyncio
async def test_harvest_seed_batch_calls_all_urls():
    crawler = AsyncMock()
    crawler.arun = AsyncMock(
        side_effect=[
            _mock_result("https://a.com"),
            _mock_result("https://b.com"),
            _mock_result("https://c.com"),
        ]
    )
    harvest = HarvestConfig(seed_batch_size=3, max_concurrent_seeds=3)

    from crawl4ai import CrawlerRunConfig

    extracted, visited = await _harvest_seed_batch(
        crawler,
        ["https://a.com", "https://b.com", "https://c.com"],
        [],
        CrawlerRunConfig(),
        harvest,
        word_bounds=(1, 500),
        max_chars=4096,
    )
    assert crawler.arun.await_count == 3
    assert len(visited) == 3
    assert len(extracted) == 3


@pytest.mark.asyncio
async def test_harvest_seed_batch_respects_concurrency_limit():
    in_flight = 0
    max_seen = 0
    lock = asyncio.Lock()

    async def slow_arun(url: str, config):
        nonlocal in_flight, max_seen
        async with lock:
            in_flight += 1
            max_seen = max(max_seen, in_flight)
        await asyncio.sleep(0.05)
        async with lock:
            in_flight -= 1
        return _mock_result(url)

    crawler = AsyncMock()
    crawler.arun = slow_arun
    harvest = HarvestConfig(max_concurrent_seeds=2)

    from crawl4ai import CrawlerRunConfig

    urls = [f"https://site{i}.com" for i in range(4)]
    await _harvest_seed_batch(
        crawler, urls, [], CrawlerRunConfig(), harvest,
        word_bounds=(1, 500), max_chars=4096,
    )
    assert max_seen <= 2


@pytest.mark.asyncio
async def test_harvester_node_batches_urls(monkeypatch):
    from src.agents import harvester as harvester_mod

    calls: list[str] = []

    class FakeCrawler:
        async def arun(self, url: str, config):
            calls.append(url)
            return _mock_result(url)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(harvester_mod, "AsyncWebCrawler", lambda: FakeCrawler())

    harvest_cfg = HarvestConfig(seed_batch_size=3, max_concurrent_seeds=3)
    wild_config = WildConfig(
        collection=CollectionConfig(theme="test theme", min_words=1, max_words=500),
        harvest=harvest_cfg,
        embeddings=EmbeddingsConfig(),
        llms=default_llms(),
    )
    state = {
        "pending_urls": [f"https://u{i}.com" for i in range(5)],
        "visited_urls": [],
        "extracted_themes": ["topic"],
        "samples": [],
        "target_count": 100,
        "current_query_index": 0,
        "search_queries": [],
        "fresh_explore": False,
        "iteration": 0,
    }

    result = await harvester_mod.harvester_node(state, wild_config=wild_config)
    assert len(calls) == 3
    assert len(result["pending_urls"]) == 2
    assert result["phase"] == "audit"
