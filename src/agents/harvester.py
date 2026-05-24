import asyncio
import json
from datetime import datetime
from typing import Any, Protocol

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field

from src.agents.crawler_session import CrawlerSession
from src.agents.state import AgentState, PendingSample
from src.config.llm_factory import build_harvester_llm_config
from src.config.models import HarvestConfig, WildConfig
from src.search.domain_filter import partition_urls_by_domain
from src.text.words import count_words, max_chars_for_words


def cycle_update(state: AgentState) -> dict:
    """Increment iteration once per explore→harvest pair."""
    if state.get("fresh_explore"):
        return {"iteration": state.get("iteration", 0) + 1, "fresh_explore": False}
    return {}


class TextSample(BaseModel):
    content: str = Field(description="Extracted text sample")
    category: str = Field(description="Brief category/label")


class ExtractedSamples(BaseModel):
    samples: list[TextSample] = Field(description="Extracted text samples")


class _CrawlerBackend(Protocol):
    async def arun(self, url: str, config: CrawlerRunConfig) -> Any: ...


def _cache_mode(harvest: HarvestConfig) -> CacheMode:
    if harvest.cache_mode == "enabled":
        return CacheMode.ENABLED
    return CacheMode.BYPASS


def create_extraction_strategy(themes: list[str], config: WildConfig) -> LLMExtractionStrategy:
    themes_str = ", ".join(themes) if themes else "the main topic"
    harvester_cfg = config.llms.harvester
    coll = config.collection
    min_w, max_w = coll.min_words, coll.max_words

    instruction = f"""Extract individual text samples from this content that are relevant to: {themes_str}

Rules:
- Each sample must be a self-contained excerpt (not navigation, headers, or boilerplate)
- Each sample must be between {min_w} and {max_w} words (inclusive)
- Extract as many non-overlapping samples as the page allows within that word range
- Focus on substantive text that matches the themes
- If no relevant content, return an empty samples list"""

    extra_args: dict = {}
    if harvester_cfg.temperature is not None:
        extra_args["temperature"] = harvester_cfg.temperature
    if harvester_cfg.max_tokens is not None:
        extra_args["max_tokens"] = harvester_cfg.max_tokens

    return LLMExtractionStrategy(
        llm_config=build_harvester_llm_config(harvester_cfg),
        schema=ExtractedSamples.model_json_schema(),
        extraction_type="schema",
        instruction=instruction,
        chunk_token_threshold=harvester_cfg.chunk_token_threshold,
        overlap_rate=harvester_cfg.overlap_rate,
        apply_chunking=True,
        input_format="markdown",
        extra_args=extra_args or None,
    )


def _word_bounds(config: WildConfig) -> tuple[int, int]:
    coll = config.collection
    return coll.min_words, coll.max_words


def _samples_from_result(
    result: Any,
    word_bounds: tuple[int, int],
    max_chars: int,
) -> list[PendingSample]:
    min_words, max_words = word_bounds
    extracted: list[PendingSample] = []
    if not result.success or not result.extracted_content:
        return extracted
    try:
        data = json.loads(result.extracted_content)
        samples_data = data.get("samples", []) if isinstance(data, dict) else data
        if not isinstance(samples_data, list):
            return extracted
        for sample in samples_data:
            content = sample.get("content", "") if isinstance(sample, dict) else str(sample)
            category = sample.get("category", "") if isinstance(sample, dict) else ""
            words = count_words(content)
            if (
                content
                and min_words <= words <= max_words
                and len(content) <= max_chars
            ):
                extracted.append({
                    "content": content,
                    "url": result.url,
                    "title": category or None,
                    "scraped_at": datetime.now().isoformat(),
                })
    except json.JSONDecodeError:
        pass
    return extracted


async def _crawl_seed_url(
    crawler: _CrawlerBackend,
    seed_url: str,
    crawler_config: CrawlerRunConfig,
    word_bounds: tuple[int, int],
    max_chars: int,
) -> tuple[list[PendingSample], list[str]]:
    visited: list[str] = []
    extracted: list[PendingSample] = []
    try:
        results = await crawler.arun(url=seed_url, config=crawler_config)
        if not isinstance(results, list):
            results = [results]
        for result in results:
            if not result.success:
                continue
            visited.append(result.url)
            extracted.extend(_samples_from_result(result, word_bounds, max_chars))
    except Exception:
        visited.append(seed_url)
    return extracted, visited


async def _harvest_seed_batch(
    crawler: _CrawlerBackend,
    seed_urls: list[str],
    visited_urls: list[str],
    crawler_config: CrawlerRunConfig,
    harvest_cfg: HarvestConfig,
    word_bounds: tuple[int, int],
    max_chars: int,
) -> tuple[list[PendingSample], list[str]]:
    to_crawl = [url for url in seed_urls if url not in visited_urls]
    if not to_crawl:
        return [], []

    semaphore = asyncio.Semaphore(harvest_cfg.max_concurrent_seeds)

    async def crawl_one(url: str) -> tuple[list[PendingSample], list[str]]:
        async with semaphore:
            return await _crawl_seed_url(
                crawler, url, crawler_config, word_bounds, max_chars
            )

    outcomes = await asyncio.gather(*(crawl_one(url) for url in to_crawl))

    extracted: list[PendingSample] = []
    newly_visited: list[str] = []
    for batch_samples, batch_visited in outcomes:
        extracted.extend(batch_samples)
        newly_visited.extend(batch_visited)
    return extracted, newly_visited


async def _run_harvest_pass(
    crawler: _CrawlerBackend,
    seed_urls: list[str],
    visited_urls: list[str],
    crawler_config: CrawlerRunConfig,
    harvest_cfg: HarvestConfig,
    wild_config: WildConfig,
) -> tuple[list[PendingSample], list[str]]:
    word_bounds = _word_bounds(wild_config)
    max_chars = max_chars_for_words(wild_config.collection.max_words)
    return await _harvest_seed_batch(
        crawler,
        seed_urls,
        visited_urls,
        crawler_config,
        harvest_cfg,
        word_bounds,
        max_chars,
    )


async def harvester_node(
    state: AgentState,
    *,
    wild_config: WildConfig,
    crawler_session: CrawlerSession | None = None,
) -> dict:
    pending_urls = list(state.get("pending_urls", []))
    visited_urls = list(state.get("visited_urls", []))

    if not pending_urls:
        if len(state.get("samples", [])) >= state.get("target_count", 10):
            return {"phase": "done"}
        if state.get("current_query_index", 0) < len(state.get("search_queries", [])):
            return {"phase": "explore"}
        return {"phase": "done"}

    harvest_cfg = wild_config.harvest
    coll = wild_config.collection
    batch_size = min(harvest_cfg.seed_batch_size, len(pending_urls))
    batch_urls = pending_urls[:batch_size]
    remaining_urls = pending_urls[batch_size:]

    accepted_urls, blocked_urls = partition_urls_by_domain(batch_urls, harvest_cfg)
    visited_urls.extend(blocked_urls)

    themes = state.get("extracted_themes", [])
    user_theme = state.get("theme", "")
    all_themes = themes + ([user_theme] if user_theme else [])

    deep_crawl = None
    if harvest_cfg.max_depth > 0:
        deep_crawl = BFSDeepCrawlStrategy(
            max_depth=harvest_cfg.max_depth,
            max_pages=harvest_cfg.max_pages,
            include_external=harvest_cfg.include_external,
        )

    crawler_config = CrawlerRunConfig(
        extraction_strategy=create_extraction_strategy(all_themes, wild_config),
        deep_crawl_strategy=deep_crawl,
        semaphore_count=harvest_cfg.semaphore_count,
        cache_mode=_cache_mode(harvest_cfg),
        word_count_threshold=min(coll.min_words, 10),
        excluded_tags=["nav", "footer", "header", "aside", "script", "style"],
    )

    extracted_samples: list[PendingSample] = []
    newly_visited: list[str] = []

    try:
        if crawler_session is not None:
            extracted_samples, newly_visited = await _run_harvest_pass(
                crawler_session,
                accepted_urls,
                visited_urls,
                crawler_config,
                harvest_cfg,
                wild_config,
            )
        else:
            async with AsyncWebCrawler() as crawler:
                extracted_samples, newly_visited = await _run_harvest_pass(
                    crawler,
                    accepted_urls,
                    visited_urls,
                    crawler_config,
                    harvest_cfg,
                    wild_config,
                )
    except Exception as e:
        return {
            **cycle_update(state),
            "pending_urls": remaining_urls,
            "visited_urls": visited_urls + newly_visited,
            "phase": "harvest" if remaining_urls else "explore",
            "error": f"Deep crawl failed: {str(e)}",
        }

    visited_urls.extend(newly_visited)

    if extracted_samples:
        return {
            **cycle_update(state),
            "pending_urls": remaining_urls,
            "visited_urls": visited_urls,
            "pending_sample": extracted_samples[0],
            "pending_scraped": extracted_samples[1:],
            "phase": "audit",
        }

    return {
        **cycle_update(state),
        "pending_urls": remaining_urls,
        "visited_urls": visited_urls,
        "phase": "harvest" if remaining_urls else "explore",
    }
