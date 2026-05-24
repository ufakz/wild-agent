import json
from datetime import datetime
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

from src.agents.state import AgentState, PendingSample
from src.config.llm_factory import build_harvester_llm_config
from src.config.models import WildConfig


def cycle_update(state: AgentState) -> dict:
    """Increment iteration once per explore→harvest pair."""
    if state.get("fresh_explore"):
        return {"iteration": state.get("iteration", 0) + 1, "fresh_explore": False}
    return {}


class TextSample(BaseModel):
    content: str = Field(description="Extracted text (128-1024 tokens)")
    category: str = Field(description="Brief category/label")


class ExtractedSamples(BaseModel):
    samples: list[TextSample] = Field(description="Extracted text samples")


def create_extraction_strategy(themes: list[str], config: WildConfig) -> LLMExtractionStrategy:
    themes_str = ", ".join(themes) if themes else "the main topic"
    harvester_cfg = config.llms.harvester
    instruction = f"""Extract individual text samples from this content that are relevant to: {themes_str}

Rules:
- Each sample should be a self-contained piece of text (story, testimony, article excerpt, forum post)
- Minimum 100 words, maximum 500 words per sample
- Extract actual content, not navigation or headers
- Focus on substantive text that matches the themes
- Extract AS MANY relevant samples as possible from the content
- If no relevant content, return empty samples list"""

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


async def harvester_node(state: AgentState, *, wild_config: WildConfig) -> dict:
    pending_urls = list(state.get("pending_urls", []))
    visited_urls = list(state.get("visited_urls", []))

    if not pending_urls:
        if len(state.get("samples", [])) >= state.get("target_count", 10):
            return {"phase": "done"}
        if state.get("current_query_index", 0) < len(state.get("search_queries", [])):
            return {"phase": "explore"}
        return {"phase": "done"}

    harvest_cfg = wild_config.harvest
    batch_size = min(harvest_cfg.seed_batch_size, len(pending_urls))
    seed_urls = pending_urls[:batch_size]
    remaining_urls = pending_urls[batch_size:]

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
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=100,
        excluded_tags=["nav", "footer", "header", "aside", "script", "style"],
    )

    extracted_samples: list[PendingSample] = []
    newly_visited: list[str] = []

    try:
        async with AsyncWebCrawler() as crawler:
            for seed_url in seed_urls:
                if seed_url in visited_urls:
                    continue
                try:
                    results = await crawler.arun(url=seed_url, config=crawler_config)
                    if not isinstance(results, list):
                        results = [results]
                    for result in results:
                        if not result.success:
                            continue
                        newly_visited.append(result.url)
                        if not result.extracted_content:
                            continue
                        try:
                            data = json.loads(result.extracted_content)
                            samples_data = (
                                data.get("samples", []) if isinstance(data, dict) else data
                            )
                            if not isinstance(samples_data, list):
                                continue
                            for sample in samples_data:
                                content = (
                                    sample.get("content", "")
                                    if isinstance(sample, dict)
                                    else str(sample)
                                )
                                category = (
                                    sample.get("category", "")
                                    if isinstance(sample, dict)
                                    else ""
                                )
                                if content and 100 <= len(content) <= 4096:
                                    extracted_samples.append({
                                        "content": content,
                                        "url": result.url,
                                        "title": category or None,
                                        "scraped_at": datetime.now().isoformat(),
                                    })
                        except json.JSONDecodeError:
                            pass
                except Exception:
                    newly_visited.append(seed_url)
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
