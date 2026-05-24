import asyncio
import logging
import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from crawl4ai import LLMConfig
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from src.config.models import (
    CRAWL4AI_PROVIDERS,
    ChatLLMConfig,
    ExplorerLLMConfig,
    HarvesterLLMConfig,
    WEB_SEARCH_PROVIDERS,
)
from src.search.url_extraction import extract_urls_from_ai_message, extract_urls_from_ddg_results

logger = logging.getLogger(__name__)

_fallback_warned = False


def get_api_key(api_key_env: str) -> str:
    value = os.environ.get(api_key_env)
    if not value:
        raise ValueError(
            f"Environment variable {api_key_env!r} is not set. "
            "Add it to your environment or .env file."
        )
    return value


def create_structured_chat_model(cfg: ChatLLMConfig) -> BaseChatModel:
    api_key = get_api_key(cfg.api_key_env)
    model_id = f"{cfg.provider}:{cfg.model}"
    return init_chat_model(model_id, api_key=api_key, **cfg.chat_parameters())


def _responses_api_kwargs(cfg: ExplorerLLMConfig) -> dict:
    if cfg.provider not in WEB_SEARCH_PROVIDERS:
        return {}
    return {
        "use_responses_api": True,
        "output_version": "responses/v1",
    }


def create_web_search_model(cfg: ExplorerLLMConfig) -> Runnable:
    """LangChain chat model with hosted web_search tool (Responses API)."""
    api_key = get_api_key(cfg.api_key_env)
    model_id = f"{cfg.provider}:{cfg.model}"
    params = {**cfg.chat_parameters(), **_responses_api_kwargs(cfg)}
    model = init_chat_model(model_id, api_key=api_key, **params)
    return model.bind_tools([{"type": "web_search"}])


def _build_search_prompt(query: str, themes: list[str], max_results: int) -> str:
    themes_str = ", ".join(themes) if themes else "general"
    return (
        f"Search the web for pages with substantial text content matching this query.\n"
        f"Query: {query}\n"
        f"Themes: {themes_str}\n"
        f"Prefer articles, documentation, and blog posts. "
        f"Return up to {max_results} useful URLs."
    )


async def _search_via_web_search_tool(
    query: str,
    themes: list[str],
    cfg: ExplorerLLMConfig,
) -> list[str]:
    model = create_web_search_model(cfg)
    prompt = _build_search_prompt(query, themes, cfg.max_results)
    response = await model.ainvoke([HumanMessage(content=prompt)])
    urls = extract_urls_from_ai_message(response)
    return urls[: cfg.max_results]


async def _search_via_duckduckgo(query: str, cfg: ExplorerLLMConfig) -> list[str]:
    global _fallback_warned
    if not _fallback_warned:
        logger.warning(
            "explorer using DuckDuckGo fallback (provider %r has no native web_search)",
            cfg.provider,
        )
        _fallback_warned = True

    wrapper = DuckDuckGoSearchAPIWrapper(max_results=cfg.max_results)
    results = await asyncio.to_thread(
        wrapper.results,
        query,
        cfg.max_results,
    )
    return extract_urls_from_ddg_results(results)[: cfg.max_results]


async def search_urls(
    query: str,
    themes: list[str],
    cfg: ExplorerLLMConfig,
) -> list[str]:
    """Discover URLs via provider web_search or DuckDuckGo fallback."""
    if cfg.provider in WEB_SEARCH_PROVIDERS:
        return await _search_via_web_search_tool(query, themes, cfg)
    return await _search_via_duckduckgo(query, cfg)


def build_harvester_llm_config(cfg: HarvesterLLMConfig) -> LLMConfig:
    if cfg.provider not in CRAWL4AI_PROVIDERS:
        supported = ", ".join(sorted(CRAWL4AI_PROVIDERS))
        raise ValueError(
            f"Harvester provider {cfg.provider!r} is not supported by crawl4ai. "
            f"Supported: {supported}"
        )
    return LLMConfig(
        provider=f"{cfg.provider}/{cfg.model}",
        api_token=get_api_key(cfg.api_key_env),
    )
