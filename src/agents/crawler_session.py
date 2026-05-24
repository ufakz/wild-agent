from __future__ import annotations

from typing import Any

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig


class CrawlerSession:
    """Reuses one AsyncWebCrawler for an entire collection run."""

    def __init__(self) -> None:
        self._crawler: AsyncWebCrawler | None = None

    async def __aenter__(self) -> CrawlerSession:
        self._crawler = AsyncWebCrawler()
        await self._crawler.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._crawler is not None:
            await self._crawler.__aexit__(*exc)
            self._crawler = None

    async def arun(self, url: str, config: CrawlerRunConfig) -> Any:
        if self._crawler is None:
            raise RuntimeError("CrawlerSession is not active; use async with CrawlerSession()")
        return await self._crawler.arun(url=url, config=config)
