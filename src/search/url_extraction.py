"""Extract and normalize URLs from LangChain messages and DuckDuckGo results."""

import re
from typing import Any

from langchain_core.messages import AIMessage

URL_PATTERN = re.compile(r"https?://[^\s\]\)\"'<>]+", re.IGNORECASE)


def normalize_url(url: str) -> str | None:
    url = url.rstrip(".,;:)\"'")
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return None


def dedupe_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def _urls_from_text(text: str) -> list[str]:
    return [u for m in URL_PATTERN.findall(text or "") if (u := normalize_url(m))]


def _urls_from_annotations(annotations: list[Any]) -> list[str]:
    urls: list[str] = []
    for ann in annotations:
        if isinstance(ann, dict):
            if ann.get("type") == "url_citation" and ann.get("url"):
                if normalized := normalize_url(str(ann["url"])):
                    urls.append(normalized)
        else:
            if getattr(ann, "type", None) == "url_citation":
                raw = getattr(ann, "url", None)
                if raw and (normalized := normalize_url(str(raw))):
                    urls.append(normalized)
    return urls


def _urls_from_web_search_block(block: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    action = block.get("action") or {}
    if isinstance(action, dict):
        for source in action.get("sources") or []:
            if isinstance(source, dict):
                raw = source.get("url")
            else:
                raw = getattr(source, "url", None)
            if raw and (normalized := normalize_url(str(raw))):
                urls.append(normalized)
    return urls


def _urls_from_content_block(block: Any) -> list[str]:
    urls: list[str] = []
    if isinstance(block, dict):
        block_type = block.get("type")
        if block_type == "web_search_call":
            urls.extend(_urls_from_web_search_block(block))
        urls.extend(_urls_from_annotations(block.get("annotations") or []))
        text = block.get("text") or block.get("content") or ""
        if isinstance(text, str):
            urls.extend(_urls_from_text(text))
    elif isinstance(block, str):
        urls.extend(_urls_from_text(block))
    else:
        urls.extend(_urls_from_annotations(getattr(block, "annotations", []) or []))
        text = getattr(block, "text", None) or getattr(block, "content", None)
        if isinstance(text, str):
            urls.extend(_urls_from_text(text))
    return urls


def extract_urls_from_ai_message(message: AIMessage) -> list[str]:
    """Parse URLs from a Responses API AIMessage (citations, web_search blocks, text)."""
    urls: list[str] = []
    content = message.content

    if isinstance(content, str):
        urls.extend(_urls_from_text(content))
    elif isinstance(content, list):
        for block in content:
            urls.extend(_urls_from_content_block(block))

    citations = message.additional_kwargs.get("citations", [])
    for item in citations or []:
        if isinstance(item, str):
            if normalized := normalize_url(item):
                urls.append(normalized)
            else:
                urls.extend(_urls_from_text(item))
        elif isinstance(item, dict):
            for key in ("url", "link", "href", "source_url"):
                if item.get(key) and (normalized := normalize_url(str(item[key]))):
                    urls.append(normalized)
                    break

    return dedupe_urls(urls)


def extract_urls_from_ddg_results(results: list[dict[str, str]]) -> list[str]:
    """Extract URLs from DuckDuckGo API wrapper result dicts."""
    urls: list[str] = []
    for row in results:
        link = row.get("link") or row.get("href") or row.get("url")
        if link and (normalized := normalize_url(str(link))):
            urls.append(normalized)
    return dedupe_urls(urls)
