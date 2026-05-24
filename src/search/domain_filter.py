from urllib.parse import urlparse

from src.config.models import HarvestConfig


def _hostname(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _domain_matches(host: str, domain: str) -> bool:
    needle = domain.lower().lstrip(".")
    return host == needle or host.endswith(f".{needle}")


def url_passes_domain_filter(
    url: str,
    *,
    blocked_domains: list[str],
    allowed_domains: list[str],
) -> bool:
    host = _hostname(url)
    if not host:
        return False
    for blocked in blocked_domains:
        if _domain_matches(host, blocked):
            return False
    if allowed_domains:
        return any(_domain_matches(host, allowed) for allowed in allowed_domains)
    return True


def filter_urls_by_domain(urls: list[str], harvest: HarvestConfig) -> list[str]:
    return [
        url
        for url in urls
        if url_passes_domain_filter(
            url,
            blocked_domains=harvest.blocked_domains,
            allowed_domains=harvest.allowed_domains,
        )
    ]


def partition_urls_by_domain(
    urls: list[str], harvest: HarvestConfig
) -> tuple[list[str], list[str]]:
    accepted: list[str] = []
    rejected: list[str] = []
    for url in urls:
        if url_passes_domain_filter(
            url,
            blocked_domains=harvest.blocked_domains,
            allowed_domains=harvest.allowed_domains,
        ):
            accepted.append(url)
        else:
            rejected.append(url)
    return accepted, rejected
