from src.config.models import HarvestConfig
from src.search.domain_filter import (
    filter_urls_by_domain,
    partition_urls_by_domain,
    url_passes_domain_filter,
)


def test_blocked_domain_rejects_url():
    harvest = HarvestConfig(blocked_domains=["wikipedia.org"])
    assert not url_passes_domain_filter(
        "https://en.wikipedia.org/wiki/Rap",
        blocked_domains=harvest.blocked_domains,
        allowed_domains=harvest.allowed_domains,
    )


def test_allowed_domain_restricts_to_suffix():
    harvest = HarvestConfig(allowed_domains=["genius.com"])
    assert url_passes_domain_filter(
        "https://genius.com/songs/123",
        blocked_domains=harvest.blocked_domains,
        allowed_domains=harvest.allowed_domains,
    )
    assert not url_passes_domain_filter(
        "https://example.com/page",
        blocked_domains=harvest.blocked_domains,
        allowed_domains=harvest.allowed_domains,
    )


def test_filter_urls_by_domain():
    harvest = HarvestConfig(blocked_domains=["arxiv.org"])
    urls = [
        "https://arxiv.org/abs/123",
        "https://genius.com/lyrics/abc",
    ]
    assert filter_urls_by_domain(urls, harvest) == ["https://genius.com/lyrics/abc"]


def test_partition_urls_by_domain():
    harvest = HarvestConfig(blocked_domains=["sagepub.com"])
    accepted, rejected = partition_urls_by_domain(
        ["https://journals.sagepub.com/doi/1", "https://lyrics.com/x"],
        harvest,
    )
    assert rejected == ["https://journals.sagepub.com/doi/1"]
    assert accepted == ["https://lyrics.com/x"]
