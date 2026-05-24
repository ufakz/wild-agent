from typing import Literal, TypedDict

CollectionMode = Literal["theme", "examples", "both"]


class Sample(TypedDict, total=False):
    content: str
    url: str
    title: str | None
    relevance_score: int | None
    similarity_score: float | None
    matched_reference_index: int | None
    themes: list[str]
    scraped_at: str


class PendingSample(TypedDict):
    content: str
    url: str
    title: str | None
    scraped_at: str


class AgentState(TypedDict, total=False):
    theme: str | None
    example_texts: list[str]
    target_count: int
    min_relevance: int
    similarity_threshold: float
    collection_mode: CollectionMode
    reference_embeddings: list[list[float]]
    embedding_model: str
    extracted_themes: list[str]
    search_queries: list[str]
    current_query_index: int
    pending_urls: list[str]
    visited_urls: list[str]
    pending_sample: PendingSample | None
    pending_scraped: list[PendingSample]
    samples: list[Sample]
    rejected_count: int
    last_rejection_reason: str | None
    phase: str
    iteration: int
    max_iterations: int
    fresh_explore: bool
    error: str | None


def derive_collection_mode(
    theme: str | None,
    examples: list[str] | None,
) -> CollectionMode:
    has_theme = bool(theme and theme.strip())
    has_examples = bool(examples)
    if has_theme and has_examples:
        return "both"
    if has_examples:
        return "examples"
    return "theme"
