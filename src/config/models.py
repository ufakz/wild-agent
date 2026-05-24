from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

CRAWL4AI_PROVIDERS = frozenset(
    {"xai", "openai", "anthropic", "gemini", "groq", "ollama"}
)

WEB_SEARCH_PROVIDERS = frozenset({"openai", "xai"})


class CollectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme: str | None = None
    examples: list[str] = Field(default_factory=list)
    target_count: int = 10
    min_relevance: int = Field(default=7, ge=0, le=10)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_iterations: int = Field(
        default=50,
        ge=1,
        description=(
            "Maximum explore→harvest pairs before stopping new searches/crawls. "
            "Auditing does not count."
        ),
    )
    output: str | None = None

    @model_validator(mode="after")
    def require_theme_or_examples(self) -> "CollectionConfig":
        if not self.theme and not self.examples:
            raise ValueError("collection must include either theme or examples")
        return self


class HarvestConfig(BaseModel):
    """Limits crawl breadth and LLM extraction concurrency per harvest pass."""

    model_config = ConfigDict(extra="forbid")

    seed_batch_size: int = Field(default=1, ge=1, le=10)
    max_depth: int = Field(
        default=0,
        ge=0,
        le=5,
        description="0 = seed URL only; >0 enables BFS link following",
    )
    max_pages: int = Field(default=1, ge=1, le=50)
    semaphore_count: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Max concurrent page fetches/extractions within one crawl",
    )
    include_external: bool = False


class EmbeddingsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "all-MiniLM-L6-v2"


class ChatLLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    api_key_env: str
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float | None = None
    base_url: str | None = None

    def chat_parameters(self) -> dict:
        params: dict = {}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.timeout is not None:
            params["timeout"] = self.timeout
        if self.base_url is not None:
            params["base_url"] = self.base_url
        return params


class HarvesterLLMConfig(ChatLLMConfig):
    chunk_token_threshold: int = 4000
    overlap_rate: float = 0.1


class ExplorerLLMConfig(ChatLLMConfig):
    max_results: int = Field(default=10, ge=1, le=50)


class LLMsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    planner: ChatLLMConfig
    auditor: ChatLLMConfig
    harvester: HarvesterLLMConfig
    explorer: ExplorerLLMConfig


class WildConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection: CollectionConfig
    harvest: HarvestConfig = Field(default_factory=HarvestConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    llms: LLMsConfig
    config_path: Path | None = Field(default=None, exclude=True)


def default_llms() -> LLMsConfig:
    return LLMsConfig(
        planner=ChatLLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
            temperature=0.3,
        ),
        auditor=ChatLLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
            temperature=0.1,
        ),
        harvester=HarvesterLLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
            temperature=0.3,
            max_tokens=4000,
        ),
        explorer=ExplorerLLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
            temperature=0.2,
            max_results=10,
        ),
    )
