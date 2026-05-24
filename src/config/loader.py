import warnings
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from src.config.models import (
    CollectionConfig,
    EmbeddingsConfig,
    HarvestConfig,
    LLMsConfig,
    WEB_SEARCH_PROVIDERS,
    WildConfig,
    default_llms,
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_examples(examples: list[str], config_dir: Path) -> list[str]:
    resolved: list[str] = []
    for entry in examples:
        candidate = Path(entry)
        if not candidate.is_absolute():
            relative = config_dir / entry
            if relative.exists():
                candidate = relative
        if candidate.exists() and candidate.is_file():
            resolved.append(candidate.read_text(encoding="utf-8"))
        else:
            resolved.append(entry)
    return resolved


def _build_defaults_dict() -> dict[str, Any]:
    return {
        "harvest": HarvestConfig().model_dump(),
        "embeddings": EmbeddingsConfig().model_dump(),
        "llms": default_llms().model_dump(),
    }


def load_config(path: Path | str) -> WildConfig:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")

    merged = _deep_merge(_build_defaults_dict(), raw)

    try:
        collection = CollectionConfig.model_validate(merged.get("collection", {}))
    except ValidationError as e:
        raise ValueError(f"Invalid collection config: {e}") from e

    config_dir = config_path.parent
    if collection.examples:
        collection = collection.model_copy(
            update={"examples": _resolve_examples(collection.examples, config_dir)}
        )

    try:
        harvest = HarvestConfig.model_validate(merged.get("harvest", {}))
        embeddings = EmbeddingsConfig.model_validate(merged.get("embeddings", {}))
        llms = LLMsConfig.model_validate(merged.get("llms", {}))
    except ValidationError as e:
        raise ValueError(f"Invalid config: {e}") from e

    if llms.explorer.provider not in WEB_SEARCH_PROVIDERS:
        warnings.warn(
            f"explorer provider {llms.explorer.provider!r} does not support native "
            f"web_search ({', '.join(sorted(WEB_SEARCH_PROVIDERS))}); "
            "DuckDuckGo fallback will be used.",
            UserWarning,
            stacklevel=2,
        )

    return WildConfig(
        collection=collection,
        harvest=harvest,
        embeddings=embeddings,
        llms=llms,
        config_path=config_path,
    )


def default_config_path() -> Path:
    return Path("config/default.yaml")
