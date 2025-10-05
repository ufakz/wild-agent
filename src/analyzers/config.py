"""Analyzer configuration and utilities."""

import os
import time
from typing import Any, Callable, TypeVar

import structlog

logger = structlog.get_logger()

T = TypeVar("T")


class AnalyzerConfig:
    """Configuration for LLM Analyzer."""

    def __init__(self):
        """Load configuration from environment variables."""
        
        # LLM Parameters
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2000"))
        self.timeout = float(os.getenv("LLM_TIMEOUT", "30.0"))
        
        # Retry Configuration
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
        self.initial_retry_delay = float(os.getenv("LLM_INITIAL_RETRY_DELAY", "1.0"))
        self.max_retry_delay = float(os.getenv("LLM_MAX_RETRY_DELAY", "60.0"))
        self.retry_backoff_factor = float(os.getenv("LLM_RETRY_BACKOFF_FACTOR", "2.0"))

    def validate(self) -> list[str]:
        """Validate configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not self.api_key:
            errors.append("XAI_API_KEY environment variable not set")
        
        if not 0.0 <= self.temperature <= 2.0:
            errors.append(f"LLM_TEMPERATURE must be 0.0-2.0, got {self.temperature}")
        
        if self.max_tokens < 1:
            errors.append(f"LLM_MAX_TOKENS must be positive, got {self.max_tokens}")
        
        if self.timeout < 1:
            errors.append(f"LLM_TIMEOUT must be >= 1, got {self.timeout}")
        
        if self.max_retries < 0:
            errors.append(f"LLM_MAX_RETRIES must be non-negative, got {self.max_retries}")
        
        return errors


async def retry_with_exponential_backoff(
    func: Callable[[], Any],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Retry async function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        backoff_factor: Multiplier for delay after each retry
        retryable_exceptions: Exceptions that trigger retry
        
    Returns:
        Result from successful function call
        
    Raises:
        Last exception if all retries exhausted
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except retryable_exceptions as e:
            last_exception = e
            
            if attempt >= max_retries:
                logger.error(
                    "retry_exhausted",
                    attempt=attempt,
                    max_retries=max_retries,
                    error=str(e),
                )
                raise
            
            logger.warning(
                "retry_attempt",
                attempt=attempt + 1,
                max_retries=max_retries,
                delay_seconds=delay,
                error=str(e),
            )
            
            import asyncio
            await asyncio.sleep(delay)
            
            # Exponential backoff
            delay = min(delay * backoff_factor, max_delay)
    
    # Should never reach here, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry loop completed without success or exception")


def get_default_config() -> AnalyzerConfig:
    """Get default analyzer configuration.
    
    Returns:
        AnalyzerConfig loaded from environment
    """
    config = AnalyzerConfig()
    
    # Log configuration (without sensitive data)
    logger.info(
        "analyzer_config_loaded",
        api_base=config.api_base,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        max_retries=config.max_retries,
        has_api_key=bool(config.api_key),
    )
    
    return config
