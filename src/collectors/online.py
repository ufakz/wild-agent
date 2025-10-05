"""Online search collector using LLM."""

import json
import os
import time
from typing import Any

import structlog
from xai_sdk import Client
from xai_sdk.chat import system, user
from xai_sdk.search import SearchParameters

from src.models import Collection, CollectionMethod, Sample, SampleSource

logger = structlog.get_logger()


class OnlineCollector:
    """Collects samples using online search via LLM."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "grok-4-fast-non-reasoning",
        max_results: int = 100,
        timeout: float = 600.0,
    ):
        """Initialize online collector.
        
        Args:
            api_key: API key for LLM service
            api_base: Base URL for API (ignored, kept for compatibility)
            model: Model name to use
            max_results: Maximum samples to collect per query
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided - online collection will fail")
        
        self.model = model
        self.max_results = max_results
        self.timeout = timeout
        
        self.client = Client(api_key=self.api_key, timeout=timeout)

    async def collect(
        self,
        query_context: str,
        max_samples: int = 10,
    ) -> tuple[Collection, list[Sample]]:
        """Collect samples using online search.
        
        Args:
            query_context: Search context/themes to find similar content
            max_samples: Maximum samples to collect
            
        Returns:
            Tuple of (Collection with sample metadata, list of Sample objects)
            
        Raises:
            ValueError: If query_context is invalid
            RuntimeError: If LLM API call fails
        """
        start_time = time.time()
        
        # Validate input
        if not query_context or len(query_context) < 10:
            raise ValueError("query_context must be at least 10 characters")
        if len(query_context) > 5000:
            raise ValueError("query_context must be at most 5000 characters")
        
        logger.info(
            "collecting_online",
            query_length=len(query_context),
            max_samples=max_samples,
        )
        
        logger.info("Number of samples requested", max_samples=max_samples)
        
        # Construct search prompt
        prompt = self._construct_search_prompt(query_context, max_samples)
        
        # Call LLM to search
        try:
            response = self._call_llm(prompt)
            samples = self._parse_samples(response)
            
            # Limit to max_samples
            samples = samples[:max_samples]
            
            processing_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                "collection_complete",
                sample_count=len(samples),
                processing_time_ms=processing_time,
            )
            
            collection = Collection(
                method=CollectionMethod.ONLINE_SEARCH,
                sample_ids=[s.id for s in samples],
                query_context=query_context,
                metadata={
                    "processing_time_ms": processing_time,
                    "model": self.model,
                    "samples_found": len(samples),
                },
            )
            
            return collection, samples
            
        except Exception as e:
            logger.error("search_error", error=str(e))
            raise RuntimeError(f"Search service unavailable: {e}") from e

    def _construct_search_prompt(self, query_context: str, max_samples: int) -> str:
        """Construct search prompt for LLM."""
        prompt = f"""You are a search assistant. Based on the following context, find {max_samples} relevant text samples from the internet.

CONTEXT:
{query_context}

Instructions:
- Find up to {max_samples} relevant text samples
- Each sample should be 50-50000 characters
- Samples should be related to the themes in the context
- For each sample, provide:
  * content: The actual text content
  * source_url: URL where this content was found
  * title: A title or description
- Respond in JSON format

JSON Schema:
{{
  "samples": [
    {{
      "content": "Actual text content here...",
      "source_url": "https://example.com/article",
      "title": "Article Title"
    }}
  ]
}}

Respond with ONLY valid JSON, no other text."""
        
        return prompt

    def _call_llm(self, prompt: str) -> dict[str, Any]:
        """Call LLM API with prompt and search enabled."""
        if not self.api_key:
            raise RuntimeError("No API key configured")
        
        # Create chat with search enabled
        chat = self.client.chat.create(
            model=self.model,
            search_parameters=SearchParameters(
                mode="auto",
                return_citations=True,
            ),
        )
        chat.append(system("You are a search assistant that finds relevant content. Always respond with valid JSON."))
        chat.append(user(prompt))
        
        # Get response
        response = chat.sample()
        content = response.content
        
        # Parse JSON response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        return json.loads(content)

    def _parse_samples(self, response: dict[str, Any]) -> list[Sample]:
        """Parse LLM response into Sample objects."""
        samples = []
        
        for sample_data in response.get("samples", []):
            content = sample_data.get("content", "").strip()
            
            # Validate content length
            if len(content) < 5:
                logger.warning("sample_too_short", length=len(content))
                continue
            if len(content) > 50000:
                content = content[:50000]
            
            sample = Sample(
                content=content,
                source=SampleSource.INTERNET_SEARCH,
                metadata={
                    "source_url": sample_data.get("source_url", "unknown"),
                    "title": sample_data.get("title", ""),
                },
            )
            samples.append(sample)
        
        return samples

    def close(self):
        """Close client resources if needed."""
        # xai_sdk Client doesn't require explicit cleanup
        pass
