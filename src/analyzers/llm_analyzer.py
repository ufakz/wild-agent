import json
import os
import time
from typing import Any

import structlog
from xai_sdk import Client
from xai_sdk.chat import system, user

from src.models import AnalysisResult, Sample, Theme

logger = structlog.get_logger()

class LLMAnalyzer:
    """Analyzes samples using LLM to extract themes."""

    def __init__(
        self,
        api_key: str | None = os.getenv("OPENAI_API_KEY"),
        model: str = "grok-4",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        """Initialize LLM Analyzer.
        
        Args:
            api_key: API key for LLM service (or set XAI_API_KEY env var)
            api_base: Base URL for API (ignored, kept for compatibility)
            model: Model name to use
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided - LLM calls will fail")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.client = Client(api_key=self.api_key)

    async def analyze(
        self,
        samples: list[Sample],
        max_themes: int = 5,
        min_confidence: float = 0.8,
    ) -> AnalysisResult:
        """Analyze samples to extract themes.
        
        Args:
            samples: Samples to analyze (1-10)
            max_themes: Maximum themes to extract (1-10)
            min_confidence: Minimum confidence threshold (0.0-1.0)
            
        Returns:
            AnalysisResult with extracted themes
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If LLM API call fails
        """
        start_time = time.time()
        
        if not samples:
            raise ValueError("Samples list cannot be empty")
        if len(samples) > 10:
            raise ValueError(f"Maximum 10 samples allowed, got {len(samples)}")
        if not 1 <= max_themes <= 10:
            raise ValueError(f"max_themes must be 1-10, got {max_themes}")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be 0.0-1.0, got {min_confidence}")
        
        logger.info(
            "analyzing_samples",
            sample_count=len(samples),
            max_themes=max_themes,
            min_confidence=min_confidence,
        )
        
        prompt = self._construct_prompt(samples, max_themes, min_confidence)
        
        try:
            response = self._call_llm(prompt)
            themes, summary, confidence = self._parse_response(response, samples)
            
            filtered_themes = [t for t in themes if t.confidence >= min_confidence]
            filtered_themes = filtered_themes[:max_themes]
            
            processing_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                "analysis_complete",
                theme_count=len(filtered_themes),
                processing_time_ms=processing_time,
            )
            
            return AnalysisResult(
                query_context=" ".join([s.content[:100] for s in samples]),
                theme_ids=[t.id for t in filtered_themes],
                url_target_ids=[],  
                metadata={
                    "processing_time_ms": processing_time,
                    "model": self.model,
                    "total_themes_extracted": len(themes),
                    "filtered_by_confidence": len(themes) - len(filtered_themes),
                },
            )
            
        except Exception as e:
            logger.error("llm_error", error=str(e))
            raise RuntimeError(f"LLM service unavailable: {e}") from e

    def _construct_prompt(
        self,
        samples: list[Sample],
        max_themes: int,
        min_confidence: float,
    ) -> str:
        """Construct analysis prompt for LLM."""
        samples_text = "\n\n".join([
            f"Sample {i+1}:\n{sample.content}"
            for i, sample in enumerate(samples)
        ])
        
        prompt = f"""Analyze the following text samples and extract common themes/topics.

{samples_text}

Instructions:
- Identify up to {max_themes} distinct themes
- For each theme, provide:
  * A clear name (5-100 characters)
  * A detailed description (20-1000 characters)
  * 3-20 keywords (each 2-50 characters)
  * A confidence score (0.0-1.0, only themes >= {min_confidence})
  * Which sample IDs relate to this theme
- Also provide an overall summary (50-500 characters)
- Respond in JSON format

JSON Schema:
{{
  "themes": [
    {{
      "name": "Theme Name",
      "description": "Detailed description",
      "keywords": ["keyword1", "keyword2", "keyword3"],
      "confidence": 0.85,
      "sample_indices": [0, 1]
    }}
  ],
  "summary": "Overall summary of all samples",
  "confidence": 0.90
}}

Respond with ONLY valid JSON, no other text."""
        
        return prompt

    def _call_llm(self, prompt: str) -> dict[str, Any]:
        """Call LLM API with prompt."""
        if not self.api_key:
            raise RuntimeError("No API key configured")
        
        chat = self.client.chat.create(model=self.model)
        chat.append(system("You are a helpful assistant that analyzes text and extracts themes. Always respond with valid JSON."))
        chat.append(user(prompt))
        
        response = chat.sample()
        content = response.content
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        return json.loads(content)

    def _parse_response(
        self,
        response: dict[str, Any],
        samples: list[Sample],
    ) -> tuple[list[Theme], str, float]:
        """Parse LLM response into Theme objects."""
        themes = []
        
        for theme_data in response.get("themes", []):
            
            sample_indices = theme_data.get("sample_indices", [])
            sample_ids = [
                samples[i].id for i in sample_indices
                if 0 <= i < len(samples)
            ]
            
            theme = Theme(
                name=theme_data["name"],
                description=theme_data["description"],
                keywords=theme_data["keywords"],
                confidence=theme_data["confidence"],
                sample_ids=sample_ids or [samples[0].id],
            )
            
            themes.append(theme)
        
        summary = response.get("summary", "Analysis of provided samples")
        confidence = response.get("confidence", 0.7)
        
        return themes, summary, confidence