"""Embedding service using sentence-transformers with EmbeddingGemma."""

from typing import Any

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()


class EmbeddingService:
    """Service for generating text embeddings using EmbeddingGemma."""

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        device: str | None = None,
        batch_size: int = 32,
    ):
        """Initialize embedding service.
        
        Args:
            model_name: Model name from sentence-transformers (default: google/embeddinggemma-300m)
            device: Device to use ('cpu', 'cuda', or None for auto)
            batch_size: Batch size for encoding
            
        Note:
            EmbeddingGemma requires a HuggingFace token set in HF_TOKEN environment variable.
            The model produces 768-dimensional embeddings.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model: SentenceTransformer | None = None
        self.device = device
        self._embedding_dim: int | None = None
        
        logger.info(
            "embedding_service_init",
            model_name=model_name,
            device=device or "auto",
            batch_size=batch_size,
        )

    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            logger.info("loading_embedding_model", model_name=self.model_name)
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(
                "embedding_model_loaded",
                model_name=self.model_name,
                dimension=self._embedding_dim,
            )

    async def embed(self, texts: list[str], is_query: bool = True) -> list[list[float]]:
        """Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            is_query: Whether the texts are queries (True) or documents (False)
            
        Returns:
            List of embedding vectors
            
        Note:
            EmbeddingGemma distinguishes between query and document embeddings.
            Use is_query=True for search queries, questions, etc.
            Use is_query=False for documents, passages to be searched, etc.
        """
        if not texts:
            return []
        
        self._load_model()
        assert self.model is not None
        
        logger.info(
            "generating_embeddings",
            text_count=len(texts),
            mode="query" if is_query else "document",
        )
        
        # Generate embeddings using appropriate method
        if is_query:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                prompt_name="query",
            )
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                prompt_name="document",
            )
        
        # Convert to list of lists
        result = embeddings.tolist()
        
        logger.info(
            "embeddings_generated",
            count=len(result),
            dimension=len(result[0]) if result else 0,
        )
        
        return result

    async def embed_single(self, text: str, is_query: bool = True) -> list[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            is_query: Whether the text is a query (True) or document (False)
            
        Returns:
            Embedding vector
        """
        embeddings = await self.embed([text], is_query=is_query)
        return embeddings[0] if embeddings else []

    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension.
        
        Returns:
            Dimension of embeddings (768 for EmbeddingGemma)
        """
        if self._embedding_dim is None:
            self._load_model()
        return self._embedding_dim or 768

    def get_model_info(self) -> dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary with model details
        """
        self._load_model()
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "device": str(self.model.device) if self.model else "not loaded",
            "max_sequence_length": self.model.max_seq_length if self.model else None,
        }
