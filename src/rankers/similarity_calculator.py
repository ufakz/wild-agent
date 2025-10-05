"""Similarity calculator for embeddings."""

import numpy as np
from scipy.spatial.distance import cosine, euclidean
import structlog

logger = structlog.get_logger()


class SimilarityCalculator:
    """Calculator for similarity scores between embeddings."""

    def __init__(self, method: str = "cosine"):
        """Initialize similarity calculator.
        
        Args:
            method: Similarity method ('cosine', 'euclidean', 'dot_product')
        """
        if method not in ["cosine", "euclidean", "dot_product"]:
            raise ValueError(
                f"Invalid method '{method}'. Must be 'cosine', 'euclidean', or 'dot_product'"
            )
        
        self.method = method
        logger.info("similarity_calculator_init", method=method)

    async def calculate(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """Calculate similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0.0-1.0, higher = more similar)
        """
        if len(embedding1) != len(embedding2):
            raise ValueError(
                f"Embedding dimensions must match: {len(embedding1)} != {len(embedding2)}"
            )
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        if self.method == "cosine":
            # Cosine similarity: 1 - cosine_distance
            # scipy.cosine returns distance, we want similarity
            similarity = 1.0 - cosine(vec1, vec2)
        elif self.method == "euclidean":
            # Convert euclidean distance to similarity (0-1 range)
            # Closer = higher similarity
            distance = euclidean(vec1, vec2)
            # Normalize to 0-1 range (assuming max distance is sqrt(2))
            similarity = 1.0 / (1.0 + distance)
        else:  # dot_product
            # Dot product (assumes normalized vectors)
            similarity = float(np.dot(vec1, vec2))
            # Ensure 0-1 range
            similarity = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
        
        # Ensure valid range
        similarity = max(0.0, min(1.0, similarity))
        
        return float(similarity)

    async def calculate_batch(
        self,
        reference_embedding: list[float],
        candidate_embeddings: list[list[float]],
    ) -> list[float]:
        """Calculate similarities between one reference and multiple candidates.
        
        Args:
            reference_embedding: Reference embedding vector
            candidate_embeddings: List of candidate embedding vectors
            
        Returns:
            List of similarity scores
        """
        if not candidate_embeddings:
            return []
        
        # Check dimensions
        ref_dim = len(reference_embedding)
        for i, emb in enumerate(candidate_embeddings):
            if len(emb) != ref_dim:
                raise ValueError(
                    f"Embedding {i} dimension mismatch: {len(emb)} != {ref_dim}"
                )
        
        ref_vec = np.array(reference_embedding)
        cand_matrix = np.array(candidate_embeddings)
        
        if self.method == "cosine":
            # Vectorized cosine similarity
            # Normalize vectors
            ref_norm = ref_vec / (np.linalg.norm(ref_vec) + 1e-8)
            cand_norms = cand_matrix / (np.linalg.norm(cand_matrix, axis=1, keepdims=True) + 1e-8)
            # Dot product of normalized vectors = cosine similarity
            similarities = np.dot(cand_norms, ref_norm)
        elif self.method == "euclidean":
            # Vectorized euclidean distance
            distances = np.linalg.norm(cand_matrix - ref_vec, axis=1)
            similarities = 1.0 / (1.0 + distances)
        else:  # dot_product
            similarities = np.dot(cand_matrix, ref_vec)
            # Normalize to 0-1 range
            similarities = (similarities + 1.0) / 2.0
        
        # Ensure valid range
        similarities = np.clip(similarities, 0.0, 1.0)
        
        return similarities.tolist()

    async def calculate_pairwise(
        self,
        embeddings: list[list[float]],
    ) -> list[list[float]]:
        """Calculate pairwise similarities between all embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Matrix of pairwise similarities (n x n)
        """
        if not embeddings:
            return []
        
        n = len(embeddings)
        matrix = np.array(embeddings)
        
        if self.method == "cosine":
            # Normalize all vectors
            norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
            # Pairwise dot products
            similarities = np.dot(norms, norms.T)
        elif self.method == "euclidean":
            # Pairwise euclidean distances
            # Using broadcasting for efficiency
            diff = matrix[:, np.newaxis, :] - matrix[np.newaxis, :, :]
            distances = np.linalg.norm(diff, axis=2)
            similarities = 1.0 / (1.0 + distances)
        else:  # dot_product
            similarities = np.dot(matrix, matrix.T)
            # Normalize to 0-1 range
            similarities = (similarities + 1.0) / 2.0
        
        # Ensure valid range
        similarities = np.clip(similarities, 0.0, 1.0)
        
        return similarities.tolist()
