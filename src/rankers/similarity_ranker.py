"""Similarity-based ranking of samples."""

import time
from typing import Optional
import structlog

from src.models.sample import Sample
from src.models.similarity_score import SimilarityScore
from src.models.ranked_result import RankedResult, SimilarityDetails
from src.models.ranking_result import RankingResult
from src.rankers.embedding_service import EmbeddingService
from src.rankers.similarity_calculator import SimilarityCalculator

logger = structlog.get_logger()


class SimilarityRanker:
    """Ranks samples by similarity to reference samples."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        similarity_calculator: Optional[SimilarityCalculator] = None,
    ):
        """Initialize similarity ranker.
        
        Args:
            embedding_service: Service for generating embeddings (default: new instance)
            similarity_calculator: Calculator for similarities (default: new instance with cosine)
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.similarity_calculator = similarity_calculator or SimilarityCalculator(method="cosine")
        logger.info(
            "similarity_ranker_init",
            embedding_dim=self.embedding_service.embedding_dimension,
            similarity_method=self.similarity_calculator.method,
        )

    async def rank(
        self,
        reference_samples: list[Sample],
        candidate_samples: list[Sample],
        threshold: float = 0.7,
        top_n: Optional[int] = 10,
        include_embeddings: bool = False,
    ) -> RankingResult:
        """Rank candidate samples by similarity to reference samples.
        
        Args:
            reference_samples: Reference samples to compare against (1-10 samples)
            candidate_samples: Candidate samples to rank (1-500 samples)
            threshold: Minimum similarity threshold (0.0-1.0, default: 0.7)
            top_n: Maximum number of results to return (1-100, None = all, default: 10)
            include_embeddings: Whether to include embedding vectors in results
            
        Returns:
            RankingResult with ranked samples and statistics
            
        Raises:
            ValueError: If parameters are out of valid ranges
        """
        # Start timing
        start_time = time.time()
        
        # Validate reference samples count
        if not reference_samples:
            raise ValueError("At least one reference sample is required")
        
        if len(reference_samples) > 10:
            raise ValueError(f"Maximum 10 reference samples allowed, got {len(reference_samples)}")
        
        # Validate candidate samples count
        if not candidate_samples:
            logger.info("no_candidates", reference_count=len(reference_samples))
            return RankingResult(
                results=[],
                total_candidates=0,
                returned_count=0,
                threshold=threshold,
                below_threshold_count=0,
                processing_time_ms=0,
            )
        
        if len(candidate_samples) > 500:
            raise ValueError(f"Maximum 500 candidate samples allowed, got {len(candidate_samples)}")
        
        # Validate threshold
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}")
        
        # Validate top_n
        if top_n is not None and (top_n < 1 or top_n > 100):
            raise ValueError(f"top_n must be between 1 and 100 or None, got {top_n}")
        
        logger.info(
            "ranking_start",
            reference_count=len(reference_samples),
            candidate_count=len(candidate_samples),
            threshold=threshold,
            top_n=top_n,
        )
        
        # Generate embeddings for all samples
        logger.info("generating_embeddings")

        reference_texts = [s.content for s in reference_samples]
        candidate_texts = [s.content for s in candidate_samples]
        reference_ids = [s.id for s in reference_samples]

        # Reference samples are treated as queries (what we're looking for)
        # Candidate samples are treated as documents (what we're searching through)
        reference_embeddings = await self.embedding_service.embed(reference_texts, is_query=True)
        candidate_embeddings = await self.embedding_service.embed(candidate_texts, is_query=False)
        
        logger.info(
            "embeddings_generated",
            reference_count=len(reference_embeddings),
            candidate_count=len(candidate_embeddings),
        )
        
        # Calculate similarities
        # For each reference, compute similarities to all candidates
        logger.info("calculating_similarities")
        
        # Store results for each reference's ranking
        all_rankings: list[list[tuple[float, int, Sample]]] = []
        
        for ref_idx, reference_embedding in enumerate(reference_embeddings):
            # Compute similarities for this reference against all candidates
            similarities = await self.similarity_calculator.calculate_batch(
                reference_embedding,
                candidate_embeddings,
            )
            
            # Create ranking: (similarity_score, candidate_idx, candidate_sample)
            ranking = [
                (similarities[i], i, candidate_samples[i])
                for i in range(len(candidate_samples))
                if similarities[i] >= threshold
            ]
            
            # Sort by similarity descending
            ranking.sort(key=lambda x: x[0], reverse=True)
            
            all_rankings.append(ranking)
        
        # Merge rankings from all references
        # Track best score for each candidate across all references
        candidate_scores: dict[int, tuple[float, str, list[float]]] = {}
        
        for ref_idx, ranking in enumerate(all_rankings):
            ref_id = reference_ids[ref_idx]
            
            for similarity, candidate_idx, _ in ranking:
                if candidate_idx not in candidate_scores:
                    # First time seeing this candidate
                    candidate_scores[candidate_idx] = (
                        similarity,  # max_score
                        ref_id,      # closest_reference_id
                        [similarity], # all scores
                    )
                else:
                    max_score, closest_ref, scores = candidate_scores[candidate_idx]
                    scores.append(similarity)
                    # Update if this reference has higher similarity
                    if similarity > max_score:
                        candidate_scores[candidate_idx] = (similarity, ref_id, scores)
                    else:
                        candidate_scores[candidate_idx] = (max_score, closest_ref, scores)
        
        total_candidates = len(candidate_samples)
        below_threshold_count = total_candidates - len(candidate_scores)
        
        logger.info(
            "similarities_calculated",
            passed_threshold=len(candidate_scores),
            filtered_out=below_threshold_count,
        )
        
        ranked_results = [
            (max_score, candidate_idx, closest_ref, scores)
            for candidate_idx, (max_score, closest_ref, scores) in candidate_scores.items()
        ]
        
        ranked_results.sort(key=lambda x: x[0], reverse=True)
        
        if top_n is not None:
            ranked_results = ranked_results[:top_n]
        
        final_results = [
            RankedResult(
                rank=rank + 1,  # 1-indexed rank
                sample=candidate_samples[candidate_idx],
                score=max_score,
                similarity_details=SimilarityDetails(
                    max_score=max_score,
                    avg_score=sum(scores) / len(scores),
                    closest_reference_id=closest_ref,
                ),
            )
            for rank, (max_score, candidate_idx, closest_ref, scores) in enumerate(ranked_results)
        ]
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            "ranking_complete",
            total_ranked=len(final_results),
            top_similarity=final_results[0].score if final_results else None,
            processing_time_ms=processing_time_ms,
        )
        
        return RankingResult(
            results=final_results,
            total_candidates=total_candidates,
            returned_count=len(final_results),
            threshold=threshold,
            below_threshold_count=below_threshold_count,
            processing_time_ms=processing_time_ms,
        )
