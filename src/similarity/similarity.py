import asyncio
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=4)
def _get_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


class EmbeddingService:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name

    def encode_sync(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        model = _get_model(self.model_name)
        vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if vectors.ndim == 1:
            return [vectors]
        return list(vectors)

    async def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = await asyncio.to_thread(self.encode_sync, texts)
        return [v.tolist() for v in vectors]


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    a = np.array(vec1, dtype=np.float64)
    b = np.array(vec2, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _max_cosine_similarity(
    candidate_embedding: list[float],
    reference_embeddings: list[list[float]],
) -> tuple[float, int | None]:
    if not reference_embeddings:
        return 0.0, None

    best_score = -1.0
    best_index: int | None = None
    for i, ref in enumerate(reference_embeddings):
        score = _cosine_similarity(candidate_embedding, ref)
        if score > best_score:
            best_score = score
            best_index = i

    return max(best_score, 0.0), best_index


async def embed_reference_texts(
    texts: list[str],
    *,
    embedding_model: str = DEFAULT_MODEL,
) -> list[list[float]]:
    return await EmbeddingService(model_name=embedding_model).encode(texts)


async def score_candidate_similarity(
    candidate_text: str,
    reference_embeddings: list[list[float]],
    *,
    embedding_model: str = DEFAULT_MODEL,
) -> tuple[float, int | None]:
    if not reference_embeddings:
        return 0.0, None

    vecs = await EmbeddingService(model_name=embedding_model).encode([candidate_text])
    if not vecs:
        return 0.0, None

    return _max_cosine_similarity(vecs[0], reference_embeddings)
