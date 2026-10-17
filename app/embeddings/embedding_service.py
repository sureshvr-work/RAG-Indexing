"""Local Sentence Transformers encoding, with explicit validation and truncation reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from app.core.config import EmbeddingConfig, PROJECT_ROOT


def validate_vectors(vectors: np.ndarray, count: int, dimensions: int) -> None:
    if vectors.dtype != np.float32 or vectors.shape != (count, dimensions):
        raise ValueError(f'expected float32 vectors with shape {(count, dimensions)}, received {vectors.dtype} {vectors.shape}')
    if not np.isfinite(vectors).all():
        raise ValueError('embeddings contain nonfinite values')
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4, rtol=0):
        raise ValueError('embeddings must have unit L2 norm for cosine search')


class EmbeddingService:
    def __init__(self, config: EmbeddingConfig, *, offline: bool = False) -> None:
        # Import only when encoding is needed; cached artifact checks need no model.
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(config.cpu_threads)
        torch.manual_seed(config.seed)
        cache = Path(config.cache_folder)
        if not cache.is_absolute():
            cache = PROJECT_ROOT / cache
        cache.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.model = SentenceTransformer(
            config.model_name,
            revision=config.revision,
            cache_folder=str(cache),
            device=config.device,
            local_files_only=offline,
            trust_remote_code=False,
            token=False,
        )
        self.model.eval()
        actual_dimensions = self.model.get_embedding_dimension()
        if actual_dimensions != config.dimensions:
            raise ValueError(f'model produces {actual_dimensions} dimensions; configured {config.dimensions}')
        if any(self.model.prompts.values()):
            raise ValueError('This workflow expects a model without query/document prompts')
        self.max_sequence_length = self.model.max_seq_length
        if type(self.max_sequence_length) is not int or self.max_sequence_length < 1:
            raise ValueError('model must expose a positive input token limit')

    def token_counts(self, texts: Sequence[str]) -> list[int]:
        counts: list[int] = []
        for start in range(0, len(texts), self.config.batch_size):
            tokenized = self.model.tokenizer(
                list(texts[start:start + self.config.batch_size]),
                truncation=False,
                padding=False,
                add_special_tokens=True,
                return_length=True,
                verbose=False,  # We intentionally measure full lengths; inference truncates separately.
            )
            counts.extend(int(length) for length in tokenized['length'])
        return counts

    def encode_documents(self, texts: Sequence[str], *, show_progress: bool = True) -> np.ndarray:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError('embedding inputs must be nonempty strings')
        vectors = self.model.encode_document(
            list(texts),
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            precision='float32',
        )
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        validate_vectors(vectors, len(texts), self.config.dimensions)
        return vectors

    def encode_query(self, query: str) -> np.ndarray:
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query must be a nonempty string')
        vectors = self.model.encode_query(
            [query], normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False, precision='float32',
        )
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        validate_vectors(vectors, 1, self.config.dimensions)
        return vectors[0]
