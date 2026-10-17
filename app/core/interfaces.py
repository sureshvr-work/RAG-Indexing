"""Common operations for the exact-search foundation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from app.core.filters import MetadataFilter
from app.core.models import DatasetIdentity, Document, SearchResult, StoreStats
from app.embeddings.embedding_service import validate_vectors


class VectorStore(ABC):
    backend: str
    index_type: str

    def __init__(self, identity: DatasetIdentity) -> None:
        self.identity = identity

    def validate_insert(self, documents: Sequence[Document], embeddings: np.ndarray) -> np.ndarray:
        if not documents:
            raise ValueError('insert requires at least one document')
        validate_vectors(embeddings, len(documents), self.identity.dimensions)
        ids = np.array([document.id for document in documents], dtype=np.int64)
        if len(set(ids.tolist())) != len(documents):
            raise ValueError('insert contains duplicate document IDs')
        return ids

    def validate_query(self, query_vector: np.ndarray, top_k: int) -> np.ndarray:
        if type(top_k) is not int or top_k < 1:
            raise ValueError('top_k must be a positive integer')
        query = np.asarray(query_vector, dtype=np.float32)
        if query.shape != (self.identity.dimensions,) or not np.isfinite(query).all():
            raise ValueError('query must be a finite vector with the configured dimensions')
        norm = float(np.linalg.norm(query))
        if norm == 0:
            raise ValueError('query vector cannot be zero')
        return np.ascontiguousarray(query / norm, dtype=np.float32)

    @abstractmethod
    def create_index(self) -> None: ...

    @abstractmethod
    def insert(self, documents: Sequence[Document], embeddings: np.ndarray) -> None: ...

    @abstractmethod
    def update_metadata(self, patches: dict[int, dict]) -> None: ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 10, filters: MetadataFilter | None = None) -> SearchResult: ...

    @abstractmethod
    def inspect_document(self, stable_id: int) -> dict: ...

    @abstractmethod
    def list_ids(self) -> set[int]: ...

    @abstractmethod
    def get_stats(self) -> StoreStats: ...

    def close(self) -> None:
        pass
