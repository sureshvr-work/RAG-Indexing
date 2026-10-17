"""Shared typed documents and search results for every backend."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    id: int = Field(strict=True, ge=0, lt=2**63)
    document_id: str
    title: str
    abstract: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str


class DatasetIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)
    namespace: str = Field(pattern=r'^[a-z][a-z0-9_]{0,62}$')
    dimensions: int = Field(ge=1, le=16000)
    documents_sha256: str
    vectors_sha256: str
    embedding_config: dict[str, Any]
    metric: Literal['cosine'] = 'cosine'


class SearchHit(BaseModel):
    document: Document
    score: float


class SearchResult(BaseModel):
    backend: str
    namespace: str
    index_type: str
    metric: Literal['cosine'] = 'cosine'
    score_type: Literal['cosine_similarity'] = 'cosine_similarity'
    latency_ms: float
    hits: list[SearchHit]
    metrics: dict[str, Any] = Field(default_factory=dict)


class StoreStats(BaseModel):
    backend: str
    namespace: str
    count: int
    dimensions: int
    index_type: str
    metric: Literal['cosine'] = 'cosine'
    storage_bytes: int | None = None
