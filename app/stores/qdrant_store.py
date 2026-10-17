"""Qdrant dense vectors and full document payloads using supported public APIs."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Sequence

import numpy as np
from qdrant_client import QdrantClient, models

from app.core.interfaces import VectorStore
from app.core.metadata import validate_metadata_patches
from app.core.filters import MetadataFilter
from app.core.models import DatasetIdentity, Document, SearchHit, SearchResult, StoreStats


class QdrantVectorStore(VectorStore):
    backend = 'qdrant'
    index_type = 'exact cosine scan (HNSW disabled)'

    def __init__(self, identity: DatasetIdentity, url: str) -> None:
        super().__init__(identity)
        self.client = QdrantClient(url=url, timeout=30)

    def create_index(self) -> None:
        name = self.identity.namespace
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=self.identity.dimensions, distance=models.Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(m=0),
                metadata={'vector_lab_identity': self.identity.model_dump()},
            )
        info = self.client.get_collection(name)
        vectors = info.config.params.vectors
        if not isinstance(vectors, models.VectorParams) or vectors.size != self.identity.dimensions or vectors.distance != models.Distance.COSINE:
            raise ValueError('Qdrant collection dimensions or metric do not match')
        if (info.config.metadata or {}).get('vector_lab_identity') != self.identity.model_dump():
            raise ValueError('Qdrant dataset identity does not match the requested artifacts')

    def insert(self, documents: Sequence[Document], embeddings: np.ndarray) -> None:
        self.validate_insert(documents, embeddings)
        for start in range(0, len(documents), 128):
            points = [models.PointStruct(id=d.id, vector=v.tolist(), payload=d.model_dump(mode='json'))
                      for d, v in zip(documents[start:start + 128], embeddings[start:start + 128], strict=True)]
            self.client.upsert(collection_name=self.identity.namespace, points=points, wait=True)

    def update_metadata(self, patches: dict[int, dict]) -> None:
        validate_metadata_patches(patches)
        ids = list(patches)
        existing = set()
        for start in range(0, len(ids), 128):
            points = self.client.retrieve(self.identity.namespace, ids[start:start + 128],
                                          with_payload=False, with_vectors=False)
            existing.update(point.id for point in points)
        if existing != set(ids):
            raise ValueError('metadata update contains unknown document IDs')
        groups: dict[str, list[int]] = {}
        for stable_id, patch in patches.items():
            groups.setdefault(json.dumps(patch, sort_keys=True, allow_nan=False), []).append(stable_id)
        for encoded, group in groups.items():
            self.client.set_payload(collection_name=self.identity.namespace, payload=json.loads(encoded),
                                    points=group, key='metadata', wait=True)

    def search(self, query_vector: np.ndarray, top_k: int = 10, filters: MetadataFilter | None = None) -> SearchResult:
        query = self.validate_query(query_vector, top_k)
        start = perf_counter()
        conditions = []
        if filters is not None:
            if filters.groups:
                conditions.append(models.FieldCondition(key='metadata.lab_group', match=models.MatchAny(any=filters.groups)))
            if filters.review_statuses:
                conditions.append(models.FieldCondition(key='metadata.lab_review_status', match=models.MatchAny(any=filters.review_statuses)))
            if filters.year_min is not None or filters.year_max is not None:
                conditions.append(models.FieldCondition(key='metadata.lab_year', range=models.Range(gte=filters.year_min, lte=filters.year_max)))
        query_filter = models.Filter(must=conditions) if conditions else None
        points = self.client.query_points(collection_name=self.identity.namespace, query=query.tolist(),
                                          search_params=models.SearchParams(exact=True), query_filter=query_filter, limit=top_k,
                                          with_payload=True, with_vectors=False).points
        hits = []
        for point in points:
            document = Document.model_validate(point.payload)
            if document.id != point.id:
                raise ValueError('Qdrant point ID and document payload are inconsistent')
            hits.append(SearchHit(document=document, score=point.score))
        hits.sort(key=lambda hit: (-hit.score, hit.document.id))
        return SearchResult(backend=self.backend, namespace=self.identity.namespace, index_type=self.index_type,
                            latency_ms=(perf_counter() - start) * 1000, hits=hits)

    def inspect_document(self, stable_id: int) -> dict:
        points = self.client.retrieve(self.identity.namespace, [stable_id], with_payload=True, with_vectors=True)
        if not points:
            raise ValueError('document ID not found')
        document = Document.model_validate(points[0].payload)
        if document.id != stable_id:
            raise ValueError('Qdrant point ID and document payload are inconsistent')
        return {'document': document.model_dump(), 'vector': points[0].vector}

    def list_ids(self) -> set[int]:
        ids: set[int] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(collection_name=self.identity.namespace, limit=256,
                                                 offset=offset, with_payload=False, with_vectors=False)
            for point in points:
                if not isinstance(point.id, int):
                    raise ValueError('Qdrant collection contains unexpected nonnumeric IDs')
                ids.add(point.id)
            if offset is None:
                break
        return ids

    def get_stats(self) -> StoreStats:
        count = self.client.count(collection_name=self.identity.namespace, exact=True).count
        return StoreStats(backend=self.backend, namespace=self.identity.namespace, count=count,
                          dimensions=self.identity.dimensions, index_type=self.index_type)

    def close(self) -> None:
        self.client.close()
