"""Ingest the exact same saved vectors and documents into each selected store."""

from typing import Sequence

from app.core.artifacts import EmbeddingBundle
from app.core.interfaces import VectorStore
from app.core.models import StoreStats


def ingest_bundle(bundle: EmbeddingBundle, stores: Sequence[VectorStore]) -> list[StoreStats]:
    expected = set(bundle.ids.tolist())
    results = []
    for store in stores:
        if store.identity != bundle.identity:
            raise ValueError('store identity does not match the embedding bundle')
        store.create_index()
        if not store.list_ids().issubset(expected):
            raise ValueError(f'{store.backend} has IDs outside this sample; use a different dataset store')
        store.insert(bundle.documents, bundle.vectors)
        if store.list_ids() != expected:
            raise ValueError(f'{store.backend} did not ingest exactly the expected document IDs')
        stats = store.get_stats()
        if stats.count != len(expected):
            raise ValueError(f'{store.backend} document count does not match')
        results.append(stats)
    return results
