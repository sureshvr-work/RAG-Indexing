"""Construct selected stores and close their clients reliably."""

from contextlib import ExitStack, contextmanager
from typing import Iterator, Sequence

from app.core.config import BackendConfig
from app.core.interfaces import VectorStore
from app.core.models import DatasetIdentity
from app.stores.faiss_store import FaissVectorStore
from app.stores.pgvector_store import PgVectorStore
from app.stores.qdrant_store import QdrantVectorStore

BACKENDS = ('faiss', 'pgvector', 'qdrant')


@contextmanager
def open_stores(identity: DatasetIdentity, config: BackendConfig, backends: Sequence[str] = BACKENDS) -> Iterator[list[VectorStore]]:
    if not backends or len(set(backends)) != len(backends) or any(name not in BACKENDS for name in backends):
        raise ValueError(f'select unique backend names from {BACKENDS}')
    with ExitStack() as stack:
        stores = []
        for name in backends:
            if name == 'faiss':
                store = FaissVectorStore(identity, config.faiss_directory)
            elif name == 'pgvector':
                store = PgVectorStore(identity, config)
            else:
                store = QdrantVectorStore(identity, config.qdrant_url)
            stack.callback(store.close)
            stores.append(store)
        yield stores
