"""Isolate FAISS's native runtime from the embedding model's PyTorch runtime."""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from app.core.interfaces import VectorStore
from app.core.filters import MetadataFilter
from app.core.models import DatasetIdentity, Document, SearchResult, StoreStats


def _worker(connection, identity: DatasetIdentity, directory: Path) -> None:
    # Import only in the spawned child. The parent never loads FAISS's OpenMP.
    # macOS runtime conflict: https://github.com/pytorch/pytorch/issues/149201
    try:
        from app.stores._faiss_native import NativeFaissVectorStore
        store = NativeFaissVectorStore(identity, directory)
        while True:
            operation, args = connection.recv()
            if operation == 'close':
                break
            try:
                if operation not in {'create_index', 'insert', 'search', 'list_ids', 'get_stats', 'update_metadata', 'inspect_document', 'build_hnsw', 'search_hnsw'}:
                    raise ValueError('unsupported FAISS operation')
                connection.send(('ok', getattr(store, operation)(*args)))
            except Exception as error:
                connection.send(('error', (type(error).__name__, str(error))))
    except EOFError:
        pass
    finally:
        connection.close()


class FaissVectorStore(VectorStore):
    backend = 'faiss'
    index_type = 'IndexIDMap2(IndexFlatIP), exact'

    def __init__(self, identity: DatasetIdentity, directory: Path) -> None:
        super().__init__(identity)
        self.directory = directory / identity.namespace
        context = multiprocessing.get_context('spawn')
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_worker, args=(child, identity, directory), daemon=True)
        self._process.start()
        child.close()
        self._closed = False

    def _call(self, operation: str, *args):
        if self._closed:
            raise RuntimeError('FAISS store is closed')
        try:
            self._connection.send((operation, args))
            if not self._connection.poll(60):
                raise RuntimeError('FAISS worker timed out')
            status, value = self._connection.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError) as error:
            raise RuntimeError(f'FAISS worker exited unexpectedly (exit code {self._process.exitcode})') from error
        if status == 'error':
            error_type, message = value
            if error_type == 'ValueError':
                raise ValueError(message)
            raise RuntimeError(message)
        return value

    def create_index(self) -> None:
        self._call('create_index')

    def insert(self, documents: Sequence[Document], embeddings: np.ndarray) -> None:
        self.validate_insert(documents, embeddings)
        self._call('insert', documents, embeddings)

    def update_metadata(self, patches: dict[int, dict]) -> None:
        self._call('update_metadata', patches)

    def search(self, query_vector: np.ndarray, top_k: int = 10, filters: MetadataFilter | None = None) -> SearchResult:
        query = self.validate_query(query_vector, top_k)
        start = perf_counter()
        result = self._call('search', query, top_k, filters)
        # Include local IPC and result transfer, as database timings include transport.
        result.latency_ms = (perf_counter() - start) * 1000
        return result

    def inspect_document(self, stable_id: int) -> dict:
        return self._call('inspect_document', stable_id)

    def build_hnsw(self, vectors: np.ndarray, ids: np.ndarray, m: int, ef_construction: int) -> float:
        return self._call('build_hnsw', vectors, ids, m, ef_construction)

    def search_hnsw(self, query: np.ndarray, top_k: int, ef_search: int) -> SearchResult:
        start = perf_counter()
        result = self._call('search_hnsw', self.validate_query(query, top_k), top_k, ef_search)
        result.latency_ms = (perf_counter() - start) * 1000
        return result

    def list_ids(self) -> set[int]:
        return self._call('list_ids')

    def get_stats(self) -> StoreStats:
        return self._call('get_stats')

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.is_alive():
                try:
                    self._connection.send(('close', ()))
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self._process.join(timeout=5)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=5)
        finally:
            self._connection.close()
