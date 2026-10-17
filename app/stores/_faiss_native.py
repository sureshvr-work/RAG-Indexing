"""Exact cosine search with FAISS and a separately persisted document mapping."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from time import perf_counter
from typing import Sequence

import faiss
import numpy as np

from app.core.artifacts import sha256_file
from app.core.interfaces import VectorStore
from app.core.metadata import validate_metadata_patches
from app.core.filters import MetadataFilter
from app.core.models import DatasetIdentity, Document, SearchHit, SearchResult, StoreStats


class NativeFaissVectorStore(VectorStore):
    backend = 'faiss'
    index_type = 'IndexIDMap2(IndexFlatIP), exact'

    def __init__(self, identity: DatasetIdentity, directory: Path) -> None:
        super().__init__(identity)
        self.directory = directory / identity.namespace
        self.index = None
        self.documents: dict[int, Document] = {}

    def create_index(self) -> None:
        pointer = self.directory / 'active.json'
        if pointer.exists():
            generation = json.loads(pointer.read_text())['generation']
            if not isinstance(generation, str) or not generation.isalnum():
                raise ValueError('invalid FAISS generation pointer')
            folder = self.directory / generation
            manifest = json.loads((folder / 'manifest.json').read_text())
            if manifest.get('identity') != self.identity.model_dump():
                raise ValueError('FAISS dataset identity does not match the requested artifacts')
            for name in ('index.faiss', 'documents.json'):
                if sha256_file(folder / name) != manifest['sha256'][name]:
                    raise ValueError(f'FAISS {name} failed its integrity check')
            self.index = faiss.read_index(str(folder / 'index.faiss'))
            self.documents = {row['id']: Document.model_validate(row)
                              for row in json.loads((folder / 'documents.json').read_text())}
            if self.index.d != self.identity.dimensions or self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
                raise ValueError('FAISS index dimensions or metric do not match')
            if self.index.ntotal != len(self.documents) or set(faiss.vector_to_array(self.index.id_map).tolist()) != set(self.documents):
                raise ValueError('FAISS vector IDs and document mapping are inconsistent')
        else:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.identity.dimensions))

    def _save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        generation = uuid.uuid4().hex
        folder = self.directory / generation
        folder.mkdir()
        faiss.write_index(self.index, str(folder / 'index.faiss'))
        rows = [self.documents[key].model_dump(mode='json') for key in sorted(self.documents)]
        (folder / 'documents.json').write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
        manifest = {'identity': self.identity.model_dump(),
                    'sha256': {name: sha256_file(folder / name) for name in ('index.faiss', 'documents.json')}}
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        with tempfile.NamedTemporaryFile(mode='w', dir=self.directory, delete=False) as stream:
            pointer = Path(stream.name)
            stream.write(json.dumps({'generation': generation}))
        os.replace(pointer, self.directory / 'active.json')

    def insert(self, documents: Sequence[Document], embeddings: np.ndarray) -> None:
        ids = self.validate_insert(documents, embeddings)
        if self.index is None:
            raise RuntimeError('create_index must be called before insert')
        # Remove matching IDs before insertion to make repeated ingestion an upsert.
        self.index.remove_ids(ids)
        self.index.add_with_ids(np.ascontiguousarray(embeddings), ids)
        self.documents.update({document.id: document for document in documents})
        self._save()

    def update_metadata(self, patches: dict[int, dict]) -> None:
        validate_metadata_patches(patches)
        if self.index is None:
            raise RuntimeError('create_index must be called before updating metadata')
        if not set(patches).issubset(self.documents):
            raise ValueError('metadata update contains unknown document IDs')
        for stable_id, patch in patches.items():
            document = self.documents[stable_id]
            self.documents[stable_id] = document.model_copy(update={'metadata': {**document.metadata, **patch}})
        self._save()

    def search(self, query_vector: np.ndarray, top_k: int = 10, filters: MetadataFilter | None = None) -> SearchResult:
        query = self.validate_query(query_vector, top_k)
        if self.index is None:
            raise RuntimeError('create_index must be called before search')
        start = perf_counter()
        hits = []
        search_index = self.index
        if filters is not None and filters.active():
            eligible = np.array([key for key, document in self.documents.items() if filters.matches(document)], dtype=np.int64)
            search_index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.identity.dimensions))
            if len(eligible):
                vectors = np.stack([self.index.reconstruct(int(key)) for key in eligible])
                search_index.add_with_ids(vectors, eligible)
        if search_index.ntotal:
            scores, ids = search_index.search(query.reshape(1, -1), min(top_k, search_index.ntotal))
            hits = [SearchHit(document=self.documents[int(stable_id)], score=float(score))
                    for stable_id, score in zip(ids[0], scores[0], strict=True)]
            hits.sort(key=lambda hit: (-hit.score, hit.document.id))
        return SearchResult(backend=self.backend, namespace=self.identity.namespace, index_type=self.index_type,
                            latency_ms=(perf_counter() - start) * 1000, hits=hits)

    def inspect_document(self, stable_id: int) -> dict:
        if stable_id not in self.documents:
            raise ValueError('document ID not found')
        return {'document': self.documents[stable_id].model_dump(),
                'vector': self.index.reconstruct(stable_id).tolist()}

    def build_hnsw(self, vectors: np.ndarray, ids: np.ndarray, m: int, ef_construction: int) -> float:
        if set(ids.tolist()) != set(self.documents):
            raise ValueError('experiment vectors must match the loaded document IDs')
        # One native OpenMP runtime here; one thread also makes this small build reproducible.
        faiss.omp_set_num_threads(1)
        self.experiment = faiss.IndexHNSWFlat(self.identity.dimensions, m, faiss.METRIC_INNER_PRODUCT)
        self.experiment.hnsw.efConstruction = ef_construction
        self.experiment_ids = ids.copy()
        start = perf_counter()
        self.experiment.add(np.ascontiguousarray(vectors))
        return (perf_counter() - start) * 1000

    def search_hnsw(self, query_vector: np.ndarray, top_k: int, ef_search: int) -> SearchResult:
        query = self.validate_query(query_vector, top_k)
        self.experiment.hnsw.efSearch = ef_search
        start = perf_counter()
        scores, positions = self.experiment.search(query.reshape(1, -1), min(top_k, self.experiment.ntotal))
        hits = [SearchHit(document=self.documents[int(self.experiment_ids[int(position)])], score=float(score))
                for position, score in zip(positions[0], scores[0], strict=True) if position >= 0]
        return SearchResult(backend=self.backend, namespace=self.identity.namespace, index_type='IndexHNSWFlat',
                            latency_ms=(perf_counter() - start) * 1000, hits=hits)

    def list_ids(self) -> set[int]:
        return set(self.documents)

    def get_stats(self) -> StoreStats:
        size = None
        pointer = self.directory / 'active.json'
        if pointer.exists():
            folder = self.directory / json.loads(pointer.read_text())['generation']
            size = sum(path.stat().st_size for path in folder.iterdir() if path.is_file())
        return StoreStats(backend=self.backend, namespace=self.identity.namespace, count=len(self.documents),
                          dimensions=self.identity.dimensions, index_type=self.index_type, storage_bytes=size)
