"""Temporary HNSW experiments; the exact baseline stores are preserved."""

import time
import uuid
from contextlib import contextmanager
from statistics import median

import numpy as np
from psycopg import sql
from pydantic import BaseModel, ConfigDict, Field, model_validator
from qdrant_client import models

from app.core.artifacts import EmbeddingBundle
from app.core.config import BackendConfig
from app.core.models import Document, SearchHit, SearchResult
from app.stores.factory import open_stores
from app.stores.pgvector_store import PgVectorStore
from app.stores.qdrant_store import QdrantVectorStore


class HnswParameters(BaseModel):
    model_config = ConfigDict(extra='forbid')
    m: int = Field(default=16, ge=2, le=64)
    ef_construction: int = Field(default=100, ge=4, le=512)
    ef_search: int = Field(default=64, ge=1, le=1000)

    @model_validator(mode='after')
    def compatible(self):
        if self.ef_construction < 2 * self.m:
            raise ValueError('ef_construction must be at least twice M for cross-backend compatibility')
        return self


@contextmanager
def experiment_search(bundle: EmbeddingBundle, config: BackendConfig, backend: str, parameters: HnswParameters):
    if backend == 'faiss':
        with open_stores(bundle.identity, config, ['faiss']) as stores:
            store = stores[0]
            store.create_index()
            build_ms = store.build_hnsw(bundle.vectors, bundle.ids, parameters.m, parameters.ef_construction)
            yield (lambda vector, k: store.search_hnsw(vector, k, parameters.ef_search)), build_ms, {'threads': 1, 'temporary': 'in-memory index'}
        return
    identity = bundle.identity.model_copy(update={'namespace': 'experiment_' + uuid.uuid4().hex})
    if backend == 'pgvector':
        store = PgVectorStore(identity, config)
        try:
            store.create_index()
            store.insert(bundle.documents, bundle.vectors)
            start = time.perf_counter()
            store.connection.execute(sql.SQL('CREATE INDEX {} ON {} USING hnsw (embedding vector_cosine_ops) WITH (m = {}, ef_construction = {})').format(
                sql.Identifier(identity.namespace + '_hnsw'), store.table, sql.Literal(parameters.m), sql.Literal(parameters.ef_construction)))
            build_ms = (time.perf_counter() - start) * 1000
            plan = None
            def search(vector, k):
                nonlocal plan
                query = store.validate_query(vector, k)
                start = time.perf_counter()
                with store.connection.transaction():
                    store.connection.execute('SET LOCAL enable_seqscan = off')
                    store.connection.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(parameters.ef_search),))
                    statement = sql.SQL('SELECT id, document_id, title, abstract, content, metadata, source, 1 - (embedding <=> %s) AS score FROM {} ORDER BY embedding <=> %s LIMIT %s').format(store.table)
                    rows = store.connection.execute(statement, (query, query, k)).fetchall()
                    elapsed = (time.perf_counter() - start) * 1000
                    if plan is None:
                        plan = store.connection.execute(sql.SQL('EXPLAIN (FORMAT JSON) ') + statement, (query, query, k)).fetchone()['QUERY PLAN']
                        def index_used(node):
                            return node.get('Index Name') == identity.namespace + '_hnsw' or any(index_used(child) for child in node.get('Plans', []))
                        if not index_used(plan[0]['Plan']):
                            raise ValueError('PostgreSQL did not use the experiment HNSW index')
                return SearchResult(backend=backend, namespace=identity.namespace, index_type='HNSW', latency_ms=elapsed,
                                    hits=[SearchHit(document=Document.model_validate(row), score=row['score']) for row in rows])
            details = {'temporary': identity.namespace, 'query_plan': lambda: plan}
            yield search, build_ms, details
        finally:
            try:
                store.connection.execute(sql.SQL('DROP TABLE IF EXISTS {}').format(store.table))
                store.connection.execute('DELETE FROM public.vector_lab_datasets WHERE namespace = %s', (identity.namespace,))
            finally:
                store.close()
        return
    if backend != 'qdrant':
        raise ValueError('unsupported experiment backend')
    store = QdrantVectorStore(identity, config.qdrant_url)
    try:
        store.client.create_collection(collection_name=identity.namespace,
                                       vectors_config=models.VectorParams(size=identity.dimensions, distance=models.Distance.COSINE),
                                       hnsw_config=models.HnswConfigDiff(m=parameters.m, ef_construct=parameters.ef_construction, full_scan_threshold=10),
                                       optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1),
                                       metadata={'vector_lab_identity': identity.model_dump()})
        start = time.perf_counter()
        store.insert(bundle.documents, bundle.vectors)
        deadline = time.monotonic() + 45
        while True:
            info = store.client.get_collection(identity.namespace)
            if (info.indexed_vectors_count or 0) >= len(bundle.ids) and info.status == models.CollectionStatus.GREEN:
                break
            if time.monotonic() >= deadline:
                raise ValueError('Qdrant HNSW construction was not ready within 45 seconds')
            time.sleep(0.25)
        build_ms = (time.perf_counter() - start) * 1000
        def search(vector, k):
            query = store.validate_query(vector, k)
            start = time.perf_counter()
            points = store.client.query_points(collection_name=identity.namespace, query=query.tolist(), limit=k,
                                               search_params=models.SearchParams(exact=False, hnsw_ef=parameters.ef_search),
                                               with_payload=True, with_vectors=False).points
            hits = [SearchHit(document=Document.model_validate(point.payload), score=point.score) for point in points]
            return SearchResult(backend=backend, namespace=identity.namespace, index_type='HNSW', latency_ms=(time.perf_counter() - start) * 1000, hits=hits)
        yield search, build_ms, {'temporary': identity.namespace, 'indexed_vectors_count': info.indexed_vectors_count,
                                'build_timing_scope': 'point upload through indexed/green readiness'}
    finally:
        try:
            if store.client.collection_exists(identity.namespace):
                store.client.delete_collection(identity.namespace)
        finally:
            store.close()


def run_experiment(bundle: EmbeddingBundle, config: BackendConfig, backend: str, parameters: HnswParameters,
                   query_vector: np.ndarray, top_k: int = 10, repetitions: int = 5) -> dict:
    # Measured recall concerns nearest-neighbor accuracy over saved vectors, not semantic relevance.
    normalized = bundle.vectors.astype(np.float64)
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True)
    query = query_vector.astype(np.float64) / np.linalg.norm(query_vector)
    scores = normalized @ query
    expected_count = min(top_k, len(bundle.ids))
    positions = np.lexsort((bundle.ids, -scores))[:expected_count]
    expected_ids = set(bundle.ids[positions].tolist())
    with experiment_search(bundle, config, backend, parameters) as (search, build_ms, details):
        search(query_vector, top_k)  # one warmup, excluded from recorded samples
        samples = []
        for _ in range(repetitions):
            result = search(query_vector, top_k)
            samples.append(result.latency_ms)
        observed = {hit.document.id for hit in result.hits}
        if len(observed) != len(result.hits) or not observed.issubset(set(bundle.ids.tolist())):
            raise ValueError('experiment returned duplicate or unknown IDs')
        detail_values = {key: value() if callable(value) else value for key, value in details.items()}
        return {'backend': backend, 'parameters': parameters.model_dump(), 'document_count': len(bundle.ids),
                'top_k': top_k, 'build_ms': build_ms, 'repetitions': repetitions, 'warmup_queries': 1,
                'latency_samples_ms': samples, 'p50_ms': median(samples), 'p95_ms': float(np.percentile(samples, 95)),
                'recall_at_k': len(observed & expected_ids) / expected_count,
                'ground_truth_ids': sorted(expected_ids), 'result': result.model_dump(), 'details': detail_values,
                'timing_scope': 'client search call including transport and result decoding; excludes embedding and setup'}
