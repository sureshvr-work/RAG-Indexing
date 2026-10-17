"""Application operations shared by the local API and dashboard."""

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from threading import RLock
from time import perf_counter

import numpy as np
import yaml

from app.core.artifacts import load_embedding_bundle
from app.core.config import PROJECT_ROOT, load_backend_config, load_embedding_config
from app.core.filters import MetadataFilter
from app.embeddings.embedding_service import EmbeddingService
from app.evaluation.exact import check_exact_result
from app.evaluation.metrics import SciFactJudgments, evaluate_search
from app.core.models import SearchResult
from app.services.tuning_service import HnswParameters, run_experiment
from app.stores.factory import BACKENDS, open_stores


class LabService:
    def __init__(self, dataset_dir: Path = PROJECT_ROOT / 'datasets/processed/scifact_1000'):
        self.dataset_dir = dataset_dir
        self.embedding_config = load_embedding_config()
        self.backend_config = load_backend_config()
        self.judgments = SciFactJudgments()
        self._encoder = None
        self._encoding_lock = RLock()
        self._experiment_lock = RLock()

    def bundle(self):
        return load_embedding_bundle(self.dataset_dir, self.embedding_config)

    def safe_error(self, error: Exception) -> str:
        return str(error).replace(self.backend_config.postgres_password, '[redacted]')

    def overview(self) -> dict:
        bundle = self.bundle()
        manifest = json.loads((self.dataset_dir / 'embeddings/manifest.json').read_text())
        statuses = []
        for backend in BACKENDS:
            try:
                with open_stores(bundle.identity, self.backend_config, [backend]) as stores:
                    store = stores[0]
                    store.create_index()
                    stats = store.get_stats()
                    ready = store.list_ids() == set(bundle.ids.tolist()) and stats.count == len(bundle.ids)
                    detail = {}
                    if backend == 'faiss':
                        folder = self.backend_config.faiss_directory / bundle.identity.namespace
                        detail = {'directory': str(folder), 'active_snapshot': json.loads((folder / 'active.json').read_text())['generation'] if (folder / 'active.json').exists() else None}
                    elif backend == 'pgvector':
                        detail = {'indexes': store.connection.execute('SELECT indexname, indexdef FROM pg_indexes WHERE schemaname=%s AND tablename=%s', ('public', bundle.identity.namespace)).fetchall()}
                    else:
                        info = store.client.get_collection(bundle.identity.namespace)
                        detail = {'hnsw_m': info.config.hnsw_config.m, 'indexed_vectors_count': info.indexed_vectors_count,
                                  'payload_indexes': {key: value.model_dump(mode='json') for key, value in info.payload_schema.items()}}
                    statuses.append({'backend': backend, 'ready': ready, 'stats': stats.model_dump(), 'details': detail})
            except Exception as error:
                statuses.append({'backend': backend, 'ready': False, 'error': self.safe_error(error)})
        settings = yaml.safe_load((PROJECT_ROOT / 'config/settings.yaml').read_text()).get('lab_ui', {})
        return {'namespace': bundle.identity.namespace, 'document_count': len(bundle.ids), 'dimensions': bundle.identity.dimensions,
                'model_name': self.embedding_config.model_name, 'model_token_limit': manifest['model_max_sequence_length'],
                'truncated_documents': manifest['truncated_document_count'], 'backends': statuses,
                'metadata_distributions': {key: dict(Counter(str(document.metadata.get(key, 'unset')) for document in bundle.documents))
                                           for key in ('lab_group', 'lab_year', 'lab_review_status')},
                'defaults': {'top_k': settings.get('top_k', 5), 'hnsw': HnswParameters(**settings.get('hnsw', {})).model_dump()}}

    def documents(self, filters: MetadataFilter, text: str, offset: int, limit: int) -> dict:
        bundle = self.bundle()
        needle = text.casefold().strip()
        selected = [document for document in bundle.documents if filters.matches(document)
                    and (not needle or needle in document.title.casefold() or needle in str(document.id))]
        rows = [{'id': document.id, 'title': document.title, **document.metadata} for document in selected[offset:offset + limit]]
        return {'total': len(selected), 'offset': offset, 'rows': rows, 'source': 'prepared documents with saved metadata updates'}

    def document(self, stable_id: int, backend: str) -> dict:
        bundle = self.bundle()
        positions = {int(key): i for i, key in enumerate(bundle.ids)}
        if stable_id not in positions:
            raise KeyError('document ID not found')
        if backend == 'prepared':
            data = {'document': bundle.documents[positions[stable_id]].model_dump(), 'vector': bundle.vectors[positions[stable_id]].tolist()}
        else:
            with open_stores(bundle.identity, self.backend_config, [backend]) as stores:
                stores[0].create_index()
                data = stores[0].inspect_document(stable_id)
        lengths = json.loads((self.dataset_dir / 'embeddings/token_lengths.json').read_text())
        token_info = next(row for row in lengths if row['id'] == stable_id)
        vector = np.asarray(data['vector'], dtype=np.float32)
        return {**data, 'backend': backend, 'dimensions': len(vector), 'vector_norm': float(np.linalg.norm(vector)),
                'token_info': token_info}

    def encode(self, query: str):
        with self._encoding_lock:
            if self._encoder is None:
                self._encoder = EmbeddingService(self.embedding_config, offline=True)
            tokens = self._encoder.token_counts([query])[0]
            start = perf_counter()
            vector = self._encoder.encode_query(query)
            timing = {'query_tokens': tokens, 'query_truncated': tokens > self._encoder.max_sequence_length,
                      'query_embedding_ms': (perf_counter() - start) * 1000}
        return vector, timing

    def evaluation_queries(self, split: str = 'test') -> dict:
        return self.judgments.catalog(self.bundle(), split)

    def search(self, query: str, top_k: int, backends: list[str], filters: MetadataFilter, query_id: str | None = None, qrels_split: str = 'test') -> dict:
        bundle = self.bundle()
        positions = [i for i, document in enumerate(bundle.documents) if filters.matches(document)]
        eligible = replace(bundle, documents=[bundle.documents[i] for i in positions], vectors=bundle.vectors[positions], ids=bundle.ids[positions])
        judgment = self.judgments.resolve(query, eligible, qrels_split, query_id)
        vector, timing = self.encode(query)
        results, errors = [], []
        for backend in backends:
            try:
                with open_stores(bundle.identity, self.backend_config, [backend]) as stores:
                    store = stores[0]
                    store.create_index()
                    if store.list_ids() != set(bundle.ids.tolist()):
                        raise ValueError('backend is not fully ingested; run scripts/ingest.py')
                    result = store.search(vector, top_k, filters)
                    error = check_exact_result(eligible, vector, result, top_k)
                    result.metrics = evaluate_search(eligible, vector, result, top_k, judgment)
                    results.append({**result.model_dump(), 'exact_check_passed': True, 'max_score_error': error})
            except Exception as error:
                errors.append({'backend': backend, 'error': self.safe_error(error)})
        return {'query': query, 'top_k': top_k, 'filters': filters.model_dump(), 'eligible_documents': len(positions),
                **timing, 'results': results, 'errors': errors,
                'evaluation': {key: value for key, value in judgment.items() if key != 'relevant_ids'}}

    def tune(self, query: str, backend: str, parameters: HnswParameters, top_k: int, repetitions: int, query_id: str | None = None, qrels_split: str = 'test') -> dict:
        bundle = self.bundle()
        judgment = self.judgments.resolve(query, bundle, qrels_split, query_id)
        vector, timing = self.encode(query)
        with self._experiment_lock:
            result = run_experiment(bundle, self.backend_config, backend, parameters, vector, top_k, repetitions)
        search_result = SearchResult.model_validate(result['result'])
        result['metrics'] = evaluate_search(bundle, vector, search_result, top_k, judgment)
        result['precision_at_k'] = result['metrics']['exact_neighbors']['precision_at_k']
        result['recall_at_k'] = result['metrics']['exact_neighbors']['recall_at_k']
        return {'query': query, **timing, **result}
