"""Per-query ranking metrics with explicit reference sets and cutoff semantics."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from app.core.artifacts import EmbeddingBundle
from app.core.config import PROJECT_ROOT
from app.core.filters import MetadataFilter
from app.core.models import SearchResult


class JudgmentQueryError(ValueError):
    """A requested label ID cannot be used for the supplied query."""


def eligible_bundle(bundle: EmbeddingBundle, filters: MetadataFilter | None) -> EmbeddingBundle:
    if filters is None or not filters.active():
        return bundle
    positions = [i for i, document in enumerate(bundle.documents) if filters.matches(document)]
    return replace(bundle, documents=[bundle.documents[i] for i in positions], vectors=bundle.vectors[positions], ids=bundle.ids[positions])


def ranking_metrics(result: SearchResult, relevant_ids: set[int] | None, top_k: int,
                    reference_count: int | None = None, reason: str | None = None) -> dict:
    """Precision@K uses requested K; recall uses positives in the stated scope."""
    if top_k < 1:
        raise ValueError('top_k must be positive')
    returned = [hit.document.id for hit in result.hits[:top_k]]
    if len(set(returned)) != len(returned):
        raise ValueError('metrics require unique result IDs')
    denominator = (len(relevant_ids) if relevant_ids is not None else 0) if reference_count is None else reference_count
    available = relevant_ids is not None and denominator > 0
    true_positives = len(set(returned) & relevant_ids) if relevant_ids is not None else None
    return {'available': available, 'precision_at_k': true_positives / top_k if available else None,
            'recall_at_k': true_positives / denominator if available else None,
            'matched_count': true_positives, 'returned_count': len(returned), 'requested_k': top_k,
            'recall_denominator': denominator if relevant_ids is not None else None,
            'reason': None if available else reason or 'No reference positives are available in this search scope.'}


def neighbor_metrics(bundle: EmbeddingBundle, query_vector: np.ndarray, result: SearchResult,
                     top_k: int, tolerance: float = 1e-5) -> dict:
    if not len(bundle.ids):
        return {**ranking_metrics(result, set(), top_k, reason='No documents match the search filters.'),
                'reference': 'filtered NumPy exact cosine top-k', 'eligible_documents': 0}
    vectors = bundle.vectors.astype(np.float64)
    vector = np.asarray(query_vector, dtype=np.float64)
    scores = (vectors @ vector) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(vector))
    count = min(top_k, len(bundle.ids))
    positions = np.lexsort((bundle.ids, -scores))[:count]
    cutoff = scores[positions[-1]]
    acceptable = set(bundle.ids[scores >= cutoff - tolerance].tolist())
    # Equivalent neighbors at the cutoff are accepted, as in exact-result verification.
    metrics = ranking_metrics(result, acceptable, top_k, reference_count=count)
    return {**metrics, 'reference': 'filtered NumPy exact cosine top-k', 'eligible_documents': len(bundle.ids),
            'reference_ids': bundle.ids[positions].tolist(), 'cutoff_tolerance': tolerance,
            'tie_policy': 'accept equivalent cutoff neighbors; recall denominator is min(K, eligible count)'}


class SciFactJudgments:
    def __init__(self, directory: Path = PROJECT_ROOT / 'datasets/raw/scifact'):
        self.directory = directory

    def load(self, split: str) -> tuple[dict[str, str], dict[str, set[int]]] | None:
        if split not in ('test', 'train'):
            raise ValueError('label split must be test or train')
        query_path, qrel_path = self.directory / 'queries.jsonl', self.directory / 'qrels' / f'{split}.tsv'
        if not query_path.is_file() or not qrel_path.is_file():
            return None
        queries = {str(row['_id']): row['text'] for row in
                   (json.loads(line) for line in query_path.read_text(encoding='utf-8').splitlines() if line.strip())}
        qrels: dict[str, set[int]] = {}
        with qrel_path.open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                if float(row['score']) > 0:
                    qrels.setdefault(row['query-id'], set()).add(int(row['corpus-id']))
        return queries, qrels

    def catalog(self, bundle: EmbeddingBundle, split: str = 'test') -> dict:
        loaded = self.load(split)
        if loaded is None:
            return {'available': False, 'split': split, 'queries': [], 'reason': 'SciFact queries/qrels files are missing.'}
        queries, qrels = loaded
        ids = set(bundle.ids.tolist())
        entries = [{'query_id': query_id, 'text': queries[query_id], 'relevant_in_sample': len(relevant & ids),
                    'relevant_in_full_corpus': len(relevant)}
                   for query_id, relevant in sorted(qrels.items(), key=lambda item: int(item[0]))
                   if query_id in queries and relevant & ids]
        return {'available': True, 'split': split, 'queries': entries, 'labeled_queries_in_split': len(qrels),
                'queries_with_relevant_documents_in_sample': len(entries)}

    def resolve(self, query: str, bundle: EmbeddingBundle, split: str = 'test', query_id: str | None = None) -> dict:
        loaded = self.load(split)
        base = {'reference': 'SciFact positive qrels', 'split': split, 'scope': 'searchable sample after metadata filters'}
        if loaded is None:
            return {**base, 'available': False, 'relevant_ids': None, 'reason': 'SciFact queries/qrels files are missing.'}
        queries, qrels = loaded
        if query_id is not None:
            if query_id not in qrels or query_id not in queries:
                raise JudgmentQueryError('query ID has no positive relevance judgments in the selected split')
            if queries[query_id].strip() != query.strip():
                raise JudgmentQueryError('query text must match the selected labeled query; edited text cannot reuse its judgments')
        else:
            matches = [key for key, text in queries.items() if key in qrels and text.strip() == query.strip()]
            if len(matches) == 1:
                query_id = matches[0]
            else:
                return {**base, 'available': False, 'relevant_ids': None,
                        'reason': 'No matching labeled query. Select a SciFact labeled query to measure relevance.'}
        relevant = qrels[query_id]
        in_scope = relevant & set(bundle.ids.tolist())
        return {**base, 'available': bool(in_scope), 'query_id': query_id, 'relevant_ids': in_scope,
                'relevant_in_full_corpus': len(relevant), 'relevant_in_search_scope': len(in_scope),
                'relevant_excluded_by_sample_or_filters': len(relevant - in_scope),
                'reason': None if in_scope else 'No judged relevant documents remain in the current sample/filter scope.'}


def relevance_metrics(judgment: dict, result: SearchResult, top_k: int) -> dict:
    relevant = judgment.get('relevant_ids')
    scores = ranking_metrics(result, relevant, top_k, reason=judgment.get('reason'))
    return {**{key: value for key, value in judgment.items() if key not in ('relevant_ids', 'available', 'reason')}, **scores}


def evaluate_search(bundle: EmbeddingBundle, vector: np.ndarray, result: SearchResult,
                    top_k: int, judgment: dict) -> dict:
    return {'exact_neighbors': neighbor_metrics(bundle, vector, result, top_k),
            'relevance': relevance_metrics(judgment, result, top_k)}
