"""Precision/recall denominators, missing judgments, filtered scope, and cutoff ties."""

import json

import numpy as np
import pytest

from app.core.artifacts import EmbeddingBundle
from app.core.filters import MetadataFilter
from app.core.models import DatasetIdentity, Document, SearchHit, SearchResult
from app.evaluation.metrics import (JudgmentQueryError, SciFactJudgments, eligible_bundle,
                                   evaluate_search, neighbor_metrics, ranking_metrics)


@pytest.fixture
def bundle():
    documents = [Document(id=key, document_id=str(key), title=str(key), abstract='Text', content='Text',
                          source='fixture', metadata={'lab_year': year}) for key, year in [(10, 2022), (20, 2024), (30, 2025)]]
    identity = DatasetIdentity(namespace='fixture', dimensions=3, documents_sha256='a' * 64,
                               vectors_sha256='b' * 64, embedding_config={})
    return EmbeddingBundle(documents, np.eye(3, dtype=np.float32), np.array([10, 20, 30], dtype=np.int64), identity)


def result(bundle, ids):
    mapping = {document.id: document for document in bundle.documents}
    return SearchResult(backend='fixture', namespace=bundle.identity.namespace, index_type='fixture', latency_ms=0,
                        hits=[SearchHit(document=mapping[key], score=1) for key in ids])


@pytest.fixture
def judgments(tmp_path):
    (tmp_path / 'qrels').mkdir()
    (tmp_path / 'queries.jsonl').write_text('\n'.join(json.dumps({'_id': key, 'text': text})
                                                   for key, text in [('1', 'Known query'), ('2', 'Outside sample')]))
    (tmp_path / 'qrels/test.tsv').write_text('query-id\tcorpus-id\tscore\n1\t10\t1\n1\t40\t1\n1\t20\t0\n2\t40\t1\n')
    return SciFactJudgments(tmp_path)


def test_precision_uses_requested_k_and_recall_uses_reference_positives(bundle):
    scores = ranking_metrics(result(bundle, [10, 30]), {10, 20}, 2)
    assert scores['precision_at_k'] == scores['recall_at_k'] == 0.5
    scores = ranking_metrics(result(bundle, [10]), {10}, 5)
    assert scores['precision_at_k'] == 0.2 and scores['recall_at_k'] == 1
    assert ranking_metrics(result(bundle, []), {10}, 5)['recall_at_k'] == 0


@pytest.mark.parametrize('reference', [None, set()])
def test_missing_or_empty_reference_is_unavailable_not_a_fabricated_zero(bundle, reference):
    scores = ranking_metrics(result(bundle, [10]), reference, 5)
    assert scores['available'] is False
    assert scores['precision_at_k'] is None and scores['recall_at_k'] is None


def test_neighbor_metrics_measure_overlap_and_accept_equivalent_cutoff_ties(bundle):
    scores = neighbor_metrics(bundle, np.array([3, 2, 1]), result(bundle, [10, 30]), 2)
    assert scores['precision_at_k'] == scores['recall_at_k'] == 0.5
    tied = neighbor_metrics(bundle, np.array([1, 1, 0]), result(bundle, [20]), 1)
    assert tied['recall_at_k'] == 1 and tied['precision_at_k'] == 1
    empty = eligible_bundle(bundle, MetadataFilter(groups=['no-match']))
    assert neighbor_metrics(empty, np.array([1, 0, 0]), result(bundle, []), 5)['available'] is False


def test_relevance_scope_excludes_unsearchable_and_filtered_documents(bundle, judgments):
    catalog = judgments.catalog(bundle)
    assert len(catalog['queries']) == 1
    assert catalog['queries'][0]['relevant_in_sample'] == 1
    assert catalog['queries'][0]['relevant_in_full_corpus'] == 2
    context = judgments.resolve('Known query', bundle)
    metrics = evaluate_search(bundle, np.array([1, 0, 0]), result(bundle, [10, 20]), 2, context)['relevance']
    assert metrics['precision_at_k'] == 0.5 and metrics['recall_at_k'] == 1
    assert metrics['relevant_in_full_corpus'] == 2 and metrics['recall_denominator'] == 1
    filtered = eligible_bundle(bundle, MetadataFilter(year_min=2024))
    missing = judgments.resolve('Known query', filtered, query_id='1')
    assert missing['available'] is False and missing['relevant_excluded_by_sample_or_filters'] == 2


def test_free_text_is_unjudged_and_edited_labeled_queries_are_rejected(bundle, judgments):
    assert judgments.resolve('my own query', bundle)['relevant_ids'] is None
    with pytest.raises(JudgmentQueryError, match='text must match'):
        judgments.resolve('Changed query', bundle, query_id='1')
    with pytest.raises(JudgmentQueryError, match='no positive'):
        judgments.resolve('Unknown query', bundle, query_id='999')


def test_missing_label_files_allow_search_without_relevance_metrics(bundle, tmp_path):
    judgments = SciFactJudgments(tmp_path)
    assert judgments.catalog(bundle)['available'] is False
    assert judgments.resolve('query', bundle)['relevant_ids'] is None


def test_duplicate_results_cannot_inflate_metrics(bundle):
    with pytest.raises(ValueError, match='unique'):
        ranking_metrics(result(bundle, [10, 10]), {10}, 2)
