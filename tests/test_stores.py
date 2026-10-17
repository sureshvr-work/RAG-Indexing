"""Backend contract tests; remote tests use disposable, uniquely named stores."""

import os
import uuid

import numpy as np
import pytest
from psycopg import sql

from app.core.artifacts import EmbeddingBundle
from app.core.config import load_backend_config
from app.core.models import DatasetIdentity, Document
from app.evaluation.exact import check_exact_result
from app.stores.faiss_store import FaissVectorStore
from app.stores.pgvector_store import PgVectorStore
from app.stores.qdrant_store import QdrantVectorStore


@pytest.fixture
def bundle():
    identity = DatasetIdentity(namespace='test_lab_' + uuid.uuid4().hex, dimensions=3,
                               documents_sha256='a' * 64, vectors_sha256='b' * 64,
                               embedding_config={'fixture': True})
    documents = [Document(id=i, document_id=str(i), title=f'Title {i}', abstract=f'Abstract {i}',
                          content=f'Title {i}\n\nAbstract {i}', metadata={'tags': ['science', str(i)]},
                          source='test') for i in [101, 202, 303]]
    vectors = np.eye(3, dtype=np.float32)
    return EmbeddingBundle(documents, vectors, np.array([101, 202, 303], dtype=np.int64), identity)


@pytest.fixture(params=['faiss', 'pgvector', 'qdrant'])
def store_factory(request, bundle, tmp_path):
    backend = request.param
    if backend != 'faiss' and os.environ.get('RAG_LAB_INTEGRATION') != '1':
        pytest.skip('set RAG_LAB_INTEGRATION=1 to test running Docker databases')
    config = load_backend_config() if backend != 'faiss' else None
    clients = []

    def create(identity=None):
        identity = identity or bundle.identity
        if backend == 'faiss':
            store = FaissVectorStore(identity, tmp_path)
        elif backend == 'pgvector':
            store = PgVectorStore(identity, config)
        else:
            store = QdrantVectorStore(identity, config.qdrant_url)
        clients.append(store)
        return store

    yield create
    # Only the uniquely named test dataset is removed; the real sample is untouched.
    if clients and backend == 'pgvector':
        connection = clients[0].connection
        connection.execute(sql.SQL('DROP TABLE IF EXISTS {}').format(sql.Identifier('public', bundle.identity.namespace)))
        connection.execute('DELETE FROM public.vector_lab_datasets WHERE namespace = %s', (bundle.identity.namespace,))
    elif clients and backend == 'qdrant' and clients[0].client.collection_exists(bundle.identity.namespace):
        clients[0].client.delete_collection(bundle.identity.namespace)
    for store in clients:
        store.close()


def test_exact_search_upsert_and_persistence(bundle, store_factory):
    store = store_factory()
    store.create_index()
    assert store.search(np.array([1, 0, 0], dtype=np.float32), 5).hits == []
    store.insert(bundle.documents, bundle.vectors)
    store.insert(bundle.documents, bundle.vectors)
    assert store.get_stats().count == 3
    assert store.list_ids() == {101, 202, 303}
    query = np.array([3, 2, 1], dtype=np.float32)  # query normalization is part of the contract
    result = store.search(query, 2)
    assert [hit.document.id for hit in result.hits] == [101, 202]
    check_exact_result(bundle, query, result, 2)
    reopened = store_factory()
    reopened.create_index()
    check_exact_result(bundle, query, reopened.search(query, 10), 10)
    updated = bundle.documents[0].model_copy(update={'title': 'Updated title', 'metadata': {'updated': True}})
    reopened.insert([updated], bundle.vectors[:1])
    assert reopened.get_stats().count == 3
    assert reopened.search(bundle.vectors[0], 1).hits[0].document == updated


def test_invalid_inputs_do_not_change_store(bundle, store_factory):
    store = store_factory()
    store.create_index()
    store.insert(bundle.documents, bundle.vectors)
    for query, k in [(np.zeros(3), 1), (np.ones(2), 1), (np.full(3, np.nan), 1), (np.ones(3), 0)]:
        with pytest.raises(ValueError):
            store.search(query, k)
    with pytest.raises(ValueError):
        store.insert([bundle.documents[0]] * 2, bundle.vectors[:2])
    with pytest.raises(ValueError):
        store.insert(bundle.documents, np.zeros((3, 3), dtype=np.float32))
    assert store.get_stats().count == 3


def test_mismatched_identity_is_rejected(bundle, store_factory):
    store = store_factory()
    store.create_index()
    store.insert(bundle.documents, bundle.vectors)
    wrong = bundle.identity.model_copy(update={'vectors_sha256': 'c' * 64})
    with pytest.raises(ValueError, match='identity'):
        store_factory(wrong).create_index()


def test_baseline_allows_boundary_ties_and_rejects_bad_results(bundle, store_factory):
    store = store_factory()
    store.create_index()
    store.insert(bundle.documents, bundle.vectors)
    tied = np.array([1, 1, 0], dtype=np.float32)
    check_exact_result(bundle, tied, store.search(tied, 1), 1)
    result = store.search(np.array([3, 2, 1], dtype=np.float32), 2)
    with pytest.raises(ValueError, match='baseline'):
        check_exact_result(bundle, np.array([3, 2, 1]),
                           result.model_copy(update={'hits': [result.hits[0].model_copy(update={'score': 0.1}), result.hits[1]]}), 2)
    with pytest.raises(ValueError, match='duplicate'):
        check_exact_result(bundle, np.array([3, 2, 1]), result.model_copy(update={'hits': [result.hits[0]] * 2}), 2)


def test_metadata_only_update_preserves_vectors_content_and_other_keys(bundle, store_factory):
    store = store_factory()
    store.create_index()
    store.insert(bundle.documents, bundle.vectors)
    query = np.array([3, 2, 1], dtype=np.float32)
    before = store.search(query, 3)
    patch = {'lab_group': 'alpha', 'lab_year': 2024, 'lab_review_status': 'reviewed'}
    store.update_metadata({101: patch})
    store.update_metadata({101: {'lab_review_status': 'approved'}})
    reopened = store_factory()
    reopened.create_index()
    after = reopened.search(query, 3)
    assert store.get_stats().count == 3
    assert [h.document.id for h in after.hits] == [h.document.id for h in before.hits]
    assert np.allclose([h.score for h in after.hits], [h.score for h in before.hits], atol=1e-7)
    assert after.hits[0].document.metadata == {**bundle.documents[0].metadata, **patch, 'lab_review_status': 'approved'}
    assert after.hits[0].document.model_dump(exclude={'metadata'}) == bundle.documents[0].model_dump(exclude={'metadata'})
    assert after.hits[1].document == bundle.documents[1]
    with pytest.raises(ValueError, match='unknown'):
        store.update_metadata({101: {'lab_group': 'should_not_apply'}, 999: {'lab_group': 'unknown'}})
    assert store.search(query, 1).hits[0].document == after.hits[0].document


def test_filtering_happens_before_top_k_and_inspection_reads_real_vectors(bundle, store_factory):
    from dataclasses import replace
    from app.core.filters import MetadataFilter
    store = store_factory()
    store.create_index()
    store.insert(bundle.documents, bundle.vectors)
    patches = {101: {'lab_group': 'alpha', 'lab_year': 2022, 'lab_review_status': 'pending'},
               202: {'lab_group': 'beta', 'lab_year': 2024, 'lab_review_status': 'reviewed'},
               303: {'lab_group': 'beta', 'lab_year': 2025, 'lab_review_status': 'approved'}}
    store.update_metadata(patches)
    updated = [document.model_copy(update={'metadata': {**document.metadata, **patches[document.id]}}) for document in bundle.documents]
    query = np.array([3, 2, 1], dtype=np.float32)
    for filters, expected_ids in [(MetadataFilter(groups=['beta']), [202, 303]),
                                  (MetadataFilter(groups=['beta'], year_min=2025, review_statuses=['approved']), [303]),
                                  (MetadataFilter(groups=['no-match']), []),
                                  (MetadataFilter(year_max=2022), [101])]:
        eligible = [i for i, document in enumerate(updated) if filters.matches(document)]
        filtered_bundle = replace(bundle, documents=[updated[i] for i in eligible], ids=bundle.ids[eligible], vectors=bundle.vectors[eligible])
        result = store.search(query, 2, filters)
        assert [hit.document.id for hit in result.hits] == expected_ids
        check_exact_result(filtered_bundle, query, result, 2)
    record = store.inspect_document(202)
    assert record['document'] == updated[1].model_dump()
    assert np.allclose(record['vector'], bundle.vectors[1])
    with pytest.raises(ValueError, match='not found'):
        store.inspect_document(999)
