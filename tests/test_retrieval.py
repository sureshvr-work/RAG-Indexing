"""Ensure comparisons share one query vector and enforce the saved model profile."""

from types import SimpleNamespace

import numpy as np
import pytest

from app.services.retrieval_service import RetrievalService


class Encoder:
    config = SimpleNamespace(fingerprint=lambda: {'fixture': True})
    max_sequence_length = 256

    def __init__(self):
        self.calls = 0

    def token_counts(self, texts):
        return [300]

    def encode_query(self, query):
        self.calls += 1
        return np.array([1, 0, 0], dtype=np.float32)


class Store:
    identity = SimpleNamespace(embedding_config={'fixture': True})

    def search(self, vector, top_k):
        self.vector = vector
        self.top_k = top_k
        return 'fixture result'


def test_query_is_embedded_once_and_truncation_is_reported():
    encoder = Encoder()
    stores = [Store(), Store(), Store()]
    results, timing = RetrievalService(encoder, stores).search('long query', 5)
    assert encoder.calls == 1
    assert all(store.vector is stores[0].vector and store.top_k == 5 for store in stores)
    assert len(results) == 3
    assert timing['query_tokens'] == 300 and timing['query_truncated'] is True
    assert timing['query_embedding_ms'] >= 0


def test_model_mismatch_and_invalid_requests_are_rejected():
    encoder = Encoder()
    wrong = Store()
    wrong.identity = SimpleNamespace(embedding_config={'different': True})
    with pytest.raises(ValueError, match='configuration'):
        RetrievalService(encoder, [wrong])
    service = RetrievalService(encoder, [Store()])
    for query, k in [('   ', 5), ('query', 0)]:
        with pytest.raises(ValueError):
            service.search(query, k)
    assert encoder.calls == 0
