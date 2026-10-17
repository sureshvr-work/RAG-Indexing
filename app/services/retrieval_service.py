"""Embed each query once and share that vector across the selected backends."""

from time import perf_counter
from typing import Sequence

from app.core.interfaces import VectorStore
from app.core.artifacts import EmbeddingBundle
from app.evaluation.metrics import SciFactJudgments, eligible_bundle, evaluate_search
from app.core.filters import MetadataFilter
from app.core.models import SearchResult
from app.embeddings.embedding_service import EmbeddingService


class RetrievalService:
    def __init__(self, embeddings: EmbeddingService, stores: Sequence[VectorStore], *, bundle: EmbeddingBundle | None = None) -> None:
        self.embeddings, self.stores = embeddings, stores
        self.bundle = bundle
        for store in stores:
            if store.identity.embedding_config != embeddings.config.fingerprint():
                raise ValueError('query model configuration does not match stored embeddings')

    def search(self, query: str, top_k: int = 10, filters: MetadataFilter | None = None, *, query_id: str | None = None, qrels_split: str = 'test') -> tuple[list[SearchResult], dict]:
        if not query.strip():
            raise ValueError('query cannot be empty')
        if type(top_k) is not int or top_k < 1:
            raise ValueError('top_k must be positive')
        tokens = self.embeddings.token_counts([query])[0]
        start = perf_counter()
        vector = self.embeddings.encode_query(query)
        embedding_ms = (perf_counter() - start) * 1000
        results = [store.search(vector, top_k, filters) if filters is not None else store.search(vector, top_k)
                   for store in self.stores]
        if self.bundle is not None:
            eligible = eligible_bundle(self.bundle, filters)
            judgment = SciFactJudgments().resolve(query, eligible, qrels_split, query_id)
            for result in results:
                result.metrics = evaluate_search(eligible, vector, result, top_k, judgment)
        return results, {'query_tokens': tokens, 'query_truncated': tokens > self.embeddings.max_sequence_length,
                         'query_embedding_ms': embedding_ms}
