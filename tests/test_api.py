"""HTTP validation and forwarding without external databases or model downloads."""

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.core.filters import MetadataFilter
from app.services.tuning_service import HnswParameters


@pytest.fixture
def client():
    class Lab:
        safe_error = staticmethod(lambda error: str(error))
        def search(self, query, top_k, backends, filters, query_id=None, qrels_split='test'):
            return {'query': query, 'top_k': top_k, 'backends': backends, 'filters': filters.model_dump()}
        def document(self, stable_id, backend):
            raise KeyError('not found')
        def tune(self, *args):
            pytest.fail('invalid tuning requests must never start a build')
    with TestClient(create_app(Lab())) as client:
        yield client


def test_http_filters_are_validated_and_forwarded(client):
    result = client.post('/search', json={'query': 'test', 'backends': ['faiss'], 'filters': {'groups': ['alpha'], 'year_min': 2024}})
    assert result.status_code == 200
    assert result.json()['filters']['groups'] == ['alpha']
    assert result.json()['filters']['year_min'] == 2024
    for data in [{'query': '   '}, {'query': 'test', 'top_k': 0}, {'query': 'test', 'backends': []},
                 {'query': 'test', 'backends': ['faiss', 'faiss']},
                 {'query': 'test', 'filters': {'year_min': 2025, 'year_max': 2022}}]:
        assert client.post('/search', json=data).status_code == 422
    assert client.get('/documents/999').status_code == 404
    assert client.get('/documents?year_min=2025&year_max=2022').status_code == 422


def test_tuning_requires_confirmation_and_compatible_parameters(client):
    for data in [{'query': 'test'}, {'query': 'test', 'confirmed': False},
                 {'query': 'test', 'confirmed': True, 'parameters': {'m': 64, 'ef_construction': 100}},
                 {'query': 'test', 'confirmed': True, 'repetitions': 100},
                 {'query': '   ', 'confirmed': True}]:
        assert client.post('/tune', json=data).status_code == 422
    with pytest.raises(ValueError):
        MetadataFilter(year_min=2025, year_max=2022)
    with pytest.raises(ValueError):
        HnswParameters(m=32, ef_construction=16)


def test_labeled_query_contract_and_catalog_endpoint():
    from app.evaluation.metrics import JudgmentQueryError
    class Lab:
        safe_error = staticmethod(lambda error: str(error))
        def evaluation_queries(self, split):
            return {'available': True, 'split': split, 'queries': []}
        def search(self, query, top_k, backends, filters, query_id, qrels_split):
            assert query_id == '49' and qrels_split == 'test'
            raise JudgmentQueryError('query text must match the selected labeled query')
    with TestClient(create_app(Lab())) as client:
        assert client.get('/evaluation/queries').json()['split'] == 'test'
        assert client.get('/evaluation/queries?split=invalid').status_code == 422
        assert client.post('/search', json={'query': 'changed', 'query_id': '49'}).status_code == 422
