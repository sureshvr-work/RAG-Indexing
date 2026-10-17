"""Local API for inspecting data, comparing exact search, and tuning HNSW."""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from app.core.config import PROJECT_ROOT
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.filters import MetadataFilter
from app.evaluation.metrics import JudgmentQueryError
from app.services.lab_service import LabService
from app.services.tuning_service import HnswParameters

Backend = Literal['faiss', 'pgvector', 'qdrant']


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    query: str = Field(min_length=1, max_length=10000)
    query_id: str | None = Field(default=None, max_length=100)
    qrels_split: Literal['test', 'train'] = 'test'
    top_k: int = Field(default=5, ge=1, le=100)
    backends: list[Backend] = Field(default_factory=lambda: ['faiss', 'pgvector', 'qdrant'], min_length=1, max_length=3)
    filters: MetadataFilter = Field(default_factory=MetadataFilter)

    @field_validator('query')
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError('query cannot be blank')
        return value

    @field_validator('backends')
    @classmethod
    def unique(cls, value):
        if len(set(value)) != len(value):
            raise ValueError('select unique backends')
        return value


class TuneRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    query: str = Field(min_length=1, max_length=10000)
    query_id: str | None = Field(default=None, max_length=100)
    qrels_split: Literal['test', 'train'] = 'test'
    backend: Backend = 'faiss'
    top_k: int = Field(default=10, ge=1, le=100)
    repetitions: int = Field(default=5, ge=2, le=20)
    parameters: HnswParameters = Field(default_factory=HnswParameters)
    confirmed: Literal[True]
    _nonempty = field_validator('query')(SearchRequest.nonempty.__func__)


def create_app(service=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.lab = service or LabService()
        yield
    app = FastAPI(title='RAG Indexing Lab', lifespan=lifespan)

    def call(operation):
        try:
            return operation(app.state.lab)
        except JudgmentQueryError as error:
            raise HTTPException(422, str(error))
        except KeyError:
            raise HTTPException(404, 'Document not found')
        except Exception as error:
            raise HTTPException(503, app.state.lab.safe_error(error))

    @app.get('/search-flow', include_in_schema=False)
    def search_flow():
        return FileResponse(PROJECT_ROOT / 'docs/search-flow.html', media_type='text/html')

    @app.get('/health')
    def health():
        return {'status': 'ok', 'application': 'rag-indexing-lab'}

    @app.get('/overview')
    def overview():
        return call(lambda lab: lab.overview())

    @app.get('/evaluation/queries')
    def evaluation_queries(split: Literal['test', 'train'] = 'test'):
        return call(lambda lab: lab.evaluation_queries(split))

    @app.get('/documents')
    def documents(text: str = Query(default='', max_length=500), offset: int = Query(default=0, ge=0), limit: int = Query(default=50, ge=1, le=200),
                  groups: list[str] = Query(default=[]), review_statuses: list[str] = Query(default=[]),
                  year_min: int | None = Query(default=None, ge=0, le=9999), year_max: int | None = Query(default=None, ge=0, le=9999)):
        try:
            filters = MetadataFilter(groups=groups, review_statuses=review_statuses, year_min=year_min, year_max=year_max)
        except ValueError as error:
            raise HTTPException(422, str(error))
        return call(lambda lab: lab.documents(filters, text, offset, limit))

    @app.get('/documents/{stable_id}')
    def document(stable_id: int, backend: Literal['prepared', 'faiss', 'pgvector', 'qdrant'] = 'prepared'):
        return call(lambda lab: lab.document(stable_id, backend))

    @app.post('/search')
    def search(request: SearchRequest):
        return call(lambda lab: lab.search(request.query, request.top_k, request.backends, request.filters, request.query_id, request.qrels_split))

    @app.post('/tune')
    def tune(request: TuneRequest):
        return call(lambda lab: lab.tune(request.query, request.backend, request.parameters, request.top_k, request.repetitions, request.query_id, request.qrels_split))

    return app


app = create_app()
