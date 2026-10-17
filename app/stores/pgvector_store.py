"""PostgreSQL exact cosine search; SQL identifiers and values are parameterized."""

from __future__ import annotations

from time import perf_counter
from typing import Sequence

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.core.config import BackendConfig
from app.core.interfaces import VectorStore
from app.core.metadata import validate_metadata_patches
from app.core.filters import MetadataFilter
from app.core.models import DatasetIdentity, Document, SearchHit, SearchResult, StoreStats


class PgVectorStore(VectorStore):
    backend = 'pgvector'
    index_type = 'exact sequential scan (no ANN index)'

    def __init__(self, identity: DatasetIdentity, config: BackendConfig) -> None:
        super().__init__(identity)
        self.connection = psycopg.connect(host=config.postgres_host, port=config.postgres_port,
                                         dbname=config.postgres_database, user=config.postgres_user,
                                         password=config.postgres_password, connect_timeout=10,
                                         autocommit=True, row_factory=dict_row)
        try:
            register_vector(self.connection)
        except Exception:
            self.connection.close()
            raise
        self.table = sql.Identifier('public', identity.namespace)

    def create_index(self) -> None:
        with self.connection.transaction():
            self.connection.execute('''CREATE TABLE IF NOT EXISTS public.vector_lab_datasets (
                namespace TEXT PRIMARY KEY,
                identity JSONB NOT NULL
            )''')
            self.connection.execute('''INSERT INTO public.vector_lab_datasets VALUES (%s, %s)
                ON CONFLICT (namespace) DO NOTHING''', (self.identity.namespace, Jsonb(self.identity.model_dump())))
            saved = self.connection.execute('SELECT identity FROM public.vector_lab_datasets WHERE namespace = %s',
                                            (self.identity.namespace,)).fetchone()['identity']
            if saved != self.identity.model_dump():
                raise ValueError('PostgreSQL dataset identity does not match the requested artifacts')
            self.connection.execute(sql.SQL('''CREATE TABLE IF NOT EXISTS {} (
                id BIGINT PRIMARY KEY,
                document_id TEXT NOT NULL,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB NOT NULL,
                source TEXT NOT NULL,
                embedding vector({}) NOT NULL
            )''').format(self.table, sql.Literal(self.identity.dimensions)))
            dimension = self.connection.execute('''SELECT atttypmod FROM pg_attribute
                WHERE attrelid = %s::regclass AND attname = 'embedding' AND NOT attisdropped''',
                (f'public.{self.identity.namespace}',)).fetchone()
            if not dimension or dimension['atttypmod'] != self.identity.dimensions:
                raise ValueError('PostgreSQL vector column dimensions do not match')

    def insert(self, documents: Sequence[Document], embeddings: np.ndarray) -> None:
        self.validate_insert(documents, embeddings)
        statement = sql.SQL('''INSERT INTO {} (id, document_id, title, abstract, content, metadata, source, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET document_id = EXCLUDED.document_id, title = EXCLUDED.title,
            abstract = EXCLUDED.abstract, content = EXCLUDED.content, metadata = EXCLUDED.metadata,
            source = EXCLUDED.source, embedding = EXCLUDED.embedding''').format(self.table)
        rows = [(d.id, d.document_id, d.title, d.abstract, d.content, Jsonb(d.metadata), d.source, vector)
                for d, vector in zip(documents, embeddings, strict=True)]
        with self.connection.transaction(), self.connection.cursor() as cursor:
            cursor.executemany(statement, rows)
        self.connection.execute(sql.SQL('ANALYZE {}').format(self.table))

    def update_metadata(self, patches: dict[int, dict]) -> None:
        validate_metadata_patches(patches)
        with self.connection.transaction(), self.connection.cursor() as cursor:
            existing = self.connection.execute(sql.SQL('SELECT id FROM {} WHERE id = ANY(%s) FOR UPDATE').format(self.table),
                                               (list(patches),)).fetchall()
            if {row['id'] for row in existing} != set(patches):
                raise ValueError('metadata update contains unknown document IDs')
            cursor.executemany(sql.SQL('UPDATE {} SET metadata = metadata || %s WHERE id = %s').format(self.table),
                               [(Jsonb(patch), stable_id) for stable_id, patch in patches.items()])

    def search(self, query_vector: np.ndarray, top_k: int = 10, filters: MetadataFilter | None = None) -> SearchResult:
        query = self.validate_query(query_vector, top_k)
        start = perf_counter()
        clauses, parameters = [], []
        if filters is not None:
            for field, values in [('lab_group', filters.groups), ('lab_review_status', filters.review_statuses)]:
                if values:
                    clauses.append(sql.SQL("jsonb_typeof(metadata -> {}) = 'string' AND metadata ->> {} = ANY(%s)").format(sql.Literal(field), sql.Literal(field)))
                    parameters.append(values)
            if filters.year_min is not None or filters.year_max is not None:
                clauses.append(sql.SQL("jsonb_typeof(metadata -> 'lab_year') = 'number' AND metadata ->> 'lab_year' ~ '^-?[0-9]+$'"))
                # JSONB comparisons avoid casts on malformed/non-numeric metadata.
                if filters.year_min is not None:
                    clauses.append(sql.SQL("metadata -> 'lab_year' >= to_jsonb(%s::integer)"))
                    parameters.append(filters.year_min)
                if filters.year_max is not None:
                    clauses.append(sql.SQL("metadata -> 'lab_year' <= to_jsonb(%s::integer)"))
                    parameters.append(filters.year_max)
        where = sql.SQL('WHERE ') + sql.SQL(' AND ').join(clauses) if clauses else sql.SQL('')
        with self.connection.transaction():
            # Preserve exact ground truth even if somebody later adds an ANN index in the UI.
            self.connection.execute('SET LOCAL enable_indexscan = off')
            self.connection.execute('SET LOCAL enable_indexonlyscan = off')
            self.connection.execute('SET LOCAL enable_bitmapscan = off')
            rows = self.connection.execute(sql.SQL('''SELECT id, document_id, title, abstract, content,
                metadata, source, 1 - (embedding <=> %s) AS score FROM {} {}
                ORDER BY embedding <=> %s, id LIMIT %s''').format(self.table, where), (query, *parameters, query, top_k)).fetchall()
        hits = [SearchHit(document=Document.model_validate(row), score=row['score']) for row in rows]
        return SearchResult(backend=self.backend, namespace=self.identity.namespace, index_type=self.index_type,
                            latency_ms=(perf_counter() - start) * 1000, hits=hits)

    def inspect_document(self, stable_id: int) -> dict:
        row = self.connection.execute(sql.SQL('SELECT * FROM {} WHERE id = %s').format(self.table), (stable_id,)).fetchone()
        if row is None:
            raise ValueError('document ID not found')
        vector = row.pop('embedding').to_numpy().tolist()
        return {'document': Document.model_validate(row).model_dump(), 'vector': vector}

    def list_ids(self) -> set[int]:
        return {row['id'] for row in self.connection.execute(sql.SQL('SELECT id FROM {}').format(self.table))}

    def get_stats(self) -> StoreStats:
        count = self.connection.execute(sql.SQL('SELECT count(*) AS count FROM {}').format(self.table)).fetchone()['count']
        size = self.connection.execute('SELECT pg_total_relation_size(%s::regclass) AS bytes',
                                       (f'public.{self.identity.namespace}',)).fetchone()['bytes']
        return StoreStats(backend=self.backend, namespace=self.identity.namespace, count=count,
                          dimensions=self.identity.dimensions, index_type=self.index_type, storage_bytes=size)

    def close(self) -> None:
        self.connection.close()
