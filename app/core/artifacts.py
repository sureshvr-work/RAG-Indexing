"""Validate prepared documents and saved vectors before ingestion or retrieval."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import EmbeddingConfig
from app.core.models import DatasetIdentity, Document
from app.core.metadata import apply_metadata_updates, read_metadata_updates
from app.embeddings.embedding_service import validate_vectors

ARTIFACT_FILES = ('vectors.npy', 'ids.npy', 'token_lengths.json')


def sha256_file(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_prepared_documents(dataset_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    document_path = dataset_dir / 'documents.jsonl'
    manifest_path = dataset_dir / 'manifest.json'
    if not document_path.is_file() or not manifest_path.is_file():
        raise ValueError('Prepared sample missing. First run: python scripts/prepare_dataset.py')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('unsupported dataset manifest schema')
    if sha256_file(document_path) != manifest.get('documents_sha256'):
        raise ValueError('sample hash does not match its manifest; prepare the sample again')
    rows = [json.loads(line) for line in document_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or len(rows) != manifest.get('selected_count'):
        raise ValueError('sample count does not match its manifest')
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('each document must be a JSON object')
        stable_id, content = row.get('id'), row.get('content')
        if type(stable_id) is not int or not 0 <= stable_id < 2**63 or stable_id in seen:
            raise ValueError('document IDs must be unique nonnegative signed 64-bit integers')
        if not isinstance(content, str) or not content.strip():
            raise ValueError('each document must have nonempty content')
        if row.get('document_id') is None or str(stable_id) != str(int(row['document_id'])):
            raise ValueError('numeric IDs must match the original document IDs')
        seen.add(stable_id)
    return rows, manifest


def validate_saved_artifacts(output_dir: Path, rows: list[dict[str, Any]], dataset_manifest: dict[str, Any], config: EmbeddingConfig) -> dict[str, Any]:
    manifest = json.loads((output_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('unsupported embedding manifest schema')
    if manifest.get('documents_sha256') != dataset_manifest['documents_sha256']:
        raise ValueError('existing embeddings belong to a different document sample; use a new output directory')
    if manifest.get('embedding_config') != config.fingerprint():
        raise ValueError('existing embeddings use different settings; use a new output directory')
    for name in ARTIFACT_FILES:
        if sha256_file(output_dir / name) != manifest.get('file_sha256', {}).get(name):
            raise ValueError(f'{name} is missing or has changed; use a new output directory to regenerate')
    vectors = np.load(output_dir / 'vectors.npy', allow_pickle=False)
    ids = np.load(output_dir / 'ids.npy', allow_pickle=False)
    validate_vectors(vectors, len(rows), config.dimensions)
    expected_ids = np.array([row['id'] for row in rows], dtype=np.int64)
    if ids.dtype != np.int64 or ids.shape != expected_ids.shape or not np.array_equal(ids, expected_ids):
        raise ValueError('saved vector IDs do not align with the document sample')
    return manifest


@dataclass(frozen=True)
class EmbeddingBundle:
    documents: list[Document]
    vectors: np.ndarray
    ids: np.ndarray
    identity: DatasetIdentity


def load_embedding_bundle(dataset_dir: Path, config: EmbeddingConfig) -> EmbeddingBundle:
    rows, dataset_manifest = load_prepared_documents(dataset_dir)
    output = dataset_dir / 'embeddings'
    manifest = validate_saved_artifacts(output, rows, dataset_manifest, config)
    fingerprint = {
        'documents_sha256': dataset_manifest['documents_sha256'],
        'vectors_sha256': manifest['file_sha256']['vectors.npy'],
        'embedding_config': config.fingerprint(),
    }
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    identity = DatasetIdentity(namespace=f'scifact_{digest[:32]}', dimensions=config.dimensions, **fingerprint)
    documents = [Document.model_validate(row) for row in rows]
    patches = read_metadata_updates(dataset_dir, identity, {document.id for document in documents})
    return EmbeddingBundle(
        documents=apply_metadata_updates(documents, patches),
        vectors=np.load(output / 'vectors.npy', allow_pickle=False),
        ids=np.load(output / 'ids.npy', allow_pickle=False),
        identity=identity,
    )
