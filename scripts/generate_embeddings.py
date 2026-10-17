"""Generate and persist local embeddings once, with aligned IDs and provenance."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(PROJECT_ROOT))

# Disable download telemetry and excess tokenizer worker processes for local runs.
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np

from app.core.config import EmbeddingConfig, load_embedding_config
from app.embeddings.embedding_service import EmbeddingService, validate_vectors
from app.core.artifacts import ARTIFACT_FILES, load_prepared_documents, validate_saved_artifacts, sha256_file


def generate_embeddings(dataset_dir: Path, output_dir: Path, config: EmbeddingConfig, *, offline: bool = False) -> tuple[dict[str, Any], bool]:
    dataset_dir, output_dir = dataset_dir.resolve(), output_dir.resolve()
    # Never put embedding artifacts over the prepared dataset's manifest.
    if output_dir == dataset_dir:
        raise ValueError('embedding output directory must differ from the dataset directory')
    rows, dataset_manifest = load_prepared_documents(dataset_dir)
    if output_dir.exists():
        # Existing output must be a complete, matching artifact set; never overwrite vectors implicitly.
        return validate_saved_artifacts(output_dir, rows, dataset_manifest, config), True

    count = len(rows)
    raw_bytes = count * config.dimensions * np.dtype(np.float32).itemsize
    print(f'Encoding {count:,} documents into {config.dimensions}-dimensional vectors.', flush=True)
    print(f'Vector array estimate: {raw_bytes / 1024**2:.2f} MiB (excludes model and inference memory).', flush=True)
    print(f'Model: {config.model_name}; device: cpu; batch size: {config.batch_size}', flush=True)
    print('Loading pinned model weights from cache, or downloading them on the first run...', flush=True)
    load_start = perf_counter()
    service = EmbeddingService(config, offline=offline)
    model_load_seconds = perf_counter() - load_start
    texts = [row['content'] for row in rows]
    counts = service.token_counts(texts)
    if len(counts) != count:
        raise ValueError('token count report does not align with the documents')
    truncated_count = sum(length > service.max_sequence_length for length in counts)
    print(f'Model input limit: {service.max_sequence_length} tokens including special tokens.', flush=True)
    print(f'{truncated_count:,} of {count:,} documents exceed this limit and will be truncated for embedding; saved text stays complete.', flush=True)
    start = perf_counter()
    vectors = service.encode_documents(texts)
    encoding_seconds = perf_counter() - start
    validate_vectors(vectors, count, config.dimensions)
    ids = np.array([row['id'] for row in rows], dtype=np.int64)
    token_report = [{'id': row['id'], 'tokens': length, 'truncated': length > service.max_sequence_length}
                    for row, length in zip(rows, counts, strict=True)]
    if sha256_file(dataset_dir / 'documents.jsonl') != dataset_manifest['documents_sha256']:
        raise ValueError('document sample changed during encoding; no artifacts were saved')

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.embedding-build-', dir=output_dir.parent))
    try:
        np.save(staging / 'vectors.npy', vectors, allow_pickle=False)
        np.save(staging / 'ids.npy', ids, allow_pickle=False)
        (staging / 'token_lengths.json').write_text(json.dumps(token_report, indent=2) + '\n', encoding='utf-8')
        manifest = {
            'schema_version': 1,
            'documents_sha256': dataset_manifest['documents_sha256'],
            'embedding_config': config.fingerprint(),
            'count': count,
            'dimensions': config.dimensions,
            'dtype': 'float32',
            'normalized': True,
            'metric': 'cosine',
            'model_max_sequence_length': service.max_sequence_length,
            'token_count_includes_special_tokens': True,
            'truncated_document_count': truncated_count,
            'model_load_seconds': model_load_seconds,
            'encoding_seconds': encoding_seconds,
            'vector_array_bytes': vectors.nbytes,
            'versions': {name: importlib.metadata.version(name)
                         for name in ('sentence-transformers', 'transformers', 'torch', 'numpy')},
            'file_sha256': {name: sha256_file(staging / name) for name in ARTIFACT_FILES},
        }
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        validate_saved_artifacts(staging, rows, dataset_manifest, config)
        # Publish the complete local artifact set together; a failed run leaves no partial target directory.
        # Another process's completed output must never be silently replaced.
        if output_dir.exists():
            raise ValueError('output directory appeared during encoding; existing output was left untouched')
        staging.rename(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=PROJECT_ROOT / 'datasets/processed/scifact_1000')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'config/settings.yaml')
    parser.add_argument('--offline', action='store_true', help='use cached model files only')
    args = parser.parse_args()
    output_dir = args.output_dir or args.dataset_dir / 'embeddings'
    try:
        config = load_embedding_config(args.config)
        manifest, reused = generate_embeddings(args.dataset_dir, output_dir, config, offline=args.offline)
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as error:
        parser.exit(1, f'Embedding generation failed: {error}\n')
    print(f"{'Reused validated' if reused else 'Saved'} embeddings: {manifest['count']:,} x {manifest['dimensions']} float32.")
    print(f"Truncated inputs: {manifest['truncated_document_count']:,}; model limit: {manifest['model_max_sequence_length']} tokens.")
    print(f'Artifacts: {output_dir.resolve()}')
    print('vectors.npy row i belongs to ids.npy[i] and document row i. No database ingestion was performed.')


if __name__ == '__main__':
    main()
