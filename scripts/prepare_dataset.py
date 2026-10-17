"""Prepare a reproducible SciFact sample: one title + abstract per document."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_documents(source: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    seen: set[int] = set()
    with source.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('expected a JSON object')
                source_id = row['_id']
                if not isinstance(source_id, str) or not source_id.isascii() or not source_id.isdecimal():
                    raise ValueError('expected a numeric string source ID')
                stable_id = int(source_id)
                if not 0 <= stable_id < 2**63:
                    raise ValueError('source ID is outside the signed 64-bit range')
                if stable_id in seen:
                    raise ValueError(f'duplicate source ID: {source_id}')
                title, abstract = row['title'], row['text']
                if not isinstance(title, str) or not isinstance(abstract, str):
                    raise ValueError('title and abstract must be strings')
                if not title.strip() or not abstract.strip():
                    raise ValueError('title and abstract must be nonempty')
                metadata = row.get('metadata', {})
                if not isinstance(metadata, dict):
                    raise ValueError('metadata must be a JSON object')
            except (KeyError, ValueError, TypeError) as error:
                raise ValueError(f'{source}:{line_number}: {error}') from error
            seen.add(stable_id)
            documents.append({
                'id': stable_id,
                'document_id': source_id,
                'title': title,
                'abstract': abstract,
                'content': f'{title.strip()}\n\n{abstract.strip()}',
                'metadata': metadata,
                'source': 'scifact',
            })
    if not documents:
        raise ValueError(f'{source} contains no documents')
    return sorted(documents, key=lambda document: document['id'])


def select_documents(documents: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if not 1 <= count <= len(documents):
        raise ValueError(f'count must be between 1 and {len(documents)}; received {count}')
    # Sort before sampling so changing source file order cannot change the sample.
    ordered = sorted(documents, key=lambda document: document['id'])
    selected = random.Random(seed).sample(ordered, count)
    return sorted(selected, key=lambda document: document['id'])


def write_atomic(path: Path, content: str) -> None:
    """Replace a local output file only after its complete contents are written."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_dataset(source: Path, output_dir: Path, count: int = 1000, seed: int = 42) -> dict[str, Any]:
    source = source.resolve()
    output_dir = output_dir.resolve()
    output_path = output_dir / 'documents.jsonl'
    manifest_path = output_dir / 'manifest.json'
    if source in (output_path, manifest_path):
        raise ValueError('output paths must not overwrite the source corpus')
    source_sha256 = sha256_file(source)
    documents = load_documents(source)
    selected = select_documents(documents, count, seed)
    if sha256_file(source) != source_sha256:
        raise ValueError('source corpus changed during preparation; retry with an unchanged source')
    serialized = ''.join(json.dumps(document, ensure_ascii=False, sort_keys=True) + '\n' for document in selected)
    manifest = {
        'schema_version': 1,
        'dataset': 'scifact',
        'source_file': str(source),
        'source_url': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip',
        'corpus_license': 'ODC-By 1.0 (as documented by the original SciFact authors)',
        'license_source': 'https://github.com/allenai/scifact/blob/master/LICENSE.md',
        'source_sha256': source_sha256,
        'source_count': len(documents),
        'selected_count': len(selected),
        'seed': seed,
        'sampling': 'Python random.Random(seed).sample over ascending numeric source IDs',
        'document_order': 'ascending numeric source ID',
        'embedding_input': 'stripped title + two newlines + stripped abstract',
        'chunking': False,
        'documents_file': output_path.name,
        'documents_sha256': hashlib.sha256(serialized.encode('utf-8')).hexdigest(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(output_path, serialized)
    write_atomic(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=PROJECT_ROOT / 'datasets/raw/scifact/corpus.jsonl')
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or PROJECT_ROOT / f'datasets/processed/scifact_{args.count}'
    try:
        manifest = prepare_dataset(args.source, output_dir, args.count, args.seed)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Dataset preparation failed: {error}\n')
    print(f"Selected {manifest['selected_count']:,} of {manifest['source_count']:,} documents (seed={args.seed}).")
    print(f'Documents: {output_dir.resolve() / "documents.jsonl"}')
    print(f'Manifest:  {output_dir.resolve() / "manifest.json"}')
    print('Each document contains its original source ID, title, abstract, and combined content.')
    print('Next step: generate one embedding per document. No embeddings were generated here.')


if __name__ == '__main__':
    main()
