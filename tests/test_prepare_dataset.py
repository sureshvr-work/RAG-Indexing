"""Checks for reproducibility and stable document-to-vector identity."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_dataset import prepare_dataset


def write_source(path: Path, rows: list[dict]) -> None:
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def source_rows() -> list[dict]:
    return [{'_id': str(i), 'title': f'Title {i}', 'text': f'Abstract {i}', 'metadata': {}} for i in range(1, 21)]


def test_same_seed_preserves_sample_and_order_when_source_reordered(tmp_path: Path) -> None:
    source = tmp_path / 'corpus.jsonl'
    write_source(source, source_rows())
    first = prepare_dataset(source, tmp_path / 'first', count=10, seed=42)
    write_source(source, list(reversed(source_rows())))
    second = prepare_dataset(source, tmp_path / 'second', count=10, seed=42)
    assert first['documents_sha256'] == second['documents_sha256']
    rows = [json.loads(line) for line in (tmp_path / 'first/documents.jsonl').read_text().splitlines()]
    assert [row['id'] for row in rows] == sorted(row['id'] for row in rows)


def test_saved_documents_preserve_source_text_ids_and_manifest_hash(tmp_path: Path) -> None:
    source = tmp_path / 'corpus.jsonl'
    write_source(source, [{'_id': '4983', 'title': ' A title ', 'text': ' An abstract ', 'metadata': {'original': True}}])
    manifest = prepare_dataset(source, tmp_path / 'out', count=1)
    output = tmp_path / 'out/documents.jsonl'
    row = json.loads(output.read_text())
    assert row['id'] == 4983
    assert row['document_id'] == '4983'
    assert row['title'] == ' A title '
    assert row['abstract'] == ' An abstract '
    assert row['content'] == 'A title\n\nAn abstract'
    assert row['metadata'] == {'original': True}
    assert manifest['documents_sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert manifest['selected_count'] == 1


@pytest.mark.parametrize('count', [0, 21])
def test_invalid_sample_size_does_not_write_output(tmp_path: Path, count: int) -> None:
    source = tmp_path / 'corpus.jsonl'
    write_source(source, source_rows())
    with pytest.raises(ValueError, match='count must be between'):
        prepare_dataset(source, tmp_path / 'out', count=count)
    assert not (tmp_path / 'out').exists()


def test_duplicate_source_ids_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / 'corpus.jsonl'
    write_source(source, [source_rows()[0], source_rows()[0]])
    with pytest.raises(ValueError, match='duplicate source ID'):
        prepare_dataset(source, tmp_path / 'out', count=1)
