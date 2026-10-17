"""Metadata stays mutable without modifying source hashes or saved embeddings."""

import json

import numpy as np
import pytest

from app.core.artifacts import load_embedding_bundle, sha256_file
from app.core.config import EmbeddingConfig
from app.core.metadata import make_test_metadata, save_metadata_updates, validate_metadata_patches
from scripts import generate_embeddings
from scripts.prepare_dataset import prepare_dataset


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    source = tmp_path / 'corpus.jsonl'
    source.write_text(''.join(json.dumps({'_id': str(i), 'title': f'Title {i}', 'text': f'Abstract {i}',
                                         'metadata': {'original': 'preserved'}}) + '\n' for i in [101, 202, 303]))
    output = tmp_path / 'sample'
    prepare_dataset(source, output, count=3)
    config = EmbeddingConfig(model_name='fixture', revision='a' * 40, dimensions=3)
    class Encoder:
        max_sequence_length = 256
        def __init__(self, *args, **kwargs): pass
        def token_counts(self, texts): return [20] * len(texts)
        def encode_documents(self, texts): return np.eye(3, dtype=np.float32)
    monkeypatch.setattr(generate_embeddings, 'EmbeddingService', Encoder)
    generate_embeddings.generate_embeddings(output, output / 'embeddings', config)
    return output, config


def test_overlay_survives_reload_and_cached_embeddings_remain_valid(prepared):
    directory, config = prepared
    before = load_embedding_bundle(directory, config)
    hashes = [sha256_file(directory / 'documents.jsonl'), sha256_file(directory / 'embeddings/vectors.npy')]
    save_metadata_updates(directory, before.identity, set(before.ids.tolist()), {101: {'lab_group': 'alpha'}})
    save_metadata_updates(directory, before.identity, set(before.ids.tolist()), {101: {'lab_year': 2024}})
    after = load_embedding_bundle(directory, config)
    assert after.identity == before.identity
    assert after.documents[0].metadata == {'original': 'preserved', 'lab_group': 'alpha', 'lab_year': 2024}
    assert np.array_equal(after.vectors, before.vectors)
    assert hashes == [sha256_file(directory / 'documents.jsonl'), sha256_file(directory / 'embeddings/vectors.npy')]
    assert generate_embeddings.generate_embeddings(directory, directory / 'embeddings', config)[1] is True
    wrong = before.identity.model_copy(update={'vectors_sha256': 'c' * 64})
    with pytest.raises(ValueError, match='does not match'):
        save_metadata_updates(directory, wrong, set(before.ids.tolist()), {101: {'lab_group': 'beta'}})
    with pytest.raises(ValueError, match='unknown'):
        save_metadata_updates(directory, before.identity, set(before.ids.tolist()), {999: {'test': True}})


def test_test_labels_are_reproducible_diverse_and_marked_as_synthetic():
    ids = list(range(1000))
    patches = make_test_metadata(ids)
    assert patches == make_test_metadata(list(reversed(ids)))
    assert {patch['lab_group'] for patch in patches.values()} == {'alpha', 'beta', 'gamma', 'delta'}
    assert {patch['lab_year'] for patch in patches.values()} == {2022, 2023, 2024, 2025}
    assert {patch['lab_review_status'] for patch in patches.values()} == {'pending', 'reviewed', 'approved'}
    assert all(patch['lab_metadata_source'] == 'synthetic_filter_test' for patch in patches.values())


@pytest.mark.parametrize('patches', [{}, {1: {}}, {1: {'bad': float('nan')}},
                                    {1: {'nested': {2: 'non-string key'}}}, {'1': {'test': True}}])
def test_invalid_patches_are_rejected(patches):
    with pytest.raises((ValueError, TypeError)):
        validate_metadata_patches(patches)
