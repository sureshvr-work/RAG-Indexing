"""Meaningful integrity checks that run without downloading model weights."""

import json
from pathlib import Path

import numpy as np
import pytest

from app.core.config import EmbeddingConfig
from app.embeddings.embedding_service import validate_vectors
from scripts import generate_embeddings as module
from scripts.prepare_dataset import prepare_dataset


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    source = tmp_path / 'source.jsonl'
    rows = [{'_id': str(i), 'title': f'Title {i}', 'text': f'Abstract {i}', 'metadata': {}} for i in [20, 10]]
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    output = tmp_path / 'sample'
    prepare_dataset(source, output, count=2)
    return output


@pytest.fixture
def config() -> EmbeddingConfig:
    return EmbeddingConfig(model_name='test-fixture', revision='a' * 40, dimensions=3)


class FixtureEncoder:
    max_sequence_length = 256

    def __init__(self, *args, **kwargs):
        pass

    def token_counts(self, texts):
        return [20, 300]

    def encode_documents(self, texts):
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)


def test_artifacts_preserve_alignment_and_reuse_without_model_load(dataset, config, tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'EmbeddingService', FixtureEncoder)
    output = tmp_path / 'embeddings'
    manifest, reused = module.generate_embeddings(dataset, output, config)
    assert reused is False
    assert manifest['truncated_document_count'] == 1
    assert np.load(output / 'ids.npy').tolist() == [10, 20]
    assert np.load(output / 'vectors.npy').shape == (2, 3)

    def must_not_load(*args, **kwargs):
        pytest.fail('cached artifact reuse should not load a model')

    monkeypatch.setattr(module, 'EmbeddingService', must_not_load)
    cached, reused = module.generate_embeddings(dataset, output, config, offline=True)
    assert reused is True
    assert cached['documents_sha256'] == manifest['documents_sha256']


def test_edited_sample_is_rejected_before_model_load(dataset, config, tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'EmbeddingService', lambda *a, **k: pytest.fail('model must not load'))
    documents = dataset / 'documents.jsonl'
    documents.write_text(documents.read_text().replace('Title 10', 'Changed title'))
    with pytest.raises(ValueError, match='sample hash'):
        module.generate_embeddings(dataset, tmp_path / 'embeddings', config)


def test_corrupt_saved_vectors_are_rejected(dataset, config, tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'EmbeddingService', FixtureEncoder)
    output = tmp_path / 'embeddings'
    module.generate_embeddings(dataset, output, config)
    np.save(output / 'vectors.npy', np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match='vectors.npy'):
        module.generate_embeddings(dataset, output, config)


def test_failed_encode_leaves_no_published_artifacts(dataset, config, tmp_path, monkeypatch):
    class BrokenEncoder(FixtureEncoder):
        def encode_documents(self, texts):
            raise RuntimeError('inference interrupted')

    monkeypatch.setattr(module, 'EmbeddingService', BrokenEncoder)
    output = tmp_path / 'embeddings'
    with pytest.raises(RuntimeError, match='inference interrupted'):
        module.generate_embeddings(dataset, output, config)
    assert not output.exists()


@pytest.mark.parametrize('vectors', [
    np.zeros((2, 3), dtype=np.float32),
    np.full((2, 3), np.nan, dtype=np.float32),
    np.ones((2, 2), dtype=np.float32),
    np.eye(2, 3, dtype=np.float64),
])
def test_invalid_vectors_are_rejected(vectors):
    with pytest.raises(ValueError):
        validate_vectors(vectors, count=2, dimensions=3)


@pytest.mark.parametrize('prompt, accepted', [(None, True), ('passage: ', False)])
def test_service_handles_library_default_prompts(config, tmp_path, monkeypatch, prompt, accepted):
    import sentence_transformers
    from app.embeddings.embedding_service import EmbeddingService

    class Model:
        prompts = {'query': None, 'document': prompt}
        max_seq_length = 256

        def __init__(self, *args, **kwargs):
            assert kwargs['revision'] == config.revision
            assert kwargs['local_files_only'] is True
            assert kwargs['device'] == 'cpu'

        def eval(self):
            return self

        def get_embedding_dimension(self):
            return 3

    monkeypatch.setattr(sentence_transformers, 'SentenceTransformer', Model)
    settings = EmbeddingConfig(model_name=config.model_name, revision=config.revision, dimensions=3,
                               cache_folder=str(tmp_path / 'cache'))
    if accepted:
        service = EmbeddingService(settings, offline=True)
        assert service.max_sequence_length == 256
    else:
        with pytest.raises(ValueError, match='without query/document prompts'):
            EmbeddingService(settings, offline=True)
