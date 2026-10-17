"""Validated configuration for the local embedding workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    revision: str
    dimensions: int = 384
    device: str = 'cpu'
    batch_size: int = 32
    cpu_threads: int = 4
    seed: int = 42
    normalize: bool = True
    cache_folder: str = '.cache/huggingface'

    def __post_init__(self) -> None:
        if not self.model_name or len(self.revision) != 40 or any(c not in '0123456789abcdef' for c in self.revision):
            raise ValueError('model_name and an immutable 40-character model revision are required')
        for name in ('dimensions', 'batch_size', 'cpu_threads'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.seed) is not int:
            raise ValueError('seed must be an integer')
        if self.device != 'cpu' or self.normalize is not True:
            raise ValueError('This workflow requires CPU execution and normalized embeddings')
        if not isinstance(self.cache_folder, str) or not self.cache_folder:
            raise ValueError('cache_folder must be a nonempty path')

    def fingerprint(self) -> dict:
        settings = asdict(self)
        # The location of identical cached weights does not change the vectors.
        settings.pop('cache_folder')
        return settings


def load_embedding_config(path: Path = PROJECT_ROOT / 'config/settings.yaml') -> EmbeddingConfig:
    settings = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(settings, dict) or not isinstance(settings.get('embeddings'), dict):
        raise ValueError('configuration must contain an embeddings mapping')
    return EmbeddingConfig(**settings['embeddings'])


@dataclass(frozen=True)
class BackendConfig:
    postgres_host: str
    postgres_port: int
    postgres_database: str
    postgres_user: str
    postgres_password: str = field(repr=False)
    qdrant_url: str = 'http://127.0.0.1:6333'
    faiss_directory: Path = PROJECT_ROOT / 'datasets/indexes/faiss'


def load_backend_config(env_path: Path = PROJECT_ROOT / '.env') -> BackendConfig:
    import os
    from dotenv import dotenv_values

    values = {**dotenv_values(env_path), **os.environ}
    password = values.get('POSTGRES_PASSWORD')
    if not password:
        raise ValueError('POSTGRES_PASSWORD is missing; configure the project .env file')
    return BackendConfig(
        postgres_host=values.get('POSTGRES_HOST') or '127.0.0.1',
        postgres_port=int(values.get('POSTGRES_PORT') or '5432'),
        postgres_database=values.get('POSTGRES_DB') or 'vector_search_lab',
        postgres_user=values.get('POSTGRES_USER') or 'rag_lab',
        postgres_password=password,
        qdrant_url=f"http://127.0.0.1:{int(values.get('QDRANT_HTTP_PORT') or '6333')}",
    )
