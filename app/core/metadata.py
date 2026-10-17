"""Mutable metadata overlays kept separate from immutable embedding artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

from app.core.models import DatasetIdentity, Document

OVERLAY_FILE = 'metadata_updates.json'


def validate_metadata_patches(patches: dict[int, dict]) -> None:
    if not isinstance(patches, dict) or not patches:
        raise ValueError('provide at least one document metadata patch')
    for stable_id, patch in patches.items():
        if type(stable_id) is not int or not 0 <= stable_id < 2**63:
            raise ValueError('metadata patch IDs must be nonnegative signed 64-bit integers')
        if not isinstance(patch, dict) or not patch or any(not isinstance(key, str) or not key for key in patch):
            raise ValueError('metadata patches must be nonempty objects with nonempty string keys')
        if 'lab_year' in patch and type(patch['lab_year']) is not int:
            raise ValueError('lab_year must be an integer')
        for key in ('lab_group', 'lab_review_status'):
            if key in patch and not isinstance(patch[key], str):
                raise ValueError(f'{key} must be a string')
    # Enforce JSON types recursively (including string keys) without silent coercion.
    def validate_json(value):
        if value is None or type(value) in (str, bool, int, float):
            return
        if isinstance(value, list):
            for item in value:
                validate_json(item)
        elif isinstance(value, dict) and all(isinstance(key, str) for key in value):
            for item in value.values():
                validate_json(item)
        else:
            raise ValueError('metadata must contain only JSON values and string object keys')
    for patch in patches.values():
        validate_json(patch)
    json.dumps(list(patches.values()), allow_nan=False)


def read_metadata_updates(dataset_dir: Path, identity: DatasetIdentity, ids: set[int]) -> dict[int, dict]:
    path = dataset_dir / OVERLAY_FILE
    if not path.exists():
        return {}
    saved = json.loads(path.read_text(encoding='utf-8'))
    if saved.get('schema_version') != 1 or saved.get('identity') != identity.model_dump():
        raise ValueError('metadata overlay does not match the saved embedding artifacts')
    rows = saved.get('patches')
    if not isinstance(rows, dict) or any(not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key for key in rows):
        raise ValueError('metadata overlay requires canonical numeric document IDs')
    patches = {int(key): patch for key, patch in rows.items()}
    if patches:
        validate_metadata_patches(patches)
    if not set(patches).issubset(ids):
        raise ValueError('metadata overlay contains IDs outside this dataset')
    return patches


def apply_metadata_updates(documents: Sequence[Document], patches: dict[int, dict]) -> list[Document]:
    return [document.model_copy(update={'metadata': {**document.metadata, **patches.get(document.id, {})}})
            for document in documents]


def save_metadata_updates(dataset_dir: Path, identity: DatasetIdentity, ids: set[int],
                          patches: dict[int, dict]) -> None:
    validate_metadata_patches(patches)
    if not set(patches).issubset(ids):
        raise ValueError('metadata update contains unknown document IDs')
    saved = read_metadata_updates(dataset_dir, identity, ids)
    for stable_id, patch in patches.items():
        saved[stable_id] = {**saved.get(stable_id, {}), **patch}
    payload = {'schema_version': 1, 'identity': identity.model_dump(),
               'patches': {str(key): saved[key] for key in sorted(saved)}}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=dataset_dir, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, dataset_dir / OVERLAY_FILE)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def make_test_metadata(ids: Sequence[int], seed: int = 42) -> dict[int, dict]:
    """Stable synthetic labels unrelated to scientific content or publication dates."""
    patches = {}
    for stable_id in ids:
        digest = hashlib.sha256(f'{seed}:{int(stable_id)}'.encode()).digest()
        patches[int(stable_id)] = {
            'lab_metadata_source': 'synthetic_filter_test',
            'lab_group': ('alpha', 'beta', 'gamma', 'delta')[digest[0] % 4],
            'lab_year': 2022 + digest[1] % 4,
            'lab_review_status': ('pending', 'reviewed', 'approved')[digest[2] % 3],
        }
    return patches
