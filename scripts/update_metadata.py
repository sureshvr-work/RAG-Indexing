"""Save metadata patches and apply them to all three backends without re-embedding."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.artifacts import load_embedding_bundle
from app.core.config import load_backend_config, load_embedding_config
from app.core.metadata import (make_test_metadata, read_metadata_updates,
                               save_metadata_updates, validate_metadata_patches)
from app.stores.factory import open_stores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--generate-test-data', action='store_true', help='add synthetic labels to every document')
    mode.add_argument('--id', type=int, help='update a single document ID using --patch')
    mode.add_argument('--apply-saved', action='store_true', help='retry all saved patches after an interruption')
    parser.add_argument('--patch', help='JSON object; merges top-level metadata keys')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset-dir', type=Path, default=PROJECT_ROOT / 'datasets/processed/scifact_1000')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'config/settings.yaml')
    args = parser.parse_args()
    try:
        config = load_backend_config()
        bundle = load_embedding_bundle(args.dataset_dir, load_embedding_config(args.config))
        expected = set(bundle.ids.tolist())
        if args.id is not None:
            if not args.patch:
                raise ValueError('--id requires --patch with a JSON object')
            patches = {args.id: json.loads(args.patch)}
        else:
            if args.patch:
                raise ValueError('--patch requires --id')
            patches = (make_test_metadata(bundle.ids.tolist(), args.seed) if args.generate_test_data
                       else read_metadata_updates(args.dataset_dir, bundle.identity, expected))
        validate_metadata_patches(patches)
        if not set(patches).issubset(expected):
            raise ValueError('metadata update contains unknown document IDs')
        with open_stores(bundle.identity, config) as stores:
            # Check every backend before saving the update or mutating any metadata.
            for store in stores:
                store.create_index()
                if store.list_ids() != expected or store.get_stats().count != len(expected):
                    raise ValueError(f'{store.backend} is not fully ingested; run scripts/ingest.py first')
            # Persist the desired state first, so interrupted backend writes can be retried.
            save_metadata_updates(args.dataset_dir, bundle.identity, expected, patches)
            for store in stores:
                store.update_metadata(patches)
                print(f'{store.backend}: metadata updated for {len(patches):,} documents; count={store.get_stats().count}')
        updated = load_embedding_bundle(args.dataset_dir, load_embedding_config(args.config))
        distributions = {key: dict(sorted(Counter(str(document.metadata[key]) for document in updated.documents
                                                 if key in document.metadata).items()))
                         for key in ('lab_group', 'lab_year', 'lab_review_status')}
        print(json.dumps({'namespace': bundle.identity.namespace, 'distributions': distributions}, indent=2))
        print('Saved vectors and source documents were reused unchanged. Lab labels are synthetic test data.')
    except Exception as error:
        message = str(error)
        if 'config' in locals():
            message = message.replace(config.postgres_password, '[redacted]')
        parser.exit(1, f'Metadata update failed: {message}\n')


if __name__ == '__main__':
    main()
