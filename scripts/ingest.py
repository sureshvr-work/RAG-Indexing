"""Load the prepared documents and saved vectors into selected local stores."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.artifacts import load_embedding_bundle
from app.core.config import load_backend_config, load_embedding_config
from app.services.ingestion_service import ingest_bundle
from app.stores.factory import BACKENDS, open_stores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=PROJECT_ROOT / 'datasets/processed/scifact_1000')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'config/settings.yaml')
    parser.add_argument('--backends', nargs='+', choices=BACKENDS, default=list(BACKENDS))
    args = parser.parse_args()
    try:
        config = load_backend_config()
        bundle = load_embedding_bundle(args.dataset_dir, load_embedding_config(args.config))
        print(f'Dataset: {bundle.identity.namespace}', flush=True)
        print(f'Loading {len(bundle.documents):,} documents with the existing {bundle.identity.dimensions}-dimensional vectors.', flush=True)
        with open_stores(bundle.identity, config, args.backends) as stores:
            stats = ingest_bundle(bundle, stores)
        for result in stats:
            print(json.dumps(result.model_dump()))
    except Exception as error:
        message = str(error)
        if 'config' in locals():
            message = message.replace(config.postgres_password, '[redacted]')
        parser.exit(1, f'Ingestion failed: {message}\n')


if __name__ == '__main__':
    main()
