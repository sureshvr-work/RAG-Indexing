"""Verify persisted IDs, document payloads, and exact top-k results against NumPy."""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from app.core.artifacts import load_embedding_bundle
from app.core.config import load_backend_config, load_embedding_config
from app.embeddings.embedding_service import EmbeddingService
from app.evaluation.exact import check_exact_result
from app.stores.factory import BACKENDS, open_stores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=PROJECT_ROOT / 'datasets/processed/scifact_1000')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'config/settings.yaml')
    parser.add_argument('--backends', nargs='+', choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args()
    try:
        if args.top_k < 1:
            raise ValueError('top_k must be positive')
        config = load_backend_config()
        embedding_config = load_embedding_config(args.config)
        bundle = load_embedding_bundle(args.dataset_dir, embedding_config)
        checks = []
        with open_stores(bundle.identity, config, args.backends) as stores:
            for store in stores:
                store.create_index()
                if store.list_ids() != set(bundle.ids.tolist()) or store.get_stats().count != len(bundle.ids):
                    raise ValueError(f'{store.backend}: ID set/count mismatch; run scripts/ingest.py first')
            embeddings = EmbeddingService(embedding_config, offline=True)
            queries = ['cancer immunotherapy', 'brain magnetic resonance imaging', 'antibiotic resistance']
            probes = [(query, embeddings.encode_query(query)) for query in queries]
            probes += [(f'saved document vector {int(bundle.ids[i])}', bundle.vectors[i])
                       for i in sorted({0, len(bundle.ids) // 2, len(bundle.ids) - 1})]
            for label, vector in probes:
                for store in stores:
                    result = store.search(vector, args.top_k)
                    error = check_exact_result(bundle, vector, result, args.top_k)
                    checks.append({'probe': label, 'backend': store.backend, 'passed': True,
                                   'max_score_error': error, 'ids': [hit.document.id for hit in result.hits]})
        print(json.dumps({'namespace': bundle.identity.namespace, 'document_count': len(bundle.ids),
                          'top_k': args.top_k, 'checks_passed': len(checks), 'checks': checks}, indent=2))
    except Exception as error:
        message = str(error)
        if 'config' in locals():
            message = message.replace(config.postgres_password, '[redacted]')
        parser.exit(1, f'Verification failed: {message}\n')


if __name__ == '__main__':
    main()
