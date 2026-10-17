"""Run a query through the same local embedding model and selected exact stores."""

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
from app.services.retrieval_service import RetrievalService
from app.stores.factory import BACKENDS, open_stores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query')
    parser.add_argument('--query-id', help='SciFact labeled query ID; text must match its saved query')
    parser.add_argument('--qrels-split', choices=['test', 'train'], default='test')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--backends', nargs='+', choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument('--dataset-dir', type=Path, default=PROJECT_ROOT / 'datasets/processed/scifact_1000')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'config/settings.yaml')
    parser.add_argument('--json', action='store_true', help='include full documents, metadata, and timing as JSON')
    args = parser.parse_args()
    try:
        if not args.query.strip() or args.top_k < 1:
            raise ValueError('provide a nonempty query and positive top_k')
        config = load_backend_config()
        embedding_config = load_embedding_config(args.config)
        bundle = load_embedding_bundle(args.dataset_dir, embedding_config)
        with open_stores(bundle.identity, config, args.backends) as stores:
            for store in stores:
                store.create_index()
                if store.list_ids() != set(bundle.ids.tolist()):
                    raise ValueError(f'{store.backend} is not fully ingested; first run python scripts/ingest.py')
            embeddings = EmbeddingService(embedding_config, offline=True)
            results, timing = RetrievalService(embeddings, stores, bundle=bundle).search(args.query, args.top_k, query_id=args.query_id, qrels_split=args.qrels_split)
        if args.json:
            print(json.dumps({'query': args.query, **timing, 'results': [r.model_dump() for r in results]}, indent=2))
        else:
            print(f"Query embedded once in {timing['query_embedding_ms']:.2f} ms; {timing['query_tokens']} input tokens.")
            if timing['query_truncated']:
                print('The query exceeded the model input limit and was truncated.')
            for result in results:
                print(f'\n{result.backend}: {result.index_type}; search call {result.latency_ms:.2f} ms')
                for label, key in [('Exact-neighbor', 'exact_neighbors'), ('Labeled relevance', 'relevance')]:
                    metrics = result.metrics[key]
                    if metrics['available']:
                        print(f"{label}: Precision@{args.top_k}={metrics['precision_at_k']:.1%}, Recall@{args.top_k}={metrics['recall_at_k']:.1%}")
                    else:
                        print(f"{label}: N/A — {metrics['reason']}")
                for rank, hit in enumerate(result.hits, start=1):
                    print(f'{rank}. ID={hit.document.id} cosine={hit.score:.6f} | {hit.document.title}')
                if not result.hits:
                    print('No results.')
            print('\nSearch-call timings are individual observations, not benchmark latency statistics.')
    except Exception as error:
        message = str(error)
        if 'config' in locals():
            message = message.replace(config.postgres_password, '[redacted]')
        parser.exit(1, f'Search failed: {message}\n')


if __name__ == '__main__':
    main()
