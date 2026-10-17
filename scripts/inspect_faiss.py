"""Read a saved FAISS snapshot in a standalone process, without loading PyTorch."""

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.models import DatasetIdentity
from app.stores._faiss_native import NativeFaissVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-root', type=Path, default=PROJECT_ROOT / 'datasets/indexes/faiss')
    parser.add_argument('--namespace', help='choose a dataset if multiple indexes exist')
    parser.add_argument('--limit', type=int, default=5, help='number of sample documents to display')
    parser.add_argument('--id', type=int, help='show a specific document and its full vector')
    args = parser.parse_args()
    try:
        if args.limit < 1:
            raise ValueError('--limit must be positive')
        pointers = sorted(args.index_root.glob('*/active.json'))
        if args.namespace:
            pointers = [path for path in pointers if path.parent.name == args.namespace]
        if not pointers:
            raise ValueError('saved index not found; run scripts/ingest.py first')
        if len(pointers) != 1:
            raise ValueError('choose --namespace from: ' + ', '.join(path.parent.name for path in pointers))
        pointer = pointers[0]
        generation = json.loads(pointer.read_text(encoding='utf-8'))['generation']
        if not isinstance(generation, str) or not generation.isalnum():
            raise ValueError('invalid FAISS generation pointer')
        folder = pointer.parent / generation
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        identity = DatasetIdentity.model_validate(manifest['identity'])
        if identity.namespace != pointer.parent.name:
            raise ValueError('namespace does not match the saved index directory')
        store = NativeFaissVectorStore(identity, args.index_root)
        store.create_index()  # validates file hashes and ID-to-document alignment
        index = store.index
        base = faiss.downcast_index(index.index)
        print(f'Namespace: {identity.namespace}')
        print(f'Active folder: {folder}')
        print(f'Index type: {type(index).__name__}({type(base).__name__})')
        print(f'Dimensions: {index.d}')
        print(f'Vectors: {index.ntotal}')
        print(f'Trained/ready: {index.is_trained} (Flat indexes need no training step)')
        print('Metric: inner product on normalized vectors, equivalent to cosine similarity')
        print(f'Index file bytes: {(folder / "index.faiss").stat().st_size}')
        print('Integrity: file hashes and document IDs verified')
        selected = sorted(store.documents)[:args.limit]
        if args.id is not None:
            if args.id not in store.documents:
                raise ValueError(f'document ID {args.id} is not in this index')
            selected = [args.id]
        for stable_id in selected:
            document = store.documents[stable_id]
            vector = index.reconstruct(stable_id)
            print(f'\nID: {stable_id}\nTitle: {document.title}')
            print('Metadata: ' + json.dumps(document.metadata, ensure_ascii=False, sort_keys=True))
            print(f'Vector norm: {float(np.linalg.norm(vector)):.8f}')
            if args.id is not None:
                print('Content: ' + document.content)
                print('Full vector: ' + json.dumps(vector.tolist()))
            else:
                print('Vector preview (first 8 dimensions): ' + json.dumps(vector[:8].tolist()))
    except Exception as error:
        parser.exit(1, f'FAISS inspection failed: {error}\n')


if __name__ == '__main__':
    main()
