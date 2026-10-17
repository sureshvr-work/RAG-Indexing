"""Compare exact backend results with NumPy cosine scores, allowing cutoff ties."""

import numpy as np

from app.core.artifacts import EmbeddingBundle
from app.core.models import SearchResult


def check_exact_result(bundle: EmbeddingBundle, query: np.ndarray, result: SearchResult,
                       top_k: int, tolerance: float = 1e-5) -> float:
    if top_k < 1 or result.namespace != bundle.identity.namespace:
        raise ValueError('invalid top_k or mismatched dataset')
    vectors = bundle.vectors.astype(np.float64)
    query = np.asarray(query, dtype=np.float64)
    if query.shape != (bundle.identity.dimensions,) or not np.isfinite(query).all() or np.linalg.norm(query) == 0:
        raise ValueError('invalid query vector')
    scores = (vectors @ query) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query))
    expected_count = min(top_k, len(bundle.ids))
    if len(result.hits) != expected_count:
        raise ValueError(f'{result.backend}: incorrect hit count')
    positions = {int(stable_id): i for i, stable_id in enumerate(bundle.ids)}
    found = [hit.document.id for hit in result.hits]
    if len(set(found)) != len(found) or not set(found).issubset(positions):
        raise ValueError(f'{result.backend}: duplicate or unknown result IDs')
    if not expected_count:
        return 0.0
    cutoff = np.sort(scores)[-expected_count]
    # Different engines may choose different IDs for equal scores at the boundary.
    required = set(bundle.ids[scores > cutoff + tolerance].tolist())
    if not required.issubset(found):
        raise ValueError(f'{result.backend}: omitted a result above the top-k cutoff')
    max_error = 0.0
    previous = float('inf')
    for hit in result.hits:
        position = positions[hit.document.id]
        expected = float(scores[position])
        error = abs(hit.score - expected)
        if not np.isfinite(hit.score) or error > tolerance or expected < cutoff - tolerance:
            raise ValueError(f'{result.backend}: result differs from exact cosine baseline')
        if hit.score > previous + tolerance:
            raise ValueError(f'{result.backend}: results are not ordered by similarity')
        if hit.document != bundle.documents[position]:
            raise ValueError(f'{result.backend}: document payload differs from the source sample')
        previous = hit.score
        max_error = max(max_error, error)
    return max_error
