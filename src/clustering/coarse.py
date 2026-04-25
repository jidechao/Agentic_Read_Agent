"""Embedding-only coarse clustering utilities."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return a float32 L2-normalized vector.

    Zero vectors are returned unchanged to avoid NaNs in downstream cosine math.
    """
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr
    return (arr / norm).astype(np.float32)


def l2_normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize every row in an embedding matrix."""
    arr = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (arr / norms).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity for vectors that may not be normalized."""
    va = l2_normalize(a)
    vb = l2_normalize(b)
    return float(np.dot(va, vb))


def centroid(vectors: list[np.ndarray]) -> np.ndarray:
    """Compute a normalized centroid from one or more vectors."""
    if not vectors:
        return np.array([], dtype=np.float32)
    return l2_normalize(np.mean(np.vstack(vectors), axis=0).astype(np.float32))


def cosine_agglomerative(matrix: np.ndarray, threshold: float) -> list[int]:
    """Cluster embeddings using average-linkage cosine distance.

    The caller supplies a distance threshold instead of a fixed K, which is more
    stable for mixed enterprise corpora and avoids KMeans centroid artifacts.
    """
    if len(matrix) == 0:
        return []
    if len(matrix) == 1:
        return [0]

    normalized = l2_normalize_matrix(matrix)
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=threshold,
    )
    return [int(label) for label in model.fit_predict(normalized)]

