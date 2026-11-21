"""Cosine/Spherical K-Means implementation."""

from __future__ import annotations

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config.params import KMEANS_ITERATIONS
from preprocessing.normalizer import l2_normalize


def _init_centroids(
    matrix: np.ndarray, k: int, rng: np.random.RandomState
) -> np.ndarray:
    indices = rng.choice(len(matrix), size=min(k, len(matrix)), replace=False)
    return l2_normalize(matrix[indices].copy())


def run_legacy_cosine_kmeans(
    standardized_200: np.ndarray,
    k: int,
    iterations: int = KMEANS_ITERATIONS,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster rows by cosine similarity and return normalized centroids."""
    matrix = np.asarray(standardized_200, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"standardized_200 must be 2-D, got {matrix.shape}")
    if len(matrix) == 0:
        raise ValueError("standardized_200 cannot be empty")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    k = min(k, len(matrix))

    rng = np.random.RandomState(random_state)
    centroids = _init_centroids(matrix, k, rng)
    cluster_assignments = np.zeros(len(matrix), dtype=np.int32)

    for _ in range(iterations):
        previous = centroids.copy()
        sim_matrix = cosine_similarity(matrix, centroids)
        cluster_assignments = np.argmax(sim_matrix, axis=1).astype(np.int32)

        updated = np.zeros_like(centroids)
        empty_clusters = []
        for cluster_index in range(k):
            members = matrix[cluster_assignments == cluster_index]
            if len(members) == 0:
                empty_clusters.append(cluster_index)
                continue
            mean_vec = members.mean(axis=0)
            norm = np.linalg.norm(mean_vec)
            updated[cluster_index] = (
                previous[cluster_index] if norm < 1e-10 else mean_vec / norm
            )

        for cluster_index in empty_clusters:
            assigned_sims = sim_matrix[np.arange(len(matrix)), cluster_assignments]
            farthest_idx = int(np.argmax(1.0 - assigned_sims))
            replacement = matrix[farthest_idx].copy()
            norm = np.linalg.norm(replacement)
            if norm < 1e-10:
                replacement = previous[cluster_index]
            else:
                replacement = replacement / norm
            updated[cluster_index] = replacement

        centroids = updated
        if np.max(np.linalg.norm(centroids - previous, axis=1)) < 1e-8:
            break

    sim_matrix = cosine_similarity(matrix, centroids)
    cluster_assignments = np.argmax(sim_matrix, axis=1).astype(np.int32)
    return centroids, cluster_assignments


def _run_emeans(
    standardized_200: np.ndarray,
    k: int,
    iterations: int = KMEANS_ITERATIONS,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Run FAISS's official spherical e-means behind the baseline interface."""
    matrix = np.asarray(standardized_200, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) == 0 or k <= 0 or iterations <= 0:
        raise ValueError("expected a nonempty 2-D matrix, positive k and iterations")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix must contain only finite values")
    try:
        from faiss.contrib.e_means import NumpyEmeans
    except ImportError as exc:
        raise ImportError("Official e-means requires faiss-cpu>=1.16.0") from exc

    k = min(k, len(matrix))
    unit_matrix = np.ascontiguousarray(l2_normalize(matrix), dtype=np.float32)
    model = NumpyEmeans(unit_matrix.shape[1], k, nepochs=iterations,
                        seed=random_state, spherical=True)
    centers, _ = model.train(unit_matrix)
    # The baseline returns assignments against its final centroids.
    _, assignments = model.assign(unit_matrix)
    return np.asarray(centers, dtype=np.float64), np.asarray(assignments, dtype=np.int32)


def run_cosine_kmeans(
    standardized_200: np.ndarray,
    k: int,
    iterations: int = KMEANS_ITERATIONS,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Existing entry point; SCUT_CLUSTER_ALGORITHM=kmeans selects the control arm."""
    algorithm = os.environ.get("SCUT_CLUSTER_ALGORITHM", "emeans").lower()
    if algorithm == "emeans":
        return _run_emeans(standardized_200, k, iterations, random_state)
    if algorithm == "kmeans":
        return run_legacy_cosine_kmeans(standardized_200, k, iterations, random_state)
    raise ValueError(f"unknown SCUT_CLUSTER_ALGORITHM: {algorithm}")
