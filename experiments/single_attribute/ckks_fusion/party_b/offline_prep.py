"""Party B offline database preparation."""

from __future__ import annotations

import pathlib
from typing import Iterable

import numpy as np
from sklearn.preprocessing import StandardScaler

from clustering.kmeans_cosine import run_cosine_kmeans
from config.params import (
    KMEANS_ITERATIONS,
    NUM_PERMUTATIONS_CLUSTER,
    NUM_PERMUTATIONS_MATCH,
    choose_k,
)
from minhash.encoder import batch_encode
from preprocessing.normalizer import l2_normalize
from protocol.types import OfflineArtifacts


def prepare_party_b_offline(
    names_b: Iterable[str],
    k_mode: str = "sqrt",
    random_state: int = 42,
) -> OfflineArtifacts:
    """Build Party B artifacts for centroid and column-wise matching."""
    names_b = list(names_b)
    if not names_b:
        raise ValueError("names_b cannot be empty")

    signatures_200 = batch_encode(names_b, NUM_PERMUTATIONS_CLUSTER)
    signatures_50 = signatures_200[:, :NUM_PERMUTATIONS_MATCH]

    normalized_200 = l2_normalize(signatures_200)
    normalized_50 = l2_normalize(signatures_50)

    _, standardized_200, scaler_mean, scaler_scale = fit_scaler(normalized_200)
    k = choose_k(len(names_b), mode=k_mode)
    centroids, cluster_assignments = run_cosine_kmeans(
        standardized_200,
        k=k,
        iterations=KMEANS_ITERATIONS,
        random_state=random_state,
    )
    cluster_matrix, max_size = build_cluster_matrix(
        normalized_50, cluster_assignments, centroids.shape[0]
    )

    return OfflineArtifacts(
        centroids=centroids,
        cluster_matrix=cluster_matrix,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        cluster_assignments=cluster_assignments,
        max_size=max_size,
    )


def fit_scaler(
    normalized_200: np.ndarray,
) -> tuple[StandardScaler, np.ndarray, np.ndarray, np.ndarray]:
    """Fit Party B's scaler and expose its parameters for Party A."""
    matrix = np.asarray(normalized_200, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != NUM_PERMUTATIONS_CLUSTER:
        raise ValueError(
            "normalized_200 must have shape "
            f"(n, {NUM_PERMUTATIONS_CLUSTER}), got {matrix.shape}"
        )
    scaler = StandardScaler()
    standardized = scaler.fit_transform(matrix)
    return scaler, standardized, scaler.mean_, scaler.scale_


def build_cluster_matrix(
    normalized_50: np.ndarray,
    cluster_assignments: np.ndarray,
    k: int,
) -> tuple[np.ndarray, int]:
    """Group EL=50 vectors by cluster and zero-pad to the largest cluster."""
    matrix = np.asarray(normalized_50, dtype=np.float64)
    assignments = np.asarray(cluster_assignments, dtype=np.int32)
    if matrix.ndim != 2 or matrix.shape[1] != NUM_PERMUTATIONS_MATCH:
        raise ValueError(
            "normalized_50 must have shape "
            f"(n, {NUM_PERMUTATIONS_MATCH}), got {matrix.shape}"
        )
    if assignments.shape != (matrix.shape[0],):
        raise ValueError(
            f"cluster_assignments shape {assignments.shape} does not match "
            f"row count {matrix.shape[0]}"
        )
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    cluster_sizes = np.bincount(assignments, minlength=k)
    max_size = int(cluster_sizes.max()) if len(cluster_sizes) else 0
    cluster_matrix = np.zeros((k, max_size, matrix.shape[1]), dtype=np.float64)

    for cluster_index in range(k):
        members = matrix[assignments == cluster_index]
        cluster_matrix[cluster_index, : len(members), :] = members

    return cluster_matrix, max_size


def save_offline_artifacts(
    artifacts: OfflineArtifacts, save_dir: str | pathlib.Path
) -> None:
    """Persist offline artifacts as ``.npy`` files."""
    output_dir = pathlib.Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "centroids.npy", artifacts.centroids)
    np.save(output_dir / "cluster_matrix.npy", artifacts.cluster_matrix)
    np.save(output_dir / "scaler_mean.npy", artifacts.scaler_mean)
    np.save(output_dir / "scaler_scale.npy", artifacts.scaler_scale)
    np.save(output_dir / "cluster_assignments.npy", artifacts.cluster_assignments)
    np.save(output_dir / "max_size.npy", np.array([artifacts.max_size]))


def load_offline_artifacts(save_dir: str | pathlib.Path) -> OfflineArtifacts:
    """Load offline artifacts saved by :func:`save_offline_artifacts`."""
    input_dir = pathlib.Path(save_dir)
    return OfflineArtifacts(
        centroids=np.load(input_dir / "centroids.npy"),
        cluster_matrix=np.load(input_dir / "cluster_matrix.npy"),
        scaler_mean=np.load(input_dir / "scaler_mean.npy"),
        scaler_scale=np.load(input_dir / "scaler_scale.npy"),
        cluster_assignments=np.load(input_dir / "cluster_assignments.npy"),
        max_size=int(np.load(input_dir / "max_size.npy")[0]),
    )
