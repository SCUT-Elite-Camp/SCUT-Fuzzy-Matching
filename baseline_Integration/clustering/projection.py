"""Data-independent random projections for cluster feature construction."""

from __future__ import annotations

import numpy as np

from preprocessing.normalizer import l2_normalize


def rademacher_matrix(input_dim: int, output_dim: int, seed: int) -> np.ndarray:
    """Build the public projection matrix shared by both parties."""
    if isinstance(input_dim, bool) or not isinstance(input_dim, int):
        raise ValueError("input_dim must be an int")
    if isinstance(output_dim, bool) or not isinstance(output_dim, int):
        raise ValueError("output_dim must be an int")
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if output_dim <= 0 or output_dim > input_dim:
        raise ValueError(
            f"output_dim must be in [1, {input_dim}], got {output_dim}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an int")
    rng = np.random.default_rng(seed)
    return rng.choice(
        np.array([-1.0, 1.0]), size=(input_dim, output_dim)
    ) / np.sqrt(output_dim)


def project_normalized_signatures(
    matrix: np.ndarray,
    output_dim: int,
    seed: int,
) -> np.ndarray:
    """Project rows with a public Rademacher matrix and restore unit norms."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("matrix must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("matrix contains NaN or Inf")
    projection = rademacher_matrix(arr.shape[1], output_dim, seed)
    return l2_normalize(arr @ projection)
