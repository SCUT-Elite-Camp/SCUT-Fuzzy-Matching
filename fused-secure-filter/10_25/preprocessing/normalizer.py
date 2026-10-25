"""Normalization helpers."""

import numpy as np
from sklearn.preprocessing import normalize


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Apply row-wise L2 normalization; zero rows remain zero."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        return normalize(matrix.reshape(1, -1), norm="l2")[0]
    return normalize(matrix, norm="l2")
