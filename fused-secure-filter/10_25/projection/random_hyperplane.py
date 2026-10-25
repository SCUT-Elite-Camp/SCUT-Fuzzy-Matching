"""Random hyperplane LSH on the baseline's EL=50 normalized matching vectors.

This is a plaintext candidate-filtering research module, NOT a secure PSI or
privacy-preserving access protocol. All projection rounds use independent planes.
"""
from __future__ import annotations
import numpy as np


def make_planes(dim: int, rounds: int, bits: int, seed: int) -> np.ndarray:
    if min(dim, rounds, bits) <= 0:
        raise ValueError('dim, rounds and bits must be positive')
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rounds, bits, dim)).astype(np.float64)


def hash_signatures(vectors: np.ndarray, planes: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    if vectors.ndim != 2 or planes.ndim != 3 or vectors.shape[1] != planes.shape[2]:
        raise ValueError('signature and projection-plane dimensions do not match')
    return np.einsum('nd,rsd->nrs', vectors, planes, optimize=True) >= 0


def collision_mask(query_signature: np.ndarray, database_signatures: np.ndarray) -> np.ndarray:
    """Keep a record iff all s bits match in at least one of R rounds."""
    query_signature = np.asarray(query_signature, dtype=bool)
    database_signatures = np.asarray(database_signatures, dtype=bool)
    if query_signature.shape != database_signatures.shape[1:]:
        raise ValueError('query signature shape mismatch')
    return np.any(np.all(database_signatures == query_signature[None, :, :], axis=2), axis=1)


def projected_candidates(cluster_member_indices: np.ndarray, query_signature: np.ndarray,
                         database_signatures: np.ndarray) -> np.ndarray:
    indices = np.asarray(cluster_member_indices, dtype=int)
    return indices[collision_mask(query_signature, database_signatures[indices])]
