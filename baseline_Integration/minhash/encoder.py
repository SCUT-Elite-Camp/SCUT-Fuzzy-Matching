"""MinHash signature generation.

The hash family is always generated at EL=200 and then truncated, so an
EL=50 signature is exactly the prefix of the corresponding EL=200 signature.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

from config.params import (
    HASH_SEED,
    MAX_HASH,
    NUM_PERMUTATIONS_CLUSTER,
    SHINGLE_SIZE,
)
from preprocessing.text_cleaner import clean_name


def _generate_shingles(name: str, shingle_size: int) -> set[str]:
    normalized = clean_name(name)
    if len(normalized) < shingle_size:
        return {normalized} if normalized else {"<empty>"}
    return {
        normalized[i : i + shingle_size]
        for i in range(len(normalized) - shingle_size + 1)
    }


def _hash_shingle(shingle: str) -> int:
    digest = hashlib.sha256(shingle.encode("utf-8")).hexdigest()
    return int(digest, 16) % MAX_HASH


def _hash_family(num_permutations: int) -> tuple[np.ndarray, np.ndarray]:
    if num_permutations > NUM_PERMUTATIONS_CLUSTER:
        raise ValueError(
            "num_permutations cannot exceed NUM_PERMUTATIONS_CLUSTER "
            f"({NUM_PERMUTATIONS_CLUSTER})"
        )
    rng = np.random.RandomState(HASH_SEED)
    a_full = rng.randint(
        1, MAX_HASH, size=NUM_PERMUTATIONS_CLUSTER
    ).astype(np.int64)
    b_full = rng.randint(
        0, MAX_HASH, size=NUM_PERMUTATIONS_CLUSTER
    ).astype(np.int64)
    return a_full[:num_permutations], b_full[:num_permutations]


def generate_signature(name: str, num_permutations: int) -> np.ndarray:
    """Generate one MinHash signature with shape ``(num_permutations,)``."""
    a, b = _hash_family(num_permutations)
    shingles = _generate_shingles(name, SHINGLE_SIZE)
    hashes = np.array([_hash_shingle(s) for s in shingles], dtype=np.int64)
    permuted = (a[:, None] * hashes[None, :] + b[:, None]) % MAX_HASH
    return permuted.min(axis=1).astype(np.float64)


def batch_encode(names: Iterable[str], num_permutations: int) -> np.ndarray:
    """Generate a MinHash signature matrix with shape ``(n, num_permutations)``."""
    names = list(names)
    if not names:
        return np.empty((0, num_permutations), dtype=np.float64)

    a, b = _hash_family(num_permutations)
    signatures = []
    for name in names:
        shingles = _generate_shingles(name, SHINGLE_SIZE)
        hashes = np.array([_hash_shingle(s) for s in shingles], dtype=np.int64)
        permuted = (a[:, None] * hashes[None, :] + b[:, None]) % MAX_HASH
        signatures.append(permuted.min(axis=1).astype(np.float64))
    return np.stack(signatures, axis=0)

