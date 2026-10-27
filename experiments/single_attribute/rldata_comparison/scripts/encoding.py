"""Binary name encodings shared by the scheme-2 (random-projection) runner.

Scheme 1 already ships a MinHash encoder (``baseline_Integration.minhash``).
Scheme 2 needs a *binary* vector because the Approx-PSI filter projects bits and
then measures Hamming distance.  We use b-bit MinHash, which is the standard
binary LSH for Jaccard similarity:

* one permutation contributes ``b`` bits (the low ``b`` bits of the min-hash);
* two names that share a shingle collide in one bit with probability
  ``0.5 + 0.5 * J ** b`` (for ``b = 1`` this is exactly ``J``).

Both schemes are therefore driven by the same trigram/Jaccard notion of name
similarity; only the retrieval mechanism differs.

The hash family is the same one used by ``baseline_Integration`` (SHA-256
shingle hashes, ``MAX_HASH = 2**20``, seed 42), so the comparison is not
confounded by a different hash family.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from names import normalize_name

MAX_HASH = 2**20
HASH_SEED = 42
SHINGLE_SIZE = 3

# 8-bit popcount table, used for fast Hamming distances on packed bytes.
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class BitEncoding:
    """A b-bit MinHash configuration."""

    permutations: int = 128
    bits_per_permutation: int = 1
    shingle_size: int = SHINGLE_SIZE
    seed: int = HASH_SEED

    @property
    def length(self) -> int:
        return self.permutations * self.bits_per_permutation


def _shingles(name: str, shingle_size: int) -> list[str]:
    normalized = normalize_name(name)
    if len(normalized) < shingle_size:
        return [normalized or "<empty>"]
    return list({normalized[i : i + shingle_size] for i in range(len(normalized) - shingle_size + 1)})


def _hash_family(permutations: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    a = rng.randint(1, MAX_HASH, size=permutations).astype(np.int64)
    # An odd multiplier keeps the low bits of ``a*x+b`` informative; an even
    # multiplier can freeze the low bit (and, for multiples of 4, the low two
    # bits) for every shingle, wasting b-bit MinHash coordinates.
    a |= 1
    b = rng.randint(0, MAX_HASH, size=permutations).astype(np.int64)
    return a, b


def encode_names(
    names: list[str] | tuple[str, ...],
    encoding: BitEncoding,
) -> np.ndarray:
    """Return an ``(n, length)`` uint8 matrix of 0/1 MinHash bits."""
    a, b = _hash_family(encoding.permutations, encoding.seed)
    shifts = np.arange(encoding.bits_per_permutation, dtype=np.int64)
    output = np.zeros((len(names), encoding.length), dtype=np.uint8)
    for row, name in enumerate(names):
        shingles = _shingles(name, encoding.shingle_size)
        hashes = np.array(
            [int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % MAX_HASH for s in shingles],
            dtype=np.int64,
        )
        minima = ((a[:, None] * hashes[None, :] + b[:, None]) % MAX_HASH).min(axis=1)
        output[row] = ((minima[:, None] >> shifts[None, :]) & 1).reshape(-1)
    return output


def pack_bits(bits: np.ndarray) -> np.ndarray:
    """Pack a ``(..., length)`` bit matrix along the last axis (big-endian)."""
    return np.packbits(bits.astype(np.uint8), axis=-1)


def hamming_packed(packed_a: np.ndarray, packed_b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance between rows of two packed matrices.

    Returns a ``(len(a), len(b))`` uint16 matrix.  Both inputs must have the
    same number of packed bytes per row.
    """
    a = np.asarray(packed_a, dtype=np.uint8)
    b = np.asarray(packed_b, dtype=np.uint8)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"packed matrices must be (n, m) with equal m: {a.shape}, {b.shape}")
    out = np.empty((a.shape[0], b.shape[0]), dtype=np.uint16)
    # Chunk over the left side to keep the temporary XOR small.
    for start in range(0, a.shape[0], 64):
        stop = min(start + 64, a.shape[0])
        xor = a[start:stop, None, :] ^ b[None, :, :]
        out[start:stop] = _POPCOUNT[xor].sum(axis=2)
    return out


def hamming_rows_to_matrix(
    query_bits: np.ndarray,
    matrix_bits: np.ndarray,
) -> np.ndarray:
    """Hamming distance from each query row to each matrix row."""
    return hamming_packed(pack_bits(query_bits), pack_bits(matrix_bits))


def encode_dob(dobs: list[str]) -> np.ndarray:
    """Encode ISO dates (``YYYY-MM-DD``) as a 24-bit ordinal binary vector.

    Missing dates become the all-zero vector.  Bit-level Hamming distance is a
    poor metric for ordinals, so the multi-attribute runner uses this only
    after converting it to a distance via the packed bits (see
    ``fused_top1.py``); it is provided here so both schemes can be
    fed the same secondary attribute.
    """
    import datetime as dt

    output = np.zeros((len(dobs), 24), dtype=np.uint8)
    for row, value in enumerate(dobs):
        if not value:
            continue
        day = dt.date.fromisoformat(value)
        ordinal = (day - dt.date(1900, 1, 1)).days
        if 0 <= ordinal < 2**24:
            output[row] = ((ordinal >> np.arange(24)) & 1).astype(np.uint8)
    return output


def dob_distance(dob_a: list[str], dob_b: list[str]) -> np.ndarray:
    """Absolute day distance with a large penalty for missing values.

    Returns ``(len(a), len(b))`` float matrix.  Missing values on *both* sides
    count as a match (0); exactly one missing side gets the penalty.
    """
    import datetime as dt

    epoch = dt.date(1900, 1, 1)

    def to_days(values: list[str]) -> np.ndarray:
        out = np.full(len(values), np.nan)
        for i, value in enumerate(values):
            if value:
                out[i] = float((dt.date.fromisoformat(value) - epoch).days)
        return out

    a = to_days(dob_a)
    b = to_days(dob_b)
    diff = np.abs(a[:, None] - b[None, :])
    both_missing = np.isnan(a)[:, None] & np.isnan(b)[None, :]
    one_missing = np.isnan(a)[:, None] ^ np.isnan(b)[None, :]
    diff[both_missing] = 0.0
    diff[one_missing] = 36500.0
    return diff
