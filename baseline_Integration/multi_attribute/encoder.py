from __future__ import annotations

import hashlib
import math
from datetime import datetime
from typing import Iterable

import numpy as np

from minhash.encoder import batch_encode
from preprocessing.normalizer import l2_normalize

from .config import MultiAttributeConfig
from .model import MatchRecord

_DOB_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%Y%m%d",
)


def normalize_dob(value: str | None) -> str | None:
    """Normalize common DOB spellings to ``YYYY-MM-DD``.

    Missing/blank DOB returns ``None``. Invalid non-empty values fail closed so
    accidental garbage does not silently become a matching feature.
    """

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in _DOB_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(
        f"unsupported DOB format {value!r}; expected YYYY-MM-DD, YYYY/MM/DD, "
        "YYYY.MM.DD or YYYYMMDD"
    )


def _exact_hash_vector(
    value: str | None,
    *,
    blocks: int,
    buckets_per_block: int,
    seed: int,
) -> np.ndarray:
    """Encode an exact categorical value into independent one-hot hash blocks.

    The vector is L2-normalized by construction. Missing value => all zeros.
    """

    dim = blocks * buckets_per_block
    out = np.zeros(dim, dtype=np.float64)
    if value is None:
        return out

    amplitude = 1.0 / math.sqrt(blocks)
    for block in range(blocks):
        payload = f"{seed}:{block}:{value}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        bucket = int.from_bytes(digest[:8], "big") % buckets_per_block
        out[block * buckets_per_block + bucket] = amplitude
    return out


def _encode_name_blocks(
    names: list[str], cfg: MultiAttributeConfig
) -> tuple[np.ndarray, np.ndarray]:
    raw_cluster = batch_encode(names, cfg.name_cluster_dim)
    raw_match = raw_cluster[:, : cfg.name_match_dim].copy()
    return l2_normalize(raw_cluster), l2_normalize(raw_match)


def encode_record_vectors(
    records: Iterable[MatchRecord],
    cfg: MultiAttributeConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode records into weighted cluster and final-match vectors.

    Returns:
        cluster_vectors: shape ``(n, cfg.cluster_dim)``
        match_vectors: shape ``(n, cfg.match_dim)``

    For records with both attributes present, the dot product between two match
    vectors equals::

        w_name * cosine(name_block_A, name_block_B)
        + w_dob * dot(dob_block_A, dob_block_B)
    """

    cfg = cfg or MultiAttributeConfig()
    rows = list(records)
    if not rows:
        return (
            np.empty((0, cfg.cluster_dim), dtype=np.float64),
            np.empty((0, cfg.match_dim), dtype=np.float64),
        )

    name_cluster, name_match = _encode_name_blocks([r.name for r in rows], cfg)
    dob_vectors = np.stack(
        [
            _exact_hash_vector(
                normalize_dob(r.dob),
                blocks=cfg.dob_hash_blocks,
                buckets_per_block=cfg.dob_buckets_per_block,
                seed=cfg.dob_hash_seed,
            )
            for r in rows
        ],
        axis=0,
    )

    sqrt_name = math.sqrt(cfg.name_weight)
    sqrt_dob = math.sqrt(cfg.dob_weight)

    cluster_vectors = np.concatenate(
        [sqrt_name * name_cluster, sqrt_dob * dob_vectors], axis=1
    )
    match_vectors = np.concatenate(
        [sqrt_name * name_match, sqrt_dob * dob_vectors], axis=1
    )

    return cluster_vectors.astype(np.float64), match_vectors.astype(np.float64)


def plaintext_similarity(
    left: MatchRecord,
    right: MatchRecord,
    cfg: MultiAttributeConfig | None = None,
) -> float:
    """Plaintext reference score for tests/debugging only."""

    cfg = cfg or MultiAttributeConfig()
    _, vectors = encode_record_vectors([left, right], cfg)
    return float(np.dot(vectors[0], vectors[1]))
