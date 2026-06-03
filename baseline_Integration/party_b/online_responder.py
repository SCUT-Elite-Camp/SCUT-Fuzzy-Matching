"""Party B online responders for centroid and column-wise matching."""

from __future__ import annotations

from random import SystemRandom
from typing import Iterator

import numpy as np
import tenseal as ts

from ckks.operations import add_plain, dot_ct_ct, dot_ct_pt, matmul_ct_pt
from config.params import (
    NUM_PERMUTATIONS_CLUSTER,
    RANDOM_MASK_MAX,
    RANDOM_MASK_MIN,
    SIMILARITY_THRESHOLD,
)
from protocol.types import EncryptedScalar, FirstRoundRequest, SecondRoundRequest

_RNG = SystemRandom()


def _load_public_context(public_context: ts.Context | bytes) -> ts.Context:
    if isinstance(public_context, bytes):
        return ts.Context.load(public_context)
    return public_context


def _load_ciphertext(ciphertext, context: ts.Context) -> ts.CKKSVector:
    if isinstance(ciphertext, bytes):
        return ts.ckks_vector_from(context, ciphertext)
    return ciphertext


def _sample_positive_mask() -> float:
    return _RNG.uniform(RANDOM_MASK_MIN, RANDOM_MASK_MAX)


def compare_to_centroids(
    first_round_request: FirstRoundRequest,
    centroids: np.ndarray,
    serialize_output: bool = False,
) -> list[EncryptedScalar]:
    """Step 3: compute encrypted query-to-centroid scores on Party B."""

    matrix = np.asarray(centroids, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"centroids must be 2-D, got shape {matrix.shape}")
    if matrix.shape[1] != NUM_PERMUTATIONS_CLUSTER:
        raise ValueError(
            f"centroids second dimension must be {NUM_PERMUTATIONS_CLUSTER}, "
            f"got {matrix.shape[1]}"
        )

    context = _load_public_context(first_round_request.public_context_bytes)
    encrypted_query_200 = _load_ciphertext(
        first_round_request.encrypted_query_200, context
    )
    if encrypted_query_200.size() != NUM_PERMUTATIONS_CLUSTER:
        raise ValueError(
            f"encrypted_query_200 length must be {NUM_PERMUTATIONS_CLUSTER}, "
            f"got {encrypted_query_200.size()}"
        )

    encrypted_scores = [
        dot_ct_pt(encrypted_query_200, centroid) for centroid in matrix
    ]
    if serialize_output:
        return [score.serialize() for score in encrypted_scores]
    return encrypted_scores


def column_wise_matching(
    cluster_matrix: np.ndarray,
    second_round_request: SecondRoundRequest,
    public_context,
    tau: float = SIMILARITY_THRESHOLD,
) -> Iterator[EncryptedScalar]:
    """Step 6-8: B 侧逐列执行 selector 选择、相似度计算和随机掩码。"""

    matrix = np.asarray(cluster_matrix, dtype=np.float64)
    if matrix.ndim != 3:
        raise ValueError(
            f"cluster_matrix must be 3-D, got shape {matrix.shape}"
        )
    if matrix.shape[2] != 50:
        raise ValueError(
            f"cluster_matrix last dimension must be 50, got {matrix.shape[2]}"
        )

    context = _load_public_context(public_context)
    encrypted_query_50 = _load_ciphertext(
        second_round_request.encrypted_query_50, context
    )
    encrypted_selector = _load_ciphertext(
        second_round_request.encrypted_selector, context
    )

    k, max_size, _ = matrix.shape
    if encrypted_selector.size() != k:
        raise ValueError(
            f"Selector length {encrypted_selector.size()} does not match "
            f"cluster count {k}"
        )
    if encrypted_query_50.size() != 50:
        raise ValueError(
            f"encrypted_query_50 length must be 50, got {encrypted_query_50.size()}"
        )

    for column_index in range(max_size):
        mask = _sample_positive_mask()
        column_j = matrix[:, column_index, :]
        masked_column_j = mask * column_j
        encrypted_selected_name_j = matmul_ct_pt(
            encrypted_selector, masked_column_j
        )
        encrypted_cos_score_j = dot_ct_ct(
            encrypted_query_50, encrypted_selected_name_j
        )
        yield add_plain(encrypted_cos_score_j, -mask * tau)
