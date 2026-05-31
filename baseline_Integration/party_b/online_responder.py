"""成员四：B 侧第二轮列式密文匹配。"""

from __future__ import annotations

from random import SystemRandom
from typing import Iterator

import numpy as np
import tenseal as ts

from ckks.operations import add_plain, dot_ct_ct, matmul_ct_pt
from config.params import RANDOM_MASK_MAX, RANDOM_MASK_MIN, SIMILARITY_THRESHOLD
from protocol.types import EncryptedScalar, SecondRoundRequest

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
