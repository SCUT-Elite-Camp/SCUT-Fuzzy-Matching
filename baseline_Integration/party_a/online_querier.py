"""成员四：A 侧最终判断逻辑。"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import tenseal as ts

from config.params import DECRYPT_EPS
from protocol.types import MatchDebug, MatchResult


def _load_ciphertext(ciphertext, context: ts.Context) -> ts.CKKSVector:
    if isinstance(ciphertext, bytes):
        return ts.ckks_vector_from(context, ciphertext)
    return ciphertext


def _decrypt_scalar(ciphertext, secret_context: ts.Context) -> float:
    values = _load_ciphertext(ciphertext, secret_context).decrypt()
    return float(np.asarray(values, dtype=np.float64).reshape(-1)[0])


def check_encrypted_scores(
    encrypted_scores,
    secret_context,
    early_stop: bool = True,
    eps: float = DECRYPT_EPS,
) -> MatchResult:
    result, _ = check_encrypted_scores_debug(
        encrypted_scores,
        secret_context,
        early_stop=early_stop,
        eps=eps,
    )
    return result


def check_encrypted_scores_debug(
    encrypted_scores,
    secret_context,
    early_stop: bool = True,
    eps: float = DECRYPT_EPS,
) -> tuple[MatchResult, MatchDebug]:
    """Step 9: A 侧逐列解密判断是否存在阈值以上匹配。"""

    checked_columns = 0
    first_positive_column = None

    for column_index, encrypted_score in enumerate(encrypted_scores):
        plain_score = _decrypt_scalar(encrypted_score, secret_context)
        checked_columns += 1
        if plain_score > eps and first_positive_column is None:
            first_positive_column = column_index
            if early_stop:
                return MatchResult(catch=True), MatchDebug(
                    checked_columns=checked_columns,
                    first_positive_column=first_positive_column,
                )

    return MatchResult(catch=first_positive_column is not None), MatchDebug(
        checked_columns=checked_columns,
        first_positive_column=first_positive_column,
    )
