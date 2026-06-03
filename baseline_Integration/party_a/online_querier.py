"""Party A online logic for cluster selection and final judgment."""

from __future__ import annotations

from typing import Any

import numpy as np
import tenseal as ts

from config.params import DECRYPT_EPS
from ckks.keys import encrypt
from protocol.types import (
    ClusterSelectionDebug,
    EncryptedVectorK,
    MatchDebug,
    MatchResult,
    PartyALocalState,
    SecondRoundRequest,
)


def _load_ciphertext(ciphertext, context: ts.Context) -> ts.CKKSVector:
    if isinstance(ciphertext, bytes):
        return ts.ckks_vector_from(context, ciphertext)
    return ciphertext


def _decrypt_scalar(ciphertext, secret_context: ts.Context) -> float:
    values = _load_ciphertext(ciphertext, secret_context).decrypt()
    return float(np.asarray(values, dtype=np.float64).reshape(-1)[0])


def decrypt_sim_scores(
    encrypted_sim_scores: Any,
    secret_context: ts.Context,
) -> np.ndarray:
    """Step 4: decrypt centroid scores on Party A."""

    if isinstance(encrypted_sim_scores, (list, tuple)):
        values = []
        for ciphertext in encrypted_sim_scores:
            decrypted = _load_ciphertext(ciphertext, secret_context).decrypt()
            values.extend(np.asarray(decrypted, dtype=np.float64).reshape(-1))
        return np.asarray(values, dtype=np.float64)

    decrypted = _load_ciphertext(encrypted_sim_scores, secret_context).decrypt()
    return np.asarray(decrypted, dtype=np.float64).reshape(-1)


def build_selector(sim_scores: np.ndarray, k: int) -> tuple[int, np.ndarray]:
    """Build a one-hot selector for the highest-scoring cluster."""

    scores = np.asarray(sim_scores, dtype=np.float64).reshape(-1)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if scores.shape != (k,):
        raise ValueError(
            f"sim_scores shape mismatch: expected ({k},), got {scores.shape}"
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("sim_scores contains NaN or Inf")

    selected_cluster = int(np.argmax(scores))
    selector = np.zeros(k, dtype=np.float64)
    selector[selected_cluster] = 1.0
    return selected_cluster, selector


def encrypt_selector(
    selector: np.ndarray,
    secret_context: ts.Context,
) -> EncryptedVectorK:
    """Step 5: encrypt the one-hot selector before sending it to Party B."""

    selector = np.asarray(selector, dtype=np.float64).reshape(-1)
    if not np.isclose(selector.sum(), 1.0):
        raise ValueError("selector must be one-hot: sum(selector) should be 1")
    if not np.all((selector == 0.0) | (selector == 1.0)):
        raise ValueError("selector must be one-hot: values must be 0 or 1")
    return encrypt(selector, secret_context)


def choose_cluster_and_build_request(
    encrypted_sim_scores: Any,
    party_a_state: PartyALocalState,
    k: int,
) -> tuple[SecondRoundRequest, ClusterSelectionDebug]:
    """Step 4-5: choose a cluster and build the second-round request."""

    sim_scores = decrypt_sim_scores(
        encrypted_sim_scores, party_a_state.secret_context
    )
    selected_cluster, selector = build_selector(sim_scores, k)
    encrypted_selector = encrypt_selector(selector, party_a_state.secret_context)

    return (
        SecondRoundRequest(
            encrypted_query_50=party_a_state.encrypted_query_50,
            encrypted_selector=encrypted_selector,
        ),
        ClusterSelectionDebug(selected_cluster=selected_cluster),
    )


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
