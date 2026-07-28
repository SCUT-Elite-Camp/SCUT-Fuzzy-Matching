"""Party A online logic for cluster selection and final judgment."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import numpy as np
import tenseal as ts

from ckks.batching import decrypt_score_batch, encrypt_feature_batch
from ckks.keys import encrypt
from ckks.tiling import (
    SlotTileLayout,
    decrypt_tiled_score_tile,
    encrypt_tiled_feature_batch,
    iter_tile_slices,
)
from config.params import BATCH_DECRYPT_EPS, DECRYPT_EPS, POLY_MODULUS_DEGREE
from protocol.types import (
    BatchClusterSelectionDebug,
    BatchMatchDebug,
    BatchMatchResult,
    BatchPartyALocalState,
    BatchSecondRoundRequest,
    CipherLike,
    ClusterSelectionDebug,
    EncryptedVectorK,
    MatchDebug,
    MatchResult,
    PartyALocalState,
    SecondRoundRequest,
    TiledClusterSelectionDebug,
    TiledMatchDebug,
    TiledMatchResult,
    TiledPartyALocalState,
    TiledSecondRoundRequest,
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

    sim_scores = decrypt_sim_scores(encrypted_sim_scores, party_a_state.secret_context)
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


def choose_clusters_and_build_tiled_request(
    encrypted_score_tiles: Iterable[CipherLike],
    party_a_state: TiledPartyALocalState,
    k: int,
) -> tuple[TiledSecondRoundRequest, TiledClusterSelectionDebug]:
    """V2 Step 4-5: 解密 centroid score tiles，按 query argmax 选 cluster，
    构造第二轮 tiled selector 请求。
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    layout = party_a_state.layout
    m = layout.batch_size

    # Decrypt and unpack all tiles, then concatenate to (m, k).
    score_iter = iter(encrypted_score_tiles)
    tile_arrays = []
    for _, _, valid_width in iter_tile_slices(k, layout):
        try:
            encrypted_tile = next(score_iter)
        except StopIteration as exc:
            raise ValueError("missing encrypted centroid score tile") from exc
        tile = decrypt_tiled_score_tile(
            encrypted_tile,
            party_a_state.secret_context,
            layout,
            valid_width,
        )
        tile_arrays.append(tile)
    try:
        next(score_iter)
    except StopIteration:
        pass
    else:
        raise ValueError("received extra encrypted centroid score tile")
    scores_matrix = np.concatenate(tile_arrays, axis=1)

    if scores_matrix.shape != (m, k):
        raise ValueError(
            f"scores_matrix shape mismatch: expected ({m}, {k}), got {scores_matrix.shape}"
        )
    if not np.all(np.isfinite(scores_matrix)):
        raise ValueError("decrypted sim_scores contain NaN or Inf")

    selected_clusters = np.argmax(scores_matrix, axis=1)

    selector_matrix = np.zeros((m, k), dtype=np.float64)
    for q in range(m):
        selector_matrix[q, selected_clusters[q]] = 1.0

    encrypted_selectors = encrypt_tiled_feature_batch(
        selector_matrix, party_a_state.secret_context, layout
    )

    return (
        TiledSecondRoundRequest(
            encrypted_query_50=party_a_state.encrypted_query_50,
            encrypted_selectors=encrypted_selectors,
            layout=layout,
        ),
        TiledClusterSelectionDebug(selected_clusters=selected_clusters),
    )


def check_tiled_score_batch_debug(
    encrypted_score_tiles: Iterable[CipherLike],
    secret_context: ts.Context,
    *,
    layout: SlotTileLayout,
    logical_width: int,
    early_stop: bool = True,
    eps: float = BATCH_DECRYPT_EPS,
) -> tuple[TiledMatchResult, TiledMatchDebug]:
    """V2 Step 9: A 侧按 tile 解密，向量化的命中判断。

    每个 tile 解密为 (m, valid_width)，按列顺序判断正负。
    early_stop=True 仅当所有 query 均命中时才停止。
    """
    if isinstance(logical_width, bool) or not isinstance(logical_width, int):
        raise ValueError("logical_width must be an int")
    if logical_width <= 0:
        raise ValueError("logical_width must be positive")
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and non-negative")

    score_iter = iter(encrypted_score_tiles)
    catches = np.zeros(layout.batch_size, dtype=bool)
    first_positive_columns = np.full(layout.batch_size, -1, dtype=int)
    checked_tiles = 0
    logical_columns_checked = 0

    for tile_idx, start, valid_width in iter_tile_slices(logical_width, layout):
        try:
            encrypted_tile = next(score_iter)
        except StopIteration as exc:
            raise ValueError(
                f"missing encrypted score tile at index {tile_idx}"
            ) from exc
        checked_tiles += 1
        tile_scores = decrypt_tiled_score_tile(
            encrypted_tile,
            secret_context,
            layout,
            valid_width,
        )
        # tile_scores shape (m, valid_width)
        positive = tile_scores > eps
        has_hit = positive.any(axis=1)
        local_first = positive.argmax(axis=1)

        new_hit = (~catches) & has_hit
        first_positive_columns[new_hit] = start + local_first[new_hit]
        catches |= has_hit
        logical_columns_checked += valid_width

        if early_stop and np.all(catches):
            break
    else:
        try:
            next(score_iter)
        except StopIteration:
            pass
        else:
            raise ValueError("received extra encrypted score tile")

    return (
        TiledMatchResult(catches=catches),
        TiledMatchDebug(
            checked_tiles=checked_tiles,
            logical_columns_checked=logical_columns_checked,
            first_positive_columns=first_positive_columns,
        ),
    )


def check_tiled_score_batch(
    encrypted_score_tiles: Iterable[CipherLike],
    secret_context: ts.Context,
    *,
    layout: SlotTileLayout,
    logical_width: int,
    early_stop: bool = True,
    eps: float = BATCH_DECRYPT_EPS,
) -> TiledMatchResult:
    result, _ = check_tiled_score_batch_debug(
        encrypted_score_tiles,
        secret_context,
        layout=layout,
        logical_width=logical_width,
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


def choose_clusters_and_build_batch_request(
    encrypted_sim_scores: Sequence[CipherLike],
    party_a_state: BatchPartyALocalState,
    k: int,
) -> tuple[BatchSecondRoundRequest, BatchClusterSelectionDebug]:
    """批量 Step 4-5：解密 k 个质心相似度分数，求 argmax 选聚类，构造第二轮请求。

    Args:
        encrypted_sim_scores: k 个相似度分数密文 (每个含 m slots)。
        party_a_state: A 侧本地批量状态。
        k: 聚类数量。

    Returns:
        tuple[BatchSecondRoundRequest, BatchClusterSelectionDebug]:
            - BatchSecondRoundRequest: 第二轮请求 (含 Q50 密文与 encrypted_selectors 密文)。
            - BatchClusterSelectionDebug: 本地调试信息 (m 个查询选择的聚类编号)。
    """
    if k <= 0:
        raise ValueError(f"k 必须为正整数，当前值: {k}")

    m = party_a_state.batch_size
    scores_matrix = decrypt_score_batch(
        encrypted_sim_scores,
        party_a_state.secret_context,
        output_count=k,
        batch_size=m,
    )  # (m, k)

    if not np.all(np.isfinite(scores_matrix)):
        raise ValueError("解密后的 sim_scores 包含 NaN 或 Inf 非法数值")

    selected_clusters = np.argmax(scores_matrix, axis=1)  # (m,)

    # 构造 (m, k) 维度的 one-hot selector 矩阵
    selector_matrix = np.zeros((m, k), dtype=np.float64)
    for q in range(m):
        selector_matrix[q, selected_clusters[q]] = 1.0

    # 将 (m, k) selector 矩阵纵向加密为 k 个长度 m 的密文
    encrypted_selectors = encrypt_feature_batch(
        selector_matrix, party_a_state.secret_context
    )

    return (
        BatchSecondRoundRequest(
            encrypted_query_50=party_a_state.encrypted_query_50,
            encrypted_selectors=encrypted_selectors,
            batch_size=m,
        ),
        BatchClusterSelectionDebug(selected_clusters=selected_clusters),
    )


def check_encrypted_score_batch_debug(
    encrypted_scores: Iterator[CipherLike] | Sequence[CipherLike],
    secret_context: ts.Context,
    *,
    batch_size: int,
    early_stop: bool = True,
    eps: float = BATCH_DECRYPT_EPS,
) -> tuple[BatchMatchResult, BatchMatchDebug]:
    """批量 Step 9：A 侧逐列解密判断每个查询是否存在阈值以上匹配。

    Args:
        encrypted_scores: 列分数密文流或列表 (每个密文含 m slots)。
        secret_context: A 侧私钥上下文。
        batch_size: 查询批量 m。
        early_stop: 是否在所有查询均命中时提前停止生成器。
        eps: 阈值浮点误差上界。

    Returns:
        tuple[BatchMatchResult, BatchMatchDebug]:
            - BatchMatchResult: 命中结果 bool 数组 (m,)。
            - BatchMatchDebug: 检查列数及每个查询首个命中列 index (未命中为 -1)。
    """
    max_batch = POLY_MODULUS_DEGREE // 2
    if batch_size <= 0 or batch_size > max_batch:
        raise ValueError(
            f"batch_size 必须位于 (0, {max_batch}] 范围内，当前值: {batch_size}"
        )

    catches = np.zeros(batch_size, dtype=bool)
    first_positive_columns = np.full(batch_size, -1, dtype=int)
    checked_columns = 0

    for column_index, enc_col_score in enumerate(encrypted_scores):
        checked_columns += 1
        col_plain_2d = decrypt_score_batch(
            [enc_col_score],
            secret_context,
            output_count=1,
            batch_size=batch_size,
        )  # (m, 1)
        col_plain = col_plain_2d.reshape(-1)  # (m,)

        for q in range(batch_size):
            if not catches[q] and col_plain[q] > eps:
                catches[q] = True
                first_positive_columns[q] = column_index

        # early_stop=True 只有在所有 query 均已命中时才能停止生成器
        if early_stop and np.all(catches):
            break

    return (
        BatchMatchResult(catches=catches),
        BatchMatchDebug(
            checked_columns=checked_columns,
            first_positive_columns=first_positive_columns,
        ),
    )


def check_encrypted_score_batch(
    encrypted_scores: Iterator[CipherLike] | Sequence[CipherLike],
    secret_context: ts.Context,
    *,
    batch_size: int,
    early_stop: bool = True,
    eps: float = BATCH_DECRYPT_EPS,
) -> BatchMatchResult:
    """批量 Step 9 对外接口：只返回 BatchMatchResult。"""
    result, _ = check_encrypted_score_batch_debug(
        encrypted_scores,
        secret_context,
        batch_size=batch_size,
        early_stop=early_stop,
        eps=eps,
    )
    return result
