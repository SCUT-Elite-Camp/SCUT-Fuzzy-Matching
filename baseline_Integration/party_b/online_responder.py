"""Party B online responders for centroid and column-wise matching."""

from __future__ import annotations

from collections.abc import Iterator
from random import SystemRandom

import numpy as np
import tenseal as ts

from ckks.batching import (
    batch_dot_ct_ct,
    batch_dot_ct_pt,
    batch_select_ct_pt,
    load_feature_batch,
    serialize_feature_batch,
)
from ckks.operations import add_plain, dot_ct_ct, dot_ct_pt, matmul_ct_pt
from ckks.tiling import (
    SlotTileLayout,
    expand_plain_tile,
    iter_tile_slices,
    pack_candidate_tile,
    pack_centroid_tile,
)
from config.params import (
    NUM_PERMUTATIONS_CLUSTER,
    NUM_PERMUTATIONS_MATCH,
    RANDOM_MASK_MAX,
    RANDOM_MASK_MIN,
    SIMILARITY_THRESHOLD,
)
from protocol.types import (
    BatchFirstRoundRequest,
    BatchSecondRoundRequest,
    CipherLike,
    EncryptedScalar,
    FirstRoundRequest,
    SecondRoundRequest,
    TiledFirstRoundRequest,
    TiledSecondRoundRequest,
)

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
    encrypted_scores = [dot_ct_pt(encrypted_query_200, centroid) for centroid in matrix]
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
        raise ValueError(f"cluster_matrix must be 3-D, got shape {matrix.shape}")
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
        encrypted_selected_name_j = matmul_ct_pt(encrypted_selector, masked_column_j)
        encrypted_cos_score_j = dot_ct_ct(encrypted_query_50, encrypted_selected_name_j)
        yield add_plain(encrypted_cos_score_j, -mask * tau)


def _sample_positive_mask_batch(batch_size: int) -> np.ndarray:
    return np.array(
        [_RNG.uniform(RANDOM_MASK_MIN, RANDOM_MASK_MAX) for _ in range(batch_size)],
        dtype=np.float64,
    )


def compare_batch_to_centroids(
    request: BatchFirstRoundRequest,
    centroids: np.ndarray,
    serialize_output: bool = False,
) -> list[EncryptedScalar]:
    """Step 3 (Batch): compute encrypted query-batch-to-centroid scores on Party B.
    Args:
        request: BatchFirstRoundRequest payload from Party A.
        centroids: Centroids matrix of shape (k, 200).
        serialize_output: Whether to return serialized bytes list.
    Returns:
        list[EncryptedScalar]: k encrypted score vectors (or bytes), each of size m.
    """
    matrix = np.asarray(centroids, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"centroids must be 2-D, got shape {matrix.shape}")
    if matrix.shape[1] != NUM_PERMUTATIONS_CLUSTER:
        raise ValueError(
            f"centroids second dimension must be {NUM_PERMUTATIONS_CLUSTER}, "
            f"got {matrix.shape[1]}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("centroids contains NaN or Inf")
    k, _ = matrix.shape
    if k <= 0:
        raise ValueError("centroids must contain at least 1 centroid")
    batch_size = request.batch_size
    context = _load_public_context(request.public_context_bytes)
    enc_query_200 = load_feature_batch(
        request.encrypted_query_200,
        context,
        feature_count=NUM_PERMUTATIONS_CLUSTER,
        batch_size=batch_size,
    )
    score_ciphertexts = batch_dot_ct_pt(enc_query_200, matrix)
    if serialize_output:
        return serialize_feature_batch(score_ciphertexts)
    return score_ciphertexts


def column_wise_batch_matching(
    cluster_matrix: np.ndarray,
    request: BatchSecondRoundRequest,
    public_context: ts.Context | bytes,
    tau: float = SIMILARITY_THRESHOLD,
    *,
    serialize_output: bool = False,
) -> Iterator[EncryptedScalar]:
    """Step 6-8 (Batch): B 侧纵向 SIMD 逐列 selector 选择、相似度计算与独项随机掩码。
    Args:
        cluster_matrix: Cluster matrix of shape (k, max_size, 50).
        request: BatchSecondRoundRequest payload from Party A.
        public_context: Public CKKS context object or bytes.
        tau: Similarity threshold float.
        serialize_output: Whether to yield bytes.
    Yields:
        Iterator[EncryptedScalar]: Yields max_size score ciphertexts (or bytes), each of size m.
    """
    matrix = np.asarray(cluster_matrix, dtype=np.float64)
    if matrix.ndim != 3:
        raise ValueError(f"cluster_matrix must be 3-D, got shape {matrix.shape}")
    if matrix.shape[2] != 50:
        raise ValueError(
            f"cluster_matrix last dimension must be 50, got {matrix.shape[2]}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("cluster_matrix contains NaN or Inf")
    k, max_size, _ = matrix.shape
    batch_size = request.batch_size
    context = _load_public_context(public_context)
    enc_query_50 = load_feature_batch(
        request.encrypted_query_50,
        context,
        feature_count=50,
        batch_size=batch_size,
    )
    enc_selectors = load_feature_batch(
        request.encrypted_selectors,
        context,
        feature_count=k,
        batch_size=batch_size,
    )
    for column_index in range(max_size):
        mask = _sample_positive_mask_batch(batch_size)
        plain_column_j = matrix[:, column_index, :]  # (k, 50)
        # 1. Selector selection scaled by mask -> 50 feature ciphertexts of size batch_size
        selected_j = batch_select_ct_pt(enc_selectors, plain_column_j, mask)
        # 2. Dot product between query_50 and selected_j -> 1 score ciphertext of size batch_size
        score_j = batch_dot_ct_ct(enc_query_50, selected_j)
        # 3. Add -tau * mask (slot-wise)
        final_score_j = score_j + (-tau * mask).tolist()
        if serialize_output:
            yield final_score_j.serialize()
        else:
            yield final_score_j


def _load_tiled_public_context(public_context: ts.Context | bytes) -> ts.Context:
    if isinstance(public_context, bytes):
        context = ts.Context.load(public_context)
    elif isinstance(public_context, ts.Context):
        context = public_context
    else:
        raise ValueError(
            f"public_context must be TenSEAL Context or bytes, got "
            f"{type(public_context)}"
        )
    if context.is_private():
        raise ValueError("Party B must receive a public-only CKKS context")
    if not context.has_relin_keys():
        raise ValueError("public context is missing relinearization keys")
    return context


def _load_tiled_ciphertexts(
    ciphertexts: list[CipherLike],
    context: ts.Context,
    feature_count: int,
    layout: SlotTileLayout,
) -> list[ts.CKKSVector]:
    """Load V2 tiled ciphertexts and verify they have active_slots length."""
    if len(ciphertexts) != feature_count:
        raise ValueError(
            f"ciphertext count mismatch: expected {feature_count}, got {len(ciphertexts)}"
        )
    loaded = []
    for idx, item in enumerate(ciphertexts):
        if isinstance(item, bytes):
            vec = ts.ckks_vector_from(context, item)
        elif isinstance(item, ts.CKKSVector):
            vec = item
        else:
            raise ValueError(f"unsupported ciphertext type: {type(item)}")
        if vec.size() != layout.active_slots:
            raise ValueError(
                f"ciphertext[{idx}] size {vec.size()} != active_slots {layout.active_slots}"
            )
        loaded.append(vec)
    return loaded


def compare_tiled_batch_to_centroids(
    request: TiledFirstRoundRequest,
    centroids: np.ndarray,
    *,
    serialize_output: bool = False,
) -> list[EncryptedScalar]:
    """V2 Step 3: 二维 tiling 同时计算多个 centroid tiles。
    输出 ceil(k / T) 个密文，每个密文含 m*T 个 score slots。
    """
    matrix = np.asarray(centroids, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"centroids must be 2-D, got shape {matrix.shape}")
    if matrix.shape[1] != NUM_PERMUTATIONS_CLUSTER:
        raise ValueError(
            f"centroids second dimension must be {NUM_PERMUTATIONS_CLUSTER}, "
            f"got {matrix.shape[1]}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("centroids contains NaN or Inf")
    k = matrix.shape[0]
    if k <= 0:
        raise ValueError("centroids must contain at least 1 centroid")
    layout = request.layout
    if not isinstance(layout, SlotTileLayout):
        raise ValueError("request.layout must be a canonical SlotTileLayout")
    context = _load_tiled_public_context(request.public_context_bytes)
    enc_query_200 = _load_tiled_ciphertexts(
        request.encrypted_query_200,
        context,
        feature_count=NUM_PERMUTATIONS_CLUSTER,
        layout=layout,
    )
    score_tiles = []
    for _, start, valid_width in iter_tile_slices(k, layout):
        tile = pack_centroid_tile(matrix, layout, start, valid_width)
        # Compute dot product per feature and sum.
        acc = enc_query_200[0] * expand_plain_tile(tile[:, 0], layout).tolist()
        for f in range(1, NUM_PERMUTATIONS_CLUSTER):
            acc = acc + (
                enc_query_200[f] * expand_plain_tile(tile[:, f], layout).tolist()
            )
        if serialize_output:
            score_tiles.append(acc.serialize())
        else:
            score_tiles.append(acc)
    return score_tiles


def _sample_positive_mask_matrix(m: int, t: int) -> np.ndarray:
    """Sample an (m, t) matrix of independent positive random masks."""
    return np.array(
        [
            [_RNG.uniform(RANDOM_MASK_MIN, RANDOM_MASK_MAX) for _ in range(t)]
            for _ in range(m)
        ],
        dtype=np.float64,
    )


def tiled_batch_matching(
    cluster_matrix: np.ndarray,
    request: TiledSecondRoundRequest,
    public_context: ts.Context | bytes,
    tau: float = SIMILARITY_THRESHOLD,
    *,
    serialize_output: bool = False,
) -> Iterator[EncryptedScalar]:
    """V2 Step 6-8: 二维 tiling 同时计算多个 candidate columns。
    外层只遍历 ceil(L / T) tiles；每 tile 输出一个 active_slots 分数密文。
    每个 (query, column) 使用独立正随机 mask。
    """
    matrix = np.asarray(cluster_matrix, dtype=np.float64)
    if matrix.ndim != 3:
        raise ValueError(f"cluster_matrix must be 3-D, got shape {matrix.shape}")
    if matrix.shape[2] != 50:
        raise ValueError(
            f"cluster_matrix last dimension must be 50, got {matrix.shape[2]}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("cluster_matrix contains NaN or Inf")
    k, L, d = matrix.shape
    if d != NUM_PERMUTATIONS_MATCH:
        raise ValueError(f"feature dimension must be {NUM_PERMUTATIONS_MATCH}, got {d}")
    layout = request.layout
    if not isinstance(layout, SlotTileLayout):
        raise ValueError("request.layout must be a canonical SlotTileLayout")
    m = layout.batch_size
    T = layout.tile_width
    context = _load_tiled_public_context(public_context)
    enc_query_50 = _load_tiled_ciphertexts(
        request.encrypted_query_50,
        context,
        feature_count=NUM_PERMUTATIONS_MATCH,
        layout=layout,
    )
    enc_selectors = _load_tiled_ciphertexts(
        request.encrypted_selectors,
        context,
        feature_count=k,
        layout=layout,
    )
    for _, start, valid_width in iter_tile_slices(L, layout):
        # Sample independent positive masks for this tile.
        R = _sample_positive_mask_matrix(m, T)
        R[:, valid_width:] = (
            1.0  # padding columns: keep mask positive but scores zero later
        )
        # Construct candidate tile with zero padding.
        B_tile = pack_candidate_tile(matrix, layout, start, valid_width)
        # For each feature, compute selected_f = sum_i selector[i] * (R * B_tile[i,t,f])
        selected_features = []
        for f in range(d):
            acc_f = None
            for i in range(k):
                # Plaintext for cluster i: R[q,t] * B_tile[i,t,f] flattened q-major.
                plain_i = (B_tile[i, :, f] * R).reshape(-1)
                term = enc_selectors[i] * plain_i.tolist()
                acc_f = term if acc_f is None else acc_f + term
            selected_features.append(acc_f)
        # Dot product with query_50 features.
        score_tile = enc_query_50[0] * selected_features[0]
        for f in range(1, d):
            score_tile = score_tile + (enc_query_50[f] * selected_features[f])
        # Subtract tau * R flattened.
        flat_R = R.reshape(-1).tolist()
        score_tile = score_tile + (-tau * np.array(flat_R)).tolist()
        if serialize_output:
            yield score_tile.serialize()
        else:
            yield score_tile


# ---------------------------------------------------------------------------
# Single-query legacy functions (kept for backwards compatibility)
# ---------------------------------------------------------------------------
