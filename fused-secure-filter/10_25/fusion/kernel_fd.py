"""可变特征维 kernel：与 tiled_batch_matching 语义一致，特征维 d 由矩阵宽度决定。

改动仅限三处：
  1. 去掉 `matrix.shape[2] != 50` 的硬断言，改用 d = matrix.shape[2]；
  2. 查询密文按 feature_count=d 加载；
  3. 特征循环范围从 50 改为 d。

其余（tile 切分、掩码、累加顺序、阈值项、序列化）逐行保持原样。
"""
from __future__ import annotations

import numpy as np
import tenseal as ts

from ckks.tiling import iter_tile_slices, pack_candidate_tile
from config.params import SIMILARITY_THRESHOLD
from party_b import online_responder as R


def tiled_batch_matching_d(
    cluster_matrix: np.ndarray,
    request,
    public_context,
    tau: float = SIMILARITY_THRESHOLD,
    *,
    serialize_output: bool = False,
):
    matrix = np.asarray(cluster_matrix, dtype=np.float64)
    if matrix.ndim != 3:
        raise ValueError(f"cluster_matrix must be 3-D, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("cluster_matrix contains NaN or Inf")
    k, L, d = matrix.shape
    layout = request.layout
    m, T = layout.batch_size, layout.tile_width
    ctx = R._load_tiled_public_context(public_context)
    enc_q = R._load_tiled_ciphertexts(
        request.encrypted_query_50, ctx, feature_count=d, layout=layout
    )
    enc_sel = R._load_tiled_ciphertexts(
        request.encrypted_selectors, ctx, feature_count=k, layout=layout
    )
    for _, start, vw in iter_tile_slices(L, layout):
        Rm = R._sample_positive_mask_matrix(m, T)
        Rm[:, vw:] = 1.0
        B_tile = pack_candidate_tile(matrix, layout, start, vw)
        feats = []
        for f in range(d):
            acc = None
            for i in range(k):
                plain = (B_tile[i, :, f] * Rm).reshape(-1)
                term = enc_sel[i] * ts.plain_tensor(plain, dtype="float")
                acc = term if acc is None else acc + term
            feats.append(acc)
        score = enc_q[0] * feats[0]
        for f in range(1, d):
            score += enc_q[f] * feats[f]
        score += ts.plain_tensor((Rm.reshape(-1) * -tau).astype(np.float64), dtype="float")
        yield score.serialize() if serialize_output else score
