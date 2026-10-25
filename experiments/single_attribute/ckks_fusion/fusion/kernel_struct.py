"""结构改造版 kernel：向量化明文预计算 + 平衡树归约。

## 两处结构变化（不改语义、不改形状、不改泄漏面）

1. **明文预计算向量化**：原实现对每个 (i,f) 单独做
   `(B_tile[i,:,f] * Rm).reshape(-1)`，共 k·d = 2,500 次 NumPy 广播；
   改为按特征一次性算整块：
   `Pf = B_tile[:, :, f][:, None, :] * Rm[None, :, :]` → (k, m*T)，
   把 2,500 次调用降为 50 次。

2. **selector 归约改为平衡树**：原实现是
   `((((t0+t1)+t2)+...)+t49)` 的串行链；改为两两配对归约，
   依赖深度从 k-1 降到 ceil(log2 k)。加法次数不变（k-1），
   但密文对象的分配顺序改变，可能影响内存与缓存行为。

语义完全一致：加法满足结合律，且 CT-CT 加法不消耗模数层级。
"""
from __future__ import annotations

import numpy as np
import tenseal as ts

from ckks.tiling import iter_tile_slices, pack_candidate_tile
from config.params import SIMILARITY_THRESHOLD
from party_b import online_responder as R


def _tree_sum(terms):
    """平衡树归约（保持原顺序的两两配对）。"""
    level = list(terms)
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(level[i] + level[i + 1])
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def tiled_batch_matching_struct(
    cluster_matrix, request, public_context, tau: float = SIMILARITY_THRESHOLD,
    *, serialize_output: bool = False,
):
    matrix = np.asarray(cluster_matrix, dtype=np.float64)
    k, L, d = matrix.shape
    layout = request.layout
    m, T = layout.batch_size, layout.tile_width
    ctx = R._load_tiled_public_context(public_context)
    enc_q = R._load_tiled_ciphertexts(request.encrypted_query_50, ctx, feature_count=d, layout=layout)
    enc_sel = R._load_tiled_ciphertexts(request.encrypted_selectors, ctx, feature_count=k, layout=layout)

    for _, start, vw in iter_tile_slices(L, layout):
        Rm = R._sample_positive_mask_matrix(m, T)
        Rm[:, vw:] = 1.0
        B = pack_candidate_tile(matrix, layout, start, vw)     # (k, T, d)

        feats = []
        for f in range(d):
            # 一次广播算出该特征下所有 cluster 的明文：(k, m, T) -> (k, m*T)
            Pf = (B[:, :, f][:, None, :] * Rm[None, :, :]).reshape(k, m * T)
            terms = [enc_sel[i] * ts.plain_tensor(Pf[i], dtype="float") for i in range(k)]
            feats.append(_tree_sum(terms))

        score = enc_q[0] * feats[0]
        for f in range(1, d):
            score += enc_q[f] * feats[f]
        score += ts.plain_tensor((Rm.reshape(-1) * -tau).astype(np.float64), dtype="float")
        yield score.serialize() if serialize_output else score


def tiled_batch_matching_struct_serial(
    cluster_matrix, request, public_context, tau: float = SIMILARITY_THRESHOLD,
    *, serialize_output: bool = False,
):
    """只保留向量化明文预计算，归约仍为增量累加（避免同时持有 k 个密文）。"""
    matrix = np.asarray(cluster_matrix, dtype=np.float64)
    k, L, d = matrix.shape
    layout = request.layout
    m, T = layout.batch_size, layout.tile_width
    ctx = R._load_tiled_public_context(public_context)
    enc_q = R._load_tiled_ciphertexts(request.encrypted_query_50, ctx, feature_count=d, layout=layout)
    enc_sel = R._load_tiled_ciphertexts(request.encrypted_selectors, ctx, feature_count=k, layout=layout)

    for _, start, vw in iter_tile_slices(L, layout):
        Rm = R._sample_positive_mask_matrix(m, T)
        Rm[:, vw:] = 1.0
        B = pack_candidate_tile(matrix, layout, start, vw)
        feats = []
        for f in range(d):
            Pf = (B[:, :, f][:, None, :] * Rm[None, :, :]).reshape(k, m * T)
            acc = enc_sel[0] * ts.plain_tensor(Pf[0], dtype="float")
            for i in range(1, k):
                acc += enc_sel[i] * ts.plain_tensor(Pf[i], dtype="float")
            feats.append(acc)
        score = enc_q[0] * feats[0]
        for f in range(1, d):
            score += enc_q[f] * feats[f]
        score += ts.plain_tensor((Rm.reshape(-1) * -tau).astype(np.float64), dtype="float")
        yield score.serialize() if serialize_output else score
