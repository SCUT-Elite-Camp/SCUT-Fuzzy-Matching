"""kernel 的实验分支实现：与基线语义完全一致，只改明文操作数的传递方式。

## 为什么做这个

实测（25 tiles → online 阶段 83.06 s）中 `round2_b_kernel` 占 96.5%；
kernel 内每 tile 做 k·d = 2,500 次密文×明文乘法，占乘法总数的 98%
（dot 只需 d=50 次）。单次 ct-pt 乘法约 1.08 ms，且几乎不随明文长度变化
（500→4000 元素仅差 0.2 ms），说明已贴近 NTT 下限。

因此可压缩的只剩**每次调用的固定开销**：TenSEAL 的 `*` 接收 Python list 时会
先做 `plain_tensor` 转换。改为直接传 `ts.plain_tensor`，省掉重复的 list 物化。

## 语义保证

- 相同的 mask 采样器、相同的 tile 顺序、相同的累加顺序；
- 输出仍是"每 tile 一个分数密文"，值相同（由 `bench_kernel_v2.py` 逐元素校验）；
- 不改变泄漏面：仍是一个加密 one-hot selector，B 仍不知道选中簇。
"""

from __future__ import annotations

import numpy as np
import tenseal as ts

from ckks.tiling import iter_tile_slices, pack_candidate_tile
from config.params import SIMILARITY_THRESHOLD
from party_b import online_responder as R


def tiled_batch_matching_v2(
    cluster_matrix: np.ndarray,
    request,
    public_context,
    tau: float = SIMILARITY_THRESHOLD,
    *,
    serialize_output: bool = False,
):
    matrix = np.asarray(cluster_matrix, dtype=np.float64)
    if matrix.ndim != 3 or matrix.shape[2] != 50:
        raise ValueError("cluster_matrix 形状不符")
    k, L, d = matrix.shape
    layout = request.layout
    m, T = layout.batch_size, layout.tile_width
    context = R._load_tiled_public_context(public_context)
    enc_query_50 = R._load_tiled_ciphertexts(
        request.encrypted_query_50, context, feature_count=50, layout=layout
    )
    enc_selectors = R._load_tiled_ciphertexts(
        request.encrypted_selectors, context, feature_count=k, layout=layout
    )
    for _, start, valid_width in iter_tile_slices(L, layout):
        Rm = R._sample_positive_mask_matrix(m, T)
        Rm[:, valid_width:] = 1.0
        B_tile = pack_candidate_tile(matrix, layout, start, valid_width)
        selected_features = []
        for f in range(d):
            acc_f = None
            for i in range(k):
                plain_i = (B_tile[i, :, f] * Rm).reshape(-1)
                # 唯一改动：直接传 plain_tensor，避免 TenSEAL 内部把 list 再转一遍
                term = enc_selectors[i] * ts.plain_tensor(plain_i, dtype="float")
                if acc_f is None:
                    acc_f = term
                else:
                    acc_f += term   # in-place，避免 k-1 次新密文分配
            selected_features.append(acc_f)
        score_tile = enc_query_50[0] * selected_features[0]
        for f in range(1, d):
            score_tile += enc_query_50[f] * selected_features[f]
        flat_R = (Rm.reshape(-1) * -tau).astype(np.float64)
        score_tile = score_tile + ts.plain_tensor(flat_R, dtype="float")
        yield score_tile.serialize() if serialize_output else score_tile
