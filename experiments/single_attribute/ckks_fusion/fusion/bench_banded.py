#!/usr/bin/env python3
"""压缩上界原型：按精确簇大小贪心分带（会泄漏精确大小）。

用途：测量"如果 B 知道每条查询的精确簇大小，最多能压到多少"，
据此判断现有 cut=250 方案还剩多少空间、以及那部分空间的价格。

注意：本方案泄漏精确尺寸（约 5.46 bit，接近簇身份），**不是可用方案**，
仅作为上界参照。

    python fusion/bench_banded.py
"""

from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
K, D, S = 50, 50, 4096


def build_bands(qs):
    """按大小排序后贪心合并，目标是最小化 Σ ceil(m_band · max_size / S)。"""
    order = np.argsort(qs)
    bands, i = [], 0
    while i < len(order):
        best_j, best_t = i + 1, math.ceil(qs[order[i]] / S)
        for j in range(i + 1, len(order) + 1):
            t = math.ceil((j - i) * qs[order[j - 1]] / S)
            if t <= best_t:
                best_j, best_t = j, t
            else:
                break
        bands.append((order[i:best_j], best_t))
        i = best_j
    return bands


def main() -> int:
    try:
        import tenseal  # noqa: F401
    except ImportError:
        print("tenseal 未安装")
        return 2

    from ckks.context import create_ckks_context
    from ckks.keys import serialize_public_context
    from ckks.tiling import (decrypt_tiled_score_tile, encrypt_tiled_feature_batch,
                             iter_tile_slices, make_slot_tile_layout)
    from config.params import SIMILARITY_THRESHOLD
    from fusion.bench_compaction import build_real_case
    from party_a.online_querier import choose_clusters_and_build_tiled_request
    from party_b.online_responder import compare_tiled_batch_to_centroids, tiled_batch_matching
    from protocol.transport import (serialize_tiled_first_round_request,
                                    serialize_tiled_second_round_request)
    from protocol.types import TiledFirstRoundRequest, TiledPartyALocalState

    q50, matrix, sel, sizes, cent, q_std = build_real_case()
    L, n = int(matrix.shape[1]), len(q50)
    qs = sizes[sel]

    ctx = create_ckks_context()
    ctx.generate_relin_keys()
    pub = serialize_public_context(ctx)

    def run(idx, c1):
        m = len(idx)
        layout = make_slot_tile_layout(m)
        enc50 = encrypt_tiled_feature_batch(q50[idx], ctx, layout)
        rng = np.random.default_rng(1)
        st = TiledPartyALocalState(secret_context=ctx, encrypted_query_50=enc50, layout=layout)
        w1 = serialize_tiled_first_round_request(TiledFirstRoundRequest(
            public_context_bytes=pub,
            encrypted_query_200=encrypt_tiled_feature_batch(q_std[idx], ctx, layout),
            layout=layout))
        sc = compare_tiled_batch_to_centroids(w1, cent, serialize_output=True)
        req2, _ = choose_clusters_and_build_tiled_request(sc, st, k=K)
        w2 = serialize_tiled_second_round_request(req2)
        sub = matrix[:, :c1, :]
        t0 = time.perf_counter()
        tiles = list(tiled_batch_matching(sub, w2, pub, tau=SIMILARITY_THRESHOLD, serialize_output=True))
        dt = time.perf_counter() - t0
        dec = np.concatenate([decrypt_tiled_score_tile(b, ctx, layout, w)
                              for b, (_, _, w) in zip(tiles, iter_tile_slices(c1, layout))], axis=1)
        return len(tiles), dt, dec

    # 基线
    t_base, d_base, s_base = run(np.arange(n), L)
    # 分带
    bands = build_bands(qs)
    full = np.full((n, L), -1.0)
    total_t, total_d, n_tiles = 0, 0.0, 0
    for idx, _want in bands:
        c1 = int(qs[idx].max())
        t, d, s = run(idx, c1)
        total_t += t; total_d += d; n_tiles += t
        full[idx, :c1] = np.sign(s)[:, :c1]

    mism = int(np.sum(np.sign(s_base) != full))
    print(f"查询 {n} 条，基线 {t_base} tiles，分带 {n_tiles} tiles（{len(bands)} 带）")
    print(f"基线耗时 {d_base:.2f}s，分带耗时 {total_d:.2f}s  → {d_base/total_d:.3f}×")
    print(f"tiles 下降 {t_base/n_tiles:.3f}×   乘法数 {t_base*K*D:,} → {n_tiles*K*D:,}")
    print(f"逐槽位符号不一致: {mism} / {n*L}")
    print(f"泄漏代价: 精确大小 H={math.log2(len(set(sizes.tolist()))):.3f} bit（簇身份 5.644 bit）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
