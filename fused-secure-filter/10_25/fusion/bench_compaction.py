#!/usr/bin/env python3
"""候选压缩原型：两级分带，只覆盖真实候选而非完整填充矩阵。

## 方案

当前固定形状：所有查询都按 L=483 列（最大簇）计算，有效数据只占 41.4%。

两级分带：
  Tier 1：全部 200 条查询，只算前 `c` 列（c = 公开阈值，如 200）
  Tier 2：仅簇大小 > c 的查询，算剩余 `483−c` 列

因为在 Tier 2 中查询集合取决于"谁的簇大"，B 会观察到**该查询簇大于 c**——
这是一条新的可观察信息（尺寸类别），但**不是簇身份**。

## 测量

- 密文乘法数、tile 数、端到端耗时（对比原始 25 tiles）
- 输出一致性：比较 sign(score)（与随机掩码无关）
- 新增可观察信息：Tier 2 的成员数

    python fusion/bench_compaction.py --cut 200
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

K, D = 50, 50


def build_real_case():
    from config.params import NUM_PERMUTATIONS_CLUSTER as K200, NUM_PERMUTATIONS_MATCH as D50
    from minhash.encoder import batch_encode
    from scripts.evaluate_binarization_frontier import l2_normalize, read_ncvr
    from scripts.evaluate_cluster_conditional_filter import cosine_kmeans, select_clusters

    db_names, _ids, entries = read_ncvr(
        ROOT / "data", query_limit=200, db_limit=10000, fuzzy_ratio=0.3, fuzzy_seed=42)
    db200 = batch_encode(db_names, K200)
    q200 = batch_encode([e.name for e in entries], K200)
    db50, q50 = l2_normalize(db200[:, :D50]), l2_normalize(q200[:, :D50])
    mu = l2_normalize(db200).mean(axis=0)
    sd = l2_normalize(db200).std(axis=0, ddof=0)
    sd = np.where(sd < 1e-10, 1.0, sd)
    db_std = (l2_normalize(db200) - mu) / sd
    q_std = (l2_normalize(q200) - mu) / sd
    cent, assign = cosine_kmeans(db_std, K, seed=42)
    sel = select_clusters(q_std, cent)
    sizes = np.bincount(assign, minlength=K)
    L = int(sizes.max())
    matrix = np.zeros((K, L, D))
    for i in range(K):
        members = np.flatnonzero(assign == i)
        matrix[i, :len(members)] = db50[members]
    return q50, matrix, sel, sizes, cent, q_std


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", type=int, default=200, help="第一级的列数（公开阈值）")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--truncate", action="store_true",
                    help="固定截断：所有查询只算前 cut 列，不做 Tier2（零路由泄漏）")
    ap.add_argument("--shuffle", action="store_true",
                    help="簇内记录位置随机置换，使真匹配位置不集中在簇首")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="小簇查询被额外路由到 Tier2 的概率 p（加噪路由）")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "compaction" / "report.json")
    args = ap.parse_args()

    try:
        import tenseal as ts
        from unittest.mock import patch
    except ImportError:
        print("tenseal 未安装")
        return 2

    from ckks.context import create_ckks_context
    from ckks.keys import serialize_public_context
    from ckks.tiling import (decrypt_tiled_score_tile, encrypt_tiled_feature_batch,
                             iter_tile_slices, make_slot_tile_layout)
    from config.params import SIMILARITY_THRESHOLD
    from party_a.online_querier import choose_clusters_and_build_tiled_request
    from party_b.online_responder import compare_tiled_batch_to_centroids, tiled_batch_matching
    from protocol.transport import (serialize_tiled_first_round_request,
                                    serialize_tiled_second_round_request)
    from protocol.types import TiledFirstRoundRequest, TiledPartyALocalState

    q50, matrix, sel, sizes, cent, q_std = build_real_case()
    if args.shuffle:
        rng_s = np.random.default_rng(7)
        for i in range(matrix.shape[0]):
            s_i = int(sizes[i])
            if s_i > 1:
                perm = rng_s.permutation(s_i)
                matrix[i, :s_i] = matrix[i, perm]
        print("已对簇内记录位置做随机置换")
    L = matrix.shape[1]
    cut = min(args.cut, L)
    print(f"簇大小: min={sizes.min()} mean={sizes.mean():.0f} max={L};  L={L}, cut={cut}")
    n2 = int(np.sum(sizes[sel] > cut))
    print(f"Tier2 查询数（簇>{cut}）: {n2} / {len(sel)}  ({n2/len(sel):.0%})")

    ctx = create_ckks_context()
    ctx.generate_relin_keys()
    pub = serialize_public_context(ctx)

    def run_tier(q_sub, q200_sub, cols_slice, m):
        """对给定查询子集、给定列区间跑一次 kernel，返回 (tiles, 耗时, scores)。"""
        layout = make_slot_tile_layout(m)
        enc50 = encrypt_tiled_feature_batch(q_sub, ctx, layout)
        state = TiledPartyALocalState(secret_context=ctx, encrypted_query_50=enc50, layout=layout)
        req1 = TiledFirstRoundRequest(public_context_bytes=pub,
                                      encrypted_query_200=encrypt_tiled_feature_batch(
                                          q200_sub, ctx, layout), layout=layout)
        w1 = serialize_tiled_first_round_request(req1)
        scores = compare_tiled_batch_to_centroids(w1, cent, serialize_output=True)
        req2, _ = choose_clusters_and_build_tiled_request(scores, state, k=K)
        w2 = serialize_tiled_second_round_request(req2)
        sub = matrix[:, cols_slice[0]:cols_slice[1], :]
        t0 = time.perf_counter()
        tiles = list(tiled_batch_matching(sub, w2, pub, tau=SIMILARITY_THRESHOLD, serialize_output=True))
        dt = time.perf_counter() - t0
        dec = np.concatenate([decrypt_tiled_score_tile(b, ctx, layout, w)
                              for b, (_, _, w) in zip(tiles, iter_tile_slices(sub.shape[1], layout))], axis=1)
        return len(tiles), dt, dec

    if args.truncate:
        t1_tiles, t1_time, s1 = run_tier(q50, q_std, (0, cut), len(q50))
        sign_full = np.sign(s1)
        print(f"\n固定截断 cut={cut}: {t1_tiles} tiles, {t1_tiles*K*D:,} 乘法, {t1_time:.2f}s")
        print("（无 Tier2 ⇒ 无路由泄漏；召回损失见 DESIGN_position_recall.md）")
        return 0

    idx2 = np.flatnonzero(sizes[sel] > cut)
    if args.noise > 0:
        rng_n = np.random.default_rng(2026)
        extra = np.flatnonzero((sizes[sel] <= cut) & (rng_n.random(len(sel)) < args.noise))
        idx2 = np.union1d(idx2, extra).astype(int)
        print(f"加噪 p={args.noise}: 额外路由 {len(extra)} 条小簇查询")
    base_runs, t1_runs, t2_runs = [], [], []
    for _ in range(args.repeats):
        _, bt, s_base = run_tier(q50, q_std, (0, L), len(q50)); base_runs.append(bt)
        t1_tiles, t1t, s1 = run_tier(q50, q_std, (0, cut), len(q50)); t1_runs.append(t1t)
        if len(idx2) > 0:
            t2_tiles, t2t, s2 = run_tier(q50[idx2], q_std[idx2], (cut, L), len(idx2)); t2_runs.append(t2t)
        else:
            t2_tiles, t2t, s2 = 0, 0.0, np.zeros((0, L - cut))
    base_tiles_m = math.ceil(L / (4096 // len(q50)))
    base_time = statistics.median(base_runs)
    t1_time = statistics.median(t1_runs)
    t2_time = statistics.median(t2_runs) if t2_runs else 0.0

    total_tiles = t1_tiles + t2_tiles
    total_time = t1_time + t2_time
    mults = total_tiles * K * D
    base_tiles = base_tiles_m
    base_mults = base_tiles * K * D
    # 输出一致性：拼接 sign(score) 与基线比较（R>0，符号与掩码无关）
    sign_base = np.sign(s_base)
    sign_cat = np.full((len(q50), L), np.nan)
    sign_cat[:, :cut] = np.sign(s1)[:, :cut]
    if len(idx2) > 0:
        sign_cat[idx2, cut:L] = np.sign(s2)[:, :L - cut]
    valid = ~np.isnan(sign_cat)
    agree = float(np.mean(sign_base[valid] == sign_cat[valid]))

    print(f"\n{'':<14}{'tiles':>8}{'乘法数':>12}{'耗时':>10}")
    print(f"{'基线(固定形状)':<14}{base_tiles:>8}{base_mults:>12,}{base_time:>9.2f}s")
    print(f"{'Tier1':<14}{t1_tiles:>8}{t1_tiles*K*D:>12,}{t1_time:>9.2f}s")
    print(f"{'Tier2':<14}{t2_tiles:>8}{t2_tiles*K*D:>12,}{t2_time:>9.2f}s")
    print(f"{'合计(压缩)':<14}{total_tiles:>8}{mults:>12,}{total_time:>9.2f}s")
    print(f"\n乘法下降 = {base_mults/mults:.3f}×   tile 下降 = {base_tiles/total_tiles:.3f}×")
    print(f"耗时变化 = {base_time:.2f}s → {total_time:.2f}s  ({base_time/total_time:.3f}×)")
    print(f"输出一致性（逐槽位符号） = {agree:.4f}")
    print(f"新增可观察信息：B 得知 {n2}/{len(q50)} 条查询的簇大小 > {cut}（尺寸类别，非簇身份）")

    payload = {
        "cut": cut, "L": L, "n_queries": len(q50), "tier2_count": n2,
        "baseline": {"tiles": base_tiles, "mults": base_mults},
        "tier1": {"tiles": t1_tiles, "mults": t1_tiles * K * D, "seconds": t1_time},
        "tier2": {"tiles": t2_tiles, "mults": t2_tiles * K * D, "seconds": t2_time},
        "total": {"tiles": total_tiles, "mults": mults, "seconds": total_time},
        "baseline_seconds": base_time,
        "mult_reduction": base_mults / mults,
        "time_ratio": base_time / total_time,
        "sign_agreement": agree,
        "new_observable": f"B 得知某查询簇大小 > {cut}",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"outputs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
