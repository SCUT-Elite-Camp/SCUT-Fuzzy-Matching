#!/usr/bin/env python3
"""三组组合实验：原始基线 / 仅压缩 / 压缩+kernel_v2。

| 组 | 计算范围优化 | kernel_v2 |
|----|------------|-----------|
| A  | 关          | 关         |
| B  | 开(cut=250) | 关         |
| C  | 开(cut=250) | 开         |

同批查询、同种子、同 CKKS 参数；交错运行（每轮轮转顺序）取中位数。
记录：乘法数、分阶段耗时、端到端耗时、**逐槽位符号不一致数量**、B 可观察属性。

正交性依据：kernel_v2 只替换 tile 内部的操作数传递方式，签名与语义不变；
压缩只改变被处理的 (查询子集, 列区间)。两者作用在不同层次。

    python fusion/bench_combined.py --cut 250 --repeats 2
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", type=int, default=250)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "combined" / "report.json")
    args = ap.parse_args()

    try:
        import tenseal as ts  # noqa: F401
    except ImportError:
        print("tenseal 未安装")
        return 2

    from ckks.context import create_ckks_context
    from ckks.keys import serialize_public_context
    from ckks.tiling import (decrypt_tiled_score_tile, encrypt_tiled_feature_batch,
                             iter_tile_slices, make_slot_tile_layout)
    from config.params import SIMILARITY_THRESHOLD
    from fusion.bench_compaction import build_real_case
    from fusion.kernel_v2 import tiled_batch_matching_v2
    from party_a.online_querier import choose_clusters_and_build_tiled_request
    from party_b.online_responder import compare_tiled_batch_to_centroids, tiled_batch_matching
    from protocol.transport import (serialize_tiled_first_round_request,
                                    serialize_tiled_second_round_request)
    from protocol.types import TiledFirstRoundRequest, TiledPartyALocalState

    q50, matrix, sel, sizes, cent, q_std = build_real_case()
    L = int(matrix.shape[1])
    n = len(q50)
    cut = min(args.cut, L)

    ctx = create_ckks_context()
    ctx.generate_relin_keys()
    pub = serialize_public_context(ctx)

    def run_kernel(kernel, q_sub, q200_sub, c0, c1, m):
        layout = make_slot_tile_layout(m)
        enc50 = encrypt_tiled_feature_batch(q_sub, ctx, layout)
        state = TiledPartyALocalState(secret_context=ctx, encrypted_query_50=enc50, layout=layout)
        req1 = TiledFirstRoundRequest(
            public_context_bytes=pub,
            encrypted_query_200=encrypt_tiled_feature_batch(q200_sub, ctx, layout),
            layout=layout)
        w1 = serialize_tiled_first_round_request(req1)
        sc = compare_tiled_batch_to_centroids(w1, cent, serialize_output=True)
        req2, _ = choose_clusters_and_build_tiled_request(sc, state, k=K)
        w2 = serialize_tiled_second_round_request(req2)
        sub = matrix[:, c0:c1, :]
        t0 = time.perf_counter()
        tiles = list(kernel(sub, w2, pub, tau=SIMILARITY_THRESHOLD, serialize_output=True))
        dt = time.perf_counter() - t0
        dec = np.concatenate([decrypt_tiled_score_tile(b, ctx, layout, w)
                              for b, (_, _, w) in zip(tiles, iter_tile_slices(sub.shape[1], layout))],
                             axis=1)
        return len(tiles), dt, dec

    def group_A():
        t, dt, s = run_kernel(tiled_batch_matching, q50, q_std, 0, L, n)
        return {"tiles": t, "seconds": dt, "scores": s}

    def group_BC(kernel):
        idx2 = np.flatnonzero(sizes[sel] > cut)
        t1, d1, s1 = run_kernel(kernel, q50, q_std, 0, cut, n)
        if len(idx2):
            t2, d2, s2 = run_kernel(kernel, q50[idx2], q_std[idx2], cut, L, len(idx2))
        else:
            t2, d2, s2 = 0, 0.0, np.zeros((0, L - cut))
        # 拼接：未覆盖区域是填充列，score = −τR < 0 ⇒ 恒为负
        full = np.full((n, L), -1.0)
        full[:, :cut] = np.sign(s1)[:, :cut]
        if len(idx2):
            full[idx2, cut:] = np.sign(s2)[:, :L - cut]
        return {"tiles": t1 + t2, "seconds": d1 + d2, "scores": full,
                "tier1_s": d1, "tier2_s": d2, "tier2_queries": int(len(idx2))}

    def sign_reference(a_scores):
        return np.sign(a_scores)

    results = {"A": [], "B": [], "C": []}
    ref_sign = None
    mism = {"B": 0, "C": 0}
    for rep in range(args.repeats):
        order = ["A", "B", "C"] if rep % 2 == 0 else ["C", "B", "A"]
        for g in order:
            if g == "A":
                r = group_A()
                ref_sign = sign_reference(r["scores"])
            elif g == "B":
                r = group_BC(tiled_batch_matching)
                mism["B"] = int(np.sum(np.sign(r["scores"]) != ref_sign))
            else:
                r = group_BC(tiled_batch_matching_v2)
                mism["C"] = int(np.sum(np.sign(r["scores"]) != ref_sign))
            r.pop("scores", None)
            results[g].append(r)

    med = {g: {k: statistics.median([x[k] for x in v]) for k in v[0]}
           for g, v in results.items()}
    base_t = med["A"]["seconds"]
    print(f"cut={cut}, 正例查询 {n}, 重复 {args.repeats} 轮（交错）\n")
    print(f"{'组':<6}{'tiles':>7}{'乘法数':>10}{'端到端':>10}{'相对A':>8}{'符号不一致':>11}")
    for g in ("A", "B", "C"):
        m_ = med[g]
        mult = int(m_["tiles"] * K * D)
        rel = base_t / m_["seconds"]
        mm = "" if g == "A" else str(mism[g])
        print(f"{g:<6}{int(m_['tiles']):>7}{mult:>10,}{m_['seconds']:>9.2f}s{rel:>8.3f}{mm:>11}")

    mult_red = (med["A"]["tiles"] * K * D) / (med["B"]["tiles"] * K * D)
    v2_gain = med["B"]["seconds"] / med["C"]["seconds"]
    print(f"\n压缩贡献: 乘法 {mult_red:.3f}×   实现层(v2)贡献: 耗时 {v2_gain:.3f}×")
    print(f"组合总收益: 乘法 {(med['A']['tiles'])/(med['C']['tiles']):.3f}×   "
          f"耗时 {base_t/med['C']['seconds']:.3f}×")

    payload = {"cut": cut, "repeats": args.repeats, "median": med,
               "sign_mismatch": mism,
               "mult_reduction_compaction": mult_red,
               "v2_time_gain": v2_gain,
               "total_mult_reduction": med["A"]["tiles"] / med["C"]["tiles"],
               "total_time_gain": base_t / med["C"]["seconds"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"outputs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
