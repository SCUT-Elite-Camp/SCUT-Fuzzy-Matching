#!/usr/bin/env python3
"""量化随机掩码 R 对 A 的幅度保护：单次与重复查询。

## 模型

B 返回 y = R·(sim − τ)，R ~ U[1,10]，A 解密后看到 y。
A 不知道 R，但知道 R 的分布，且**可以重复提交同一条查询**。

单次观测给出 d ∈ [y/10, y]（因 R∈[1,10]）。
重复 T 次得到 y_1..y_T = R_i·d，故 d ∈ [max(y)/10, min(y)]。
当 T→∞，max(y)→10d、min(y)→d，区间收敛到 d 本身。

本脚本用真实相似度分布量化该收敛速度，并与"只给符号"作对比。

    python fusion/mask_leakage.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MASK_MIN, MASK_MAX = 1.0, 10.0
TAU = 0.9


def main() -> int:
    from config.params import NUM_PERMUTATIONS_CLUSTER as K200, NUM_PERMUTATIONS_MATCH as D50
    from minhash.encoder import batch_encode
    from scripts.evaluate_binarization_frontier import l2_normalize, read_ncvr
    from scripts.evaluate_cluster_conditional_filter import cosine_kmeans, select_clusters

    db_names, _ids, entries = read_ncvr(
        ROOT / "data", query_limit=200, db_limit=10000, fuzzy_ratio=0.3, fuzzy_seed=42
    )
    db200 = batch_encode(db_names, K200)
    q200 = batch_encode([e.name for e in entries], K200)
    db50, q50 = l2_normalize(db200[:, :D50]), l2_normalize(q200[:, :D50])

    mu = l2_normalize(db200).mean(axis=0)
    sd = l2_normalize(db200).std(axis=0, ddof=0)
    sd = np.where(sd < 1e-10, 1.0, sd)
    db_std, q_std = (l2_normalize(db200) - mu) / sd, (l2_normalize(q200) - mu) / sd

    cent, assign = cosine_kmeans(db_std, 50, seed=42)
    sel = select_clusters(q_std, cent)

    # 计算每条查询在其选中簇内的真实相似度（与 kernel 语义一致：q50 · B[t]）
    sims = []
    for i in range(len(entries)):
        members = np.flatnonzero(assign == sel[i])
        sims.append(q50[i] @ db50[members].T)
    sims = np.concatenate(sims)
    d = sims - TAU

    rng = np.random.default_rng(2026)
    print(f"候选总数 = {d.size}，其中 sim>τ 的 {int((d > 0).sum())} 条")
    print(f"d = sim−τ 的范围 [{d.min():.3f}, {d.max():.3f}]，|d| 中位数 {np.median(np.abs(d)):.3f}")
    prior_width = float(d.max() - d.min())
    print(f"先验区间宽度（A 未知 d 时的取值范围）= {prior_width:.3f}\n")

    rows = []
    print(f"{'T':>6}{'区间宽度中位数':>16}{'相对先验':>10}{'泄漏比特':>10}{'中位相对误差':>14}")
    for T in (1, 2, 5, 10, 50, 100, 500, 1000):
        y = rng.uniform(MASK_MIN, MASK_MAX, size=(T, d.size)) * d[None, :]
        # d>0: y∈[d,10d] ⇒ d∈[y/10, y]；d<0: y∈[10d,d] ⇒ d∈[y, y/10]
        pos = d > 0
        lo = np.where(pos, y.max(axis=0) / MASK_MAX, y.max(axis=0))
        hi = np.where(pos, y.min(axis=0), y.min(axis=0) / MASK_MAX)
        width = hi - lo
        est = (lo + hi) / 2.0
        rel_err = np.abs(est - d) / np.maximum(np.abs(d), 1e-6)
        med_w = float(np.median(width))
        bits = math.log2(prior_width / med_w) if med_w > 0 else float("inf")
        print(f"{T:>6}{med_w:>16.4f}{med_w/prior_width:>10.4f}{bits:>10.2f}{np.median(rel_err):>14.4f}")
        rows.append({"repeats": T, "median_width": med_w,
                     "relative_to_prior": med_w / prior_width, "bits_leaked": bits,
                     "median_relative_error": float(np.median(rel_err))})

    # 只给符号时的对照
    print(f"\n对照：只给符号 ⇒ A 只学到 sign(d)，泄漏 1 bit（二分类）。")
    print(f"      给掩码分数 ⇒ T=1 时宽度 {rows[0]['median_width']:.3f}（{rows[0]['bits_leaked']:.1f} bit），"
          f"T=500 时宽度 {rows[6]['median_width']:.4f}（{rows[6]['bits_leaked']:.1f} bit）")

    out = ROOT / "artifacts" / "mask_leakage" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "tau": TAU, "mask_range": [MASK_MIN, MASK_MAX],
        "n_candidates": int(d.size), "n_positive": int((d > 0).sum()),
        "prior_width": prior_width,
        "median_abs_d": float(np.median(np.abs(d))),
        "rows": rows,
        "note": "A 可重复提交同一查询；B 无速率限制。区间下界取 max(y)/10、上界取 min(y)。",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\noutputs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
