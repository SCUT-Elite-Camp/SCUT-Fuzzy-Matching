#!/usr/bin/env python3
"""组级共享形状的中间曲线：tiles / 独立集召回 / 泄漏。

## 策略

| 策略 | 形状来源 | A 是否需要 50-bit 掩码 |
|---|---|---|
| 固定截断 | 全局常数 c | 否 |
| 随机分组 + 组内取 max | 组内需求最大值 | **是**（A 算不出 max） |
| 按需求排序分组 + 组内取 max | 同上 | **是** |
| 显式分带（两级） | 公开阈值 | **是** |
| 公开切分序列 + 随机分配 | 预定义切分，与查询无关 | 否 |

## 度量

- tiles = Σ_g ceil(n_g · cut_g / 4096)
- 召回 = 保留的 target（位置 < 该查询所属形状的宽度）
- 泄漏 = I(cluster; 观测到的形状)，用同一批查询的经验分布估计
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
K, TAU, S = 50, 0.9, 4096


def build_case():
    from config.params import NUM_PERMUTATIONS_CLUSTER as K200, NUM_PERMUTATIONS_MATCH as D50
    from minhash.encoder import batch_encode
    from scripts.evaluate_binarization_frontier import clean_name, l2_normalize
    from scripts.evaluate_cluster_conditional_filter import cosine_kmeans, select_clusters

    folder = ROOT / "data" / "ncvr_10k"
    if not folder.exists():
        folder = ROOT / "data"
    with (folder / "ncvr_10k_database.csv").open(newline="", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))[:10000]
    db_names = [clean_name(r["full_name"]) for r in rows]
    forbidden = set(db_names)
    db200 = batch_encode(db_names, K200)
    n200 = l2_normalize(db200)
    mu, sd = n200.mean(0), n200.std(0, ddof=0)
    sd = np.where(sd < 1e-10, 1.0, sd)
    cent, assign = cosine_kmeans((n200 - mu) / sd, K, seed=42)
    sizes = np.bincount(assign, minlength=K)
    rank = {}
    for i in range(K):
        for pos, row in enumerate(np.flatnonzero(assign == i)):
            rank[int(row)] = pos

    def perturb(name, rng):
        positions = [p for p, c in enumerate(name) if "a" <= c <= "z"]
        rng.shuffle(positions)
        for pos in positions:
            reps = [c for c in "abcdefghijklmnopqrstuvwxyz" if c != name[pos]]
            rng.shuffle(reps)
            for c in reps:
                cand = name[:pos] + c + name[pos + 1:]
                if cand not in forbidden:
                    return cand
        return None

    rng = random.Random(2026)
    tgts = rng.sample(range(len(db_names)), 200)
    qn, kept = [], []
    for t in tgts:
        q = perturb(db_names[t], rng)
        if q is not None:
            qn.append(q); kept.append(t)
    q200 = batch_encode(qn, K200)
    q_std = (l2_normalize(q200) - (mu if False else l2_normalize(db200).mean(0))) / np.where(
        l2_normalize(db200).std(0, ddof=0) < 1e-10, 1.0, l2_normalize(db200).std(0, ddof=0))
    # 复用与 evaluate_position_recall 相同的标准化
    q_std = (l2_normalize(q200) - mu) / sd
    sel = select_clusters(q_std, cent)
    qs = sizes[sel].astype(float)                       # 每条查询需要的列数
    pos = np.array([rank[int(t)] for t in kept])        # target 在簇内位置
    return qs, pos, sel, sizes, len(kept)


def tiles_of(shape_per_query, groups=None):
    """按组计算 tiles：每组 shpae = max(组内需求)（已给定），批大小 = 组大小。"""
    total = 0
    for grp, cut in groups:
        total += math.ceil(len(grp) * cut / S)
    return total


def mi(cluster, observed):
    """I(cluster; observed)，经验分布。"""
    from collections import Counter
    n = len(cluster)
    pc, po, pco = Counter(cluster), Counter(observed), Counter(zip(cluster, observed))
    h = 0.0
    for (c, o), cnt in pco.items():
        p = cnt / n
        h += p * math.log2(p / (pc[c] / n * po[o] / n))
    return max(0.0, h)


def main() -> int:
    qs, pos, sel, sizes, n = build_case()
    L = float(qs.max())
    rng = np.random.default_rng(11)
    rows = []

    def record(name, tiles, keep_mask, observed, needs_mask):
        rows.append({
            "strategy": name, "tiles": int(tiles),
            "mults": int(tiles * K * 50),
            "position_coverage": float(np.mean(keep_mask)),
            "leak_bits": float(mi(sel, observed)),
            "needs_mask": needs_mask,
        })

    # 1) 固定截断
    for c in (250, 300, 350, 400, 483):
        keep = pos < c
        obs = np.zeros(n, dtype=int)
        record(f"固定截断 c={c}", math.ceil(n * c / S), keep, obs, False)

    # 2) 显式分带（两级 cut=250）：Tier1 全量 + Tier2 只给大簇
    cut = 250
    t1 = math.ceil(n * cut / S)
    idx2 = np.flatnonzero(qs > cut)
    t2 = math.ceil(len(idx2) * (L - cut) / S)
    keep = (pos < cut) | np.isin(np.arange(n), idx2)
    obs = np.where(qs > cut, 1, 0)
    record("显式分带 cut=250", t1 + t2, keep, obs, True)

    # 3/4) 分组 + 组内取 max
    for mode in ("随机", "按需求排序"):
        for g in (2, 5, 10, 20, 50, 100):
            trials_t, trials_l, trials_r = [], [], []
            for _ in range(30):
                order = rng.permutation(n) if mode == "随机" else np.argsort(qs)
                groups, observed = [], np.zeros(n)
                for s0 in range(0, n, g):
                    grp = order[s0:s0 + g]
                    if len(grp) == 0:
                        continue
                    c_g = float(qs[grp].max())
                    groups.append((grp, c_g))
                    observed[grp] = c_g
                t = sum(math.ceil(len(gp) * cg / S) for gp, cg in groups)
                keep = pos < np.array([dict((int(i), cg) for gp, cg in groups for i in gp)[int(i)]
                                       for i in range(n)])
                trials_t.append(t); trials_l.append(mi(sel, observed)); trials_r.append(keep.mean())
            rows.append({"strategy": f"{mode}分组 g={g}", "tiles": float(np.median(trials_t)),
                         "mults": float(np.median(trials_t) * K * 50),
                         "position_coverage": float(np.median(trials_r)),
                         "leak_bits": float(np.median(trials_l)), "needs_mask": True})

    print(f"{'策略':<22}{'tiles':>8}{'乘法':>10}{'位置覆盖':>10}{'泄漏(bit)':>12}{'需掩码':>8}")
    for r in rows:
        print(f"{r['strategy']:<22}{r['tiles']:>8.0f}{r['mults']:>10,.0f}"
              f"{r['position_coverage']:>10.3f}{r['leak_bits']:>12.3f}{'是' if r['needs_mask'] else '否':>8}")

    out = ROOT / "artifacts" / "grouping" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": n, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\noutputs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
