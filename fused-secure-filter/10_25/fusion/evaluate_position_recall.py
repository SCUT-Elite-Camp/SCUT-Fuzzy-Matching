#!/usr/bin/env python3
"""独立召回评估：用随机库记录构造查询，测真实 target 位置分布下的分带召回。

## 为什么需要

现有 `ncvr_10k_queries.csv` 的 target 全部落在簇内第 0–7 位，这是查询文件构造
（target 取自库中靠前记录）造成的偏差。用它评估"Tier 1 单独是否够用"会严重高估。

## 做法

- **聚类冻结**：k-means 只在库上跑（seed=42），与既有实验完全一致；
- **查询独立构造**：从库中随机抽 200 条记录作为 target，做单字符替换生成查询
  （替换规则与 `read_ncvr` 一致：替换后不得与库内任一词条重合）；
- 对每条查询：选簇、算 target 在簇内的真实位置。

## 输出

Recall@1（target 落在选中簇）、位置分位数、以及各 cut 下 Tier 1 单独的覆盖率。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

K, TAU = 50, 0.9


def perturb(name: str, rng: random.Random, forbidden: set[str]) -> str | None:
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


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuzzy-ratio", type=float, default=0.3,
                    help="扰动查询的比例（与 read_ncvr 的 fuzzy_ratio 对齐）")
    args = ap.parse_args()
    from config.params import NUM_PERMUTATIONS_CLUSTER as K200, NUM_PERMUTATIONS_MATCH as D50
    from minhash.encoder import batch_encode
    from scripts.evaluate_binarization_frontier import clean_name, l2_normalize
    from scripts.evaluate_cluster_conditional_filter import cosine_kmeans, select_clusters
    import csv

    folder = ROOT / "data" / "ncvr_10k"
    if not folder.exists():
        folder = ROOT / "data"
    with (folder / "ncvr_10k_database.csv").open(newline="", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))[:10000]
    db_names = [clean_name(r["full_name"]) for r in rows]
    forbidden = set(db_names)

    # 冻结聚类：只用库
    db200 = batch_encode(db_names, K200)
    db50 = l2_normalize(db200[:, :D50])
    n200 = l2_normalize(db200)
    mu, sd = n200.mean(axis=0), n200.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-10, 1.0, sd)
    db_std = (n200 - mu) / sd
    cent, assign = cosine_kmeans(db_std, K, seed=42)
    sizes = np.bincount(assign, minlength=K)
    members = [np.flatnonzero(assign == i) for i in range(K)]
    rank = {}
    for i in range(K):
        for pos, row in enumerate(members[i]):
            rank[int(row)] = pos

    # 随机 target + 扰动生成查询
    rng = random.Random(2026)
    targets = rng.sample(range(len(db_names)), 200)
    n_perturb = int(len(targets) * args.fuzzy_ratio + 0.5)
    perturbed = set(rng.sample(range(len(targets)), n_perturb))
    q_names, kept = [], []
    for idx, t in enumerate(targets):
        q = perturb(db_names[t], rng, forbidden) if idx in perturbed else db_names[t]
        if q is not None:
            q_names.append(q)
            kept.append(t)

    q200 = batch_encode(q_names, K200)
    q50 = l2_normalize(q200[:, :D50])
    q_std = (l2_normalize(q200) - mu) / sd
    sel = select_clusters(q_std, cent)

    in_cluster = np.array([assign[t] == sel[i] for i, t in enumerate(kept)])
    pos = np.array([rank[int(t)] for t in kept])
    L = int(sizes.max())

    # 相似度是否过阈值（真匹配）
    sim = np.array([float(q50[i] @ db50[t]) for i, t in enumerate(kept)])
    strong = sim >= TAU

    print(f"扰动比例 {args.fuzzy_ratio:.0%}；查询 {len(kept)} 条；库 {len(db_names)}；簇大小 max={L} 中位={int(np.median(sizes))}")
    print(f"相似度: 中位 {np.median(sim):.3f}，≥τ 的 {(strong).sum()}")
    print(f"Recall@1（target 在选中簇）= {in_cluster.mean():.3f}  ({in_cluster.sum()}/{len(kept)})")
    print(f"位置分位: 10%={np.quantile(pos,0.1):.0f} 25%={np.quantile(pos,0.25):.0f} "
          f"50%={np.quantile(pos,0.5):.0f} 75%={np.quantile(pos,0.75):.0f} 90%={np.quantile(pos,0.9):.0f} "
          f"max={pos.max()}")

    print(f"\n{'cut':>6}{'位置<cut 占比':>14}{'Tier1 单独召回':>16}{'损失':>8}")
    rows_out = []
    for cut in (50, 100, 150, 200, 250, 300, 350, 400, 450):
        cover = float(np.mean(pos < cut))
        t1_recall = float(np.mean(in_cluster & (pos < cut)))
        print(f"{cut:>6}{cover:>14.3f}{t1_recall:>16.3f}{in_cluster.mean()-t1_recall:>8.3f}")
        rows_out.append({"cut": cut, "position_cover": cover,
                         "tier1_only_recall": t1_recall,
                         "loss_vs_full": float(in_cluster.mean() - t1_recall)})

    out = ROOT / "artifacts" / "position_recall" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_queries": len(kept), "recall_at_1": float(in_cluster.mean()),
        "position_quantiles": {q: float(np.quantile(pos, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)},
        "rows": rows_out,
        "note": "查询由随机库记录单字符替换生成；聚类冻结（仅库上训练）",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\noutputs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
