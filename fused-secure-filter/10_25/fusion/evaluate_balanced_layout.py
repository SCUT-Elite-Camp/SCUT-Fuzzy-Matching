#!/usr/bin/env python3
"""原始布局 vs 均衡布局：联合测量召回与真实耗时。

## 假设

均衡聚类能把 `max_size` 从 483 降到约 n/k≈199，从而把 tiles 25→10；
但重排簇成员会改变 Top-1 选择，可能损害召回。必须同条件联合测量。

## 判定线（取自项目既有约定，非事后设定）

`docs/approx_psi_2024_682_validation_plan.md:132`：
  - 测试集**不新增观察到的漏报**（逐查询配对）；
  - 工作量至少 2× 下降，或实测时间/通信至少 30% 下降。

## 不做的事

- 不丢弃任何记录、不截断大簇成员（那会人为降低 max_size）；
- 不改生产 baseline；
- 不重复造评估逻辑：编码、归一化、k-means、Top-1、二值码过滤全部复用既有实现。

## 用法

    python fusion/evaluate_balanced_layout.py                 # 只做召回对照
    python fusion/evaluate_balanced_layout.py --timing 5      # 追加真实计时（需 TenSEAL）
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.params import NUM_PERMUTATIONS_CLUSTER, NUM_PERMUTATIONS_MATCH  # noqa: E402
from minhash.encoder import batch_encode  # noqa: E402
from scripts.evaluate_binarization_frontier import l2_normalize, read_ncvr  # noqa: E402
from scripts.evaluate_cluster_conditional_filter import cosine_kmeans, select_clusters  # noqa: E402

SLOT_CAPACITY = 4096


def balanced_kmeans(
    matrix: np.ndarray, k: int, cap: int, iterations: int = 20, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """容量约束的球面 k-means：不丢弃任何点，只改变分配。

    做法：先用既有 `cosine_kmeans` 得到初始解，再做容量修复——
    对超容量的簇，按"离本簇质心最远"的顺序把多余点移往最近的、仍有容量的簇；
    然后重算质心，重复若干轮直到不再超容量。
    平局用簇编号打破，保证可复现。
    """
    unit = l2_normalize(matrix)
    centroids, assign = cosine_kmeans(matrix, k, iterations=iterations, seed=seed)
    assign = assign.astype(np.int64)

    for _ in range(iterations):
        sizes = np.bincount(assign, minlength=k)
        if sizes.max() <= cap:
            break
        # 超容量簇：按与本簇质心相似度升序（最远先走）
        for cluster in np.argsort(-sizes):
            sizes = np.bincount(assign, minlength=k)
            overflow = int(sizes[cluster] - cap)
            if overflow <= 0:
                continue
            members = np.flatnonzero(assign == cluster)
            sim = unit[members] @ centroids[cluster]
            order = members[np.argsort(sim, kind="stable")]
            for point in order[:overflow]:
                capacity = cap - np.bincount(assign, minlength=k)
                capacity[cluster] = 0  # 不能再回到本簇
                candidates = np.flatnonzero(capacity > 0)
                if candidates.size == 0:
                    break
                best = candidates[np.argmax(unit[point] @ centroids[candidates].T)]
                assign[point] = best
        # 重算质心
        new_centroids = np.zeros_like(centroids)
        for cluster in range(k):
            members = matrix[assign == cluster]
            if len(members) == 0:
                new_centroids[cluster] = centroids[cluster]
                continue
            mean_vec = members.mean(axis=0)
            norm = np.linalg.norm(mean_vec)
            new_centroids[cluster] = centroids[cluster] if norm < 1e-10 else mean_vec / norm
        centroids = l2_normalize(new_centroids)

    sizes = np.bincount(assign, minlength=k)
    if sizes.max() > cap:
        raise RuntimeError(f"均衡失败：max_size={sizes.max()} > cap={cap}")
    return centroids, assign


def evaluate(assignments: np.ndarray, selected: np.ndarray, survivors: list[np.ndarray],
             labels: np.ndarray, targets: np.ndarray) -> dict:
    in_cluster = np.zeros(len(labels), dtype=bool)
    survived = np.zeros(len(labels), dtype=bool)
    for i in range(len(labels)):
        if not labels[i] or targets[i] < 0:
            continue
        in_cluster[i] = assignments[targets[i]] == selected[i]
        survived[i] = in_cluster[i] and bool(np.isin(targets[i], survivors[i]))
    n_pos = int(labels.sum())
    n_in = int(in_cluster.sum())
    return {
        "n_positive": n_pos,
        "top1_recall": n_in / n_pos if n_pos else float("nan"),
        "in_cluster_filter_recall": float(survived.sum() / n_in) if n_in else float("nan"),
        "end_to_end_recall": float(survived.sum() / n_pos) if n_pos else float("nan"),
        "hit_mask": survived,
        "in_cluster_mask": in_cluster,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--clusters", type=int, default=50)
    ap.add_argument("--ell", type=int, default=512)
    ap.add_argument("--d", type=int, default=109)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--caps", type=int, nargs="+", default=None,
                    help="扫描多个容量上限，输出 max_size/tiles/召回 的折中表")
    ap.add_argument("--timing", type=int, default=0, help=">0 时追加真实计时，值为重复次数")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "balanced_layout")
    args = ap.parse_args()

    db_names, _ids, entries = read_ncvr(args.data_root, query_limit=200, db_limit=10000,
                                       fuzzy_ratio=0.3, fuzzy_seed=42)
    db200 = batch_encode(db_names, NUM_PERMUTATIONS_CLUSTER)
    q200 = batch_encode([e.name for e in entries], NUM_PERMUTATIONS_CLUSTER)
    db50, q50 = l2_normalize(db200[:, :NUM_PERMUTATIONS_MATCH]), l2_normalize(q200[:, :NUM_PERMUTATIONS_MATCH])

    mu, sd = l2_normalize(db200).mean(axis=0), l2_normalize(db200).std(axis=0, ddof=0)
    sd = np.where(sd < 1e-10, 1.0, sd)
    db_std, q_std = (l2_normalize(db200) - mu) / sd, (l2_normalize(q200) - mu) / sd

    planes = np.random.default_rng(2026).standard_normal((args.ell, NUM_PERMUTATIONS_MATCH))
    db_codes, q_codes = (db50 @ planes.T) >= 0, (q50 @ planes.T) >= 0

    labels = np.array([e.label for e in entries], dtype=bool)
    targets = np.array([e.target_index if e.target_index is not None else -1 for e in entries])

    # ---------- A. 原始布局 ----------
    cent_a, assign_a = cosine_kmeans(db_std, args.clusters, seed=args.seed)
    sel_a = select_clusters(q_std, cent_a)
    surv_a = []
    for i in range(len(entries)):
        m = np.flatnonzero(assign_a == sel_a[i])
        dist = (q_codes[i, : args.ell][None, :] != db_codes[m, : args.ell]).sum(axis=1)
        surv_a.append(m[dist <= args.d])

    # ---------- B. 均衡布局 ----------
    cap = math.ceil(len(db_names) / args.clusters)
    cent_b, assign_b = balanced_kmeans(db_std, args.clusters, cap, seed=args.seed)
    sel_b = select_clusters(q_std, cent_b)
    surv_b = []
    for i in range(len(entries)):
        m = np.flatnonzero(assign_b == sel_b[i])
        dist = (q_codes[i, : args.ell][None, :] != db_codes[m, : args.ell]).sum(axis=1)
        surv_b.append(m[dist <= args.d])

    ra = evaluate(assign_a, sel_a, surv_a, labels, targets)
    rb = evaluate(assign_b, sel_b, surv_b, labels, targets)

    sizes_a = np.bincount(assign_a, minlength=args.clusters)
    sizes_b = np.bincount(assign_b, minlength=args.clusters)
    tiles_a = math.ceil(int(sizes_a.max()) / (SLOT_CAPACITY // 200))
    tiles_b = math.ceil(int(sizes_b.max()) / (SLOT_CAPACITY // 200))

    if args.caps:
        rows = []
        for c in args.caps:
            _cent, _assign = balanced_kmeans(db_std, args.clusters, c, seed=args.seed)
            _sel = select_clusters(q_std, _cent)
            _surv = []
            for i in range(len(entries)):
                m = np.flatnonzero(_assign == _sel[i])
                dist = (q_codes[i, : args.ell][None, :] != db_codes[m, : args.ell]).sum(axis=1)
                _surv.append(m[dist <= args.d])
            _r = evaluate(_assign, _sel, _surv, labels, targets)
            sizes = np.bincount(_assign, minlength=args.clusters)
            rows.append({
                "cap": c, "max_size": int(sizes.max()),
                "tiles": math.ceil(int(sizes.max()) / (SLOT_CAPACITY // 200)),
                "top1": _r["top1_recall"], "filter": _r["in_cluster_filter_recall"],
                "e2e": _r["end_to_end_recall"],
                "new_misses": int(np.sum(ra["hit_mask"] & ~_r["hit_mask"])),
                "fixed": int(np.sum(~ra["hit_mask"] & _r["hit_mask"])),
            })
        print(f"{'cap':>5}{'max_size':>10}{'tiles':>7}{'Top-1':>8}{'簇内过滤':>9}{'端到端':>8}{'新增漏报':>9}{'修复':>6}")
        print("-" * 62)
        base = {"cap": "原始", "max_size": int(sizes_a.max()), "tiles": tiles_a,
                "top1": ra["top1_recall"], "filter": ra["in_cluster_filter_recall"],
                "e2e": ra["end_to_end_recall"], "new_misses": 0, "fixed": 0}
        print(f"{'原始':>5}{base['max_size']:>10}{base['tiles']:>7}{base['top1']:>8.3f}"
              f"{base['filter']:>9.3f}{base['e2e']:>8.3f}{0:>9}{0:>6}")
        for r in rows:
            print(f"{r['cap']:>5}{r['max_size']:>10}{r['tiles']:>7}{r['top1']:>8.3f}"
                  f"{r['filter']:>9.3f}{r['e2e']:>8.3f}{r['new_misses']:>9}{r['fixed']:>6}")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "cap_sweep.json").write_text(
            json.dumps({"baseline": base, "sweep": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0

    # 逐查询配对：新增漏报 / 修复
    hit_a, hit_b = ra["hit_mask"], rb["hit_mask"]
    pos = labels
    new_miss = int(np.sum(hit_a[pos] & ~hit_b[pos]))
    fixed = int(np.sum(~hit_a[pos] & hit_b[pos]))
    both_miss = int(np.sum(~hit_a[pos] & ~hit_b[pos]))

    result = {
        "config": {"clusters": args.clusters, "cap": cap, "ell": args.ell, "d": args.d,
                   "seed": args.seed, "n_db": len(db_names)},
        "layout_a": {"max_size": int(sizes_a.max()), "mean_size": float(sizes_a.mean()),
                     "p95": float(np.quantile(sizes_a, 0.95)), "tiles": tiles_a, **ra},
        "layout_b": {"max_size": int(sizes_b.max()), "mean_size": float(sizes_b.mean()),
                     "p95": float(np.quantile(sizes_b, 0.95)), "tiles": tiles_b, **rb},
        "paired": {"new_misses": new_miss, "fixed": fixed, "both_miss": both_miss,
                   "delta_recall_end_to_end": rb["end_to_end_recall"] - ra["end_to_end_recall"]},
    }
    for key in ("hit_mask", "in_cluster_mask"):
        result["layout_a"].pop(key, None)
        result["layout_b"].pop(key, None)

    print(f"cap = {cap}（n/k = {len(db_names)/args.clusters:.1f}）")
    print(f"{'布局':<8}{'max_size':>9}{'mean':>7}{'p95':>7}{'tiles':>7}"
          f"{'Top-1':>8}{'簇内过滤':>9}{'端到端':>8}")
    print("-" * 66)
    for name, r in (("原始", ra), ("均衡", rb)):
        lay = result["layout_a"] if name == "原始" else result["layout_b"]
        print(f"{name:<8}{lay['max_size']:>9}{lay['mean_size']:>7.1f}{lay['p95']:>7.1f}"
              f"{lay['tiles']:>7}{r['top1_recall']:>8.3f}"
              f"{r['in_cluster_filter_recall']:>9.3f}{r['end_to_end_recall']:>8.3f}")
    print(f"\n逐查询配对（{int(pos.sum())} 条正例）：新增漏报 {new_miss}，修复 {fixed}，均漏 {both_miss}")
    print(f"端到端召回变化：{rb['end_to_end_recall'] - ra['end_to_end_recall']:+.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "recall_compare.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.timing > 0:
        try:
            import tenseal  # noqa: F401
            from scripts.benchmark_he_batching import run_tiled_benchmark
        except ImportError:
            print("\n计时跳过：tenseal 未安装（召回对照结果仍然有效）")
            return 0
        cols = sorted({int(sizes_a.max()), int(sizes_b.max())})
        times_a, times_b = [], []
        for _ in range(args.timing):
            for c in cols:
                run_tiled_benchmark(batch_size=200, clusters=args.clusters, columns=c)  # 预热
            for c in cols:
                t = run_tiled_benchmark(batch_size=200, clusters=args.clusters, columns=c)
                (times_a if c == int(sizes_a.max()) else times_b).append(t["timing_sec"]["online_total"])
        result["timing"] = {
            "columns_a": int(sizes_a.max()), "columns_b": int(sizes_b.max()),
            "online_total_a_median": statistics.median(times_a),
            "online_total_b_median": statistics.median(times_b),
            "note": "同一 session 交错测量；kernel 无数据相关分支，形状决定耗时",
        }
        (args.out_dir / "recall_compare.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"计时：A {statistics.median(times_a):.2f}s vs B {statistics.median(times_b):.2f}s")

    print(f"\noutputs -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
