#!/usr/bin/env python3
"""簇内候选过滤评估：二值码在**真实候选分布**上能减少多少候选。

上一版 `evaluate_binarization_frontier.py` 在全库随机负例上定阈值，得到的 5%
全局 FPR 并不能代表簇内情形。这个脚本补上那一环：

  1. 复刻生产路径的离线聚类（标准化 200 维 + 球面 k-means + Top-1），
     得到每条查询**真实**的候选集（= 选中簇的成员）。
  2. 在这个候选集上，用二值码（随机超平面 + Hamming <= d）过滤，
     阈值按"真目标召回 >= R"在**开发集**上选取，再在**测试集**上评估。
  3. 报告候选数分布（mean/median/p95/max）与等价的 tiled kernel 工作量，
     对比 `ceil(L / T)`（生产路径按最长簇补齐后的 tile 数）。

## 为什么不复用 sklearn

生产的 `run_cosine_kmeans` 用 `sklearn.metrics.cosine_similarity`，`fit_scaler` 用
`StandardScaler`。两者都能用 numpy 精确复刻（归一化后点积；总体标准差 ddof=0），
这里内联实现，`cluster_filter.json` 的 `runtime` 字段记录走的是哪条路径。

## 阈值为什么必须在开发集上选

若在全部查询上取"满足召回的最小 d"，再在同一批查询上报候选数，就是选择性偏差
（上一版已经踩过一次）。这里的做法：查询按标签分层切成 dev / test，d 只在 dev 上定，
所有上报数字都在 test 上。

## 用法

    python scripts/evaluate_cluster_conditional_filter.py \
        --ells 64 128 256 512 --target-recall 0.98

产物（默认 `artifacts/cluster_conditional_filter/`）：
    cluster_filter.csv   每个 (ell, d) 的 dev/test 召回与候选数
    per_query.csv        测试集每条查询的候选前后对比
    cluster_filter.json  配置、簇统计、逐 ell 工作点
    summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.params import (  # noqa: E402
    KMEANS_ITERATIONS,
    NUM_PERMUTATIONS_CLUSTER,
    NUM_PERMUTATIONS_MATCH,
)
from minhash.encoder import batch_encode  # noqa: E402

from scripts.evaluate_binarization_frontier import (  # noqa: E402
    _NORMALIZER_SOURCE,
    l2_normalize,
    read_ncvr,
)

SLOT_CAPACITY = 4096  # POLY_MODULUS_DEGREE // 2


# ---------------------------------------------------------------------------
# 生产路径的 numpy 复刻
# ---------------------------------------------------------------------------


def standardize(matrix: np.ndarray) -> np.ndarray:
    """等价于 sklearn StandardScaler（总体标准差 ddof=0）。"""
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=0)
    scale = np.where(scale < 1e-10, 1.0, scale)
    return (matrix - mean) / scale


def cosine_kmeans(
    matrix: np.ndarray, k: int, iterations: int = KMEANS_ITERATIONS, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """复刻 `clustering.kmeans_cosine.run_cosine_kmeans` 的球面 k-means。"""
    data = np.asarray(matrix, dtype=np.float64)
    unit = l2_normalize(data)
    rng = np.random.RandomState(seed)
    centroids = unit[rng.choice(len(unit), size=min(k, len(unit)), replace=False)].copy()
    assignments = np.zeros(len(unit), dtype=np.int32)

    for _ in range(iterations):
        previous = centroids.copy()
        sim = unit @ centroids.T
        assignments = np.argmax(sim, axis=1).astype(np.int32)
        updated = np.zeros_like(centroids)
        empty: list[int] = []
        for index in range(k):
            members = data[assignments == index]
            if len(members) == 0:
                empty.append(index)
                continue
            mean_vec = members.mean(axis=0)
            norm = np.linalg.norm(mean_vec)
            updated[index] = previous[index] if norm < 1e-10 else mean_vec / norm
        # 空簇：把当前最"不受欢迎"的点拉过去（与生产实现同思路）
        if empty:
            assigned_sims = sim[np.arange(len(data)), assignments]
            for index in empty:
                farthest = int(np.argmax(1.0 - assigned_sims))
                updated[index] = l2_normalize(data[farthest : farthest + 1])[0]
        centroids = l2_normalize(updated)
    sim = unit @ centroids.T
    assignments = np.argmax(sim, axis=1).astype(np.int32)
    return centroids, assignments


def select_clusters(
    query_standardized: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    """Top-1：查询与质心的余弦（标准化空间），与生产 A 侧一致。"""
    sim = l2_normalize(query_standardized) @ l2_normalize(centroids).T
    return np.argmax(sim, axis=1).astype(np.int64)


# ---------------------------------------------------------------------------


def quantiles(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--query-limit", type=int, default=200)
    parser.add_argument("--db-limit", type=int, default=10000)
    parser.add_argument("--fuzzy-ratio", type=float, default=0.3)
    parser.add_argument("--fuzzy-seed", type=int, default=42)
    parser.add_argument("--clusters", type=int, default=50)
    parser.add_argument("--ells", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--target-recall", type=float, default=0.98)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--plane-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts" / "cluster_conditional_filter")
    args = parser.parse_args(argv)

    ells = sorted({int(v) for v in args.ells if v > 0})
    tile_width = SLOT_CAPACITY // args.batch_size

    # ---------------- 1. 数据与编码 ----------------
    db_names, db_ids, entries = read_ncvr(
        args.data_root,
        query_limit=args.query_limit,
        db_limit=args.db_limit,
        fuzzy_ratio=args.fuzzy_ratio,
        fuzzy_seed=args.fuzzy_seed,
    )
    # 聚类用 200 维（生产配置），匹配与过滤用其前 50 维切片
    db_sig200 = batch_encode(db_names, NUM_PERMUTATIONS_CLUSTER)
    q_sig200 = batch_encode([e.name for e in entries], NUM_PERMUTATIONS_CLUSTER)
    db_vec = l2_normalize(db_sig200[:, :NUM_PERMUTATIONS_MATCH])
    q_vec = l2_normalize(q_sig200[:, :NUM_PERMUTATIONS_MATCH])

    # 生产路径：`fit_scaler` 只在 B 侧拟合，A 侧复用同一组 mean/scale
    # （benchmark.py 把 scaler_mean/scaler_scale 传给 prepare_tiled_query_batch）。
    # 这里必须同样处理，查询不能自己重新标准化。
    db_norm200 = l2_normalize(db_sig200)
    q_norm200 = l2_normalize(q_sig200)
    scaler_mean = db_norm200.mean(axis=0)
    scaler_scale = db_norm200.std(axis=0, ddof=0)
    scaler_scale = np.where(scaler_scale < 1e-10, 1.0, scaler_scale)
    db_std = (db_norm200 - scaler_mean) / scaler_scale
    q_std = (q_norm200 - scaler_mean) / scaler_scale

    # ---------------- 2. 聚类 + Top-1 ----------------
    centroids, assignments = cosine_kmeans(db_std, args.clusters)
    selected = select_clusters(q_std, centroids)
    cluster_sizes = np.bincount(assignments, minlength=args.clusters)
    max_size = int(cluster_sizes.max())

    labels = np.array([e.label for e in entries], dtype=bool)
    id_codes: dict[str, int] = {}
    for value in db_ids + [e.ncid for e in entries]:
        id_codes.setdefault(value, len(id_codes))
    db_codes = np.array([id_codes[v] for v in db_ids], dtype=np.int64)
    q_codes = np.array([id_codes[e.ncid] for e in entries], dtype=np.int64)
    targets = np.array(
        [e.target_index if e.target_index is not None else -1 for e in entries], dtype=np.int64
    )
    # 目标是否落在选中簇（簇召回）
    target_in_cluster = np.zeros(len(entries), dtype=bool)
    for i, entry in enumerate(entries):
        if entry.label and entry.target_index is not None:
            target_in_cluster[i] = assignments[entry.target_index] == selected[i]
    positives = np.flatnonzero(labels)
    cluster_recall = float(target_in_cluster[positives].mean())

    # ---------------- 3. 二值码 ----------------
    ell_max = max(ells)
    planes = np.random.default_rng(args.plane_seed).standard_normal((ell_max, NUM_PERMUTATIONS_MATCH))
    db_bits = (db_vec @ planes.T) >= 0.0
    q_bits = (q_vec @ planes.T) >= 0.0

    # ---------------- 4. 逐 ell：阈值只在 dev 上选，多次切分聚合 ----------------
    def cluster_members(query_index: int) -> np.ndarray:
        return np.flatnonzero(assignments == selected[query_index])

    def hamming_to_cluster(query_index: int, ell: int) -> tuple[np.ndarray, np.ndarray]:
        members = cluster_members(query_index)
        if members.size == 0:
            return members, np.zeros(0, dtype=np.int32)
        dist = (q_bits[query_index, :ell][None, :] != db_bits[members, :ell]).sum(axis=1)
        return members, dist.astype(np.int32)

    # 每条查询的簇内距离只算一次；切分只改变 dev/test，不改变距离
    all_dists = {(ell, i): hamming_to_cluster(i, ell)[1] for ell in ells for i in range(len(entries))}
    all_logical_width = {}

    def stratified_split(seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng_local = np.random.default_rng(seed)
        dev_l, test_l = [], []
        for label_value in (True, False):
            pool = np.flatnonzero(labels == label_value)
            shuffled = rng_local.permutation(pool)
            half = len(shuffled) // 2
            dev_l.extend(shuffled[:half].tolist())
            test_l.extend(shuffled[half:].tolist())
        return np.array(sorted(dev_l)), np.array(sorted(test_l))

    split_seeds = [args.split_seed + 1000 * k for k in range(args.splits)]
    results = []
    per_query_rows = []
    curve_rows = []

    for ell in ells:
        max_d = int(min(ell, 200))
        d_values = list(range(0, max_d + 1))
        split_rows = []

        for split_index, split_seed in enumerate(split_seeds):
            dev_idx, test_idx = stratified_split(split_seed)
            dev_pos = [i for i in dev_idx if labels[i] and target_in_cluster[i]]

            dev_dists = {i: all_dists[(ell, i)] for i in dev_idx}
            dev_target_dist = {}
            for i in dev_pos:
                members = cluster_members(i)
                pos = int(np.searchsorted(members, entries[i].target_index))
                dev_target_dist[i] = int(dev_dists[i][pos])

            d_star = None
            for d in d_values:
                recall_dev = (
                    float(np.mean([dist <= d for dist in dev_target_dist.values()]))
                    if dev_target_dist else float("nan")
                )
                kept = sum(int((dist <= d).sum()) for dist in dev_dists.values())
                if split_index == 0:
                    curve_rows.append({
                        "ell": ell, "d": d, "dev_recall": recall_dev,
                        "dev_mean_candidates": kept / max(1, len(dev_idx)),
                    })
                if recall_dev >= args.target_recall:
                    d_star = d
                    break
            if d_star is None:
                d_star = max_d

            kept_counts, base_counts = [], []
            survived, n_in_cluster = 0, 0
            survived_e2e, n_pos_test = 0, 0
            survived_pert, n_pert = 0, 0
            survived_infeed = 0
            for i in test_idx:
                members = cluster_members(i)
                dist = all_dists[(ell, i)]
                kept_counts.append(int((dist <= d_star).sum()))
                base_counts.append(len(members))
                if not labels[i]:
                    continue
                n_pos_test += 1
                ok = False
                if target_in_cluster[i]:
                    n_in_cluster += 1
                    pos = int(np.searchsorted(members, entries[i].target_index))
                    ok = bool(dist[pos] <= d_star)
                    survived += int(ok)
                    survived_infeed += 1
                survived_e2e += int(ok)
                if entries[i].perturbed:
                    n_pert += 1
                    survived_pert += int(ok)
                if split_index == 0:
                    per_query_rows.append({
                        "ell": ell, "query_ncid": entries[i].ncid,
                        "perturbed": entries[i].perturbed,
                        "target_in_cluster": bool(target_in_cluster[i]),
                        "baseline_candidates": len(members),
                        "filtered_candidates": int((dist <= d_star).sum()),
                        "target_survived": ok,
                        "target_hamming": int(dist[np.searchsorted(members, entries[i].target_index)])
                        if target_in_cluster[i] else None,
                        "d_star": d_star,
                    })

            kept_arr = np.array(kept_counts, dtype=np.float64)
            base_arr = np.array(base_counts, dtype=np.float64)
            split_rows.append({
                "d_star": d_star,
                # 条件召回：只在"真目标已进入选中簇"的正例上统计
                "recall_cond": survived / max(1, n_in_cluster),
                # 端到端：分母是全部正例
                "recall_e2e": survived_e2e / max(1, n_pos_test),
                "recall_perturbed": survived_pert / max(1, n_pert),
                "n_pos_test": n_pos_test,
                "n_in_cluster": n_in_cluster,
                "n_test_perturbed": n_pert,
                "base_mean": float(base_arr.mean()),
                "filt_mean": float(kept_arr.mean()),
                "filt_p95": float(np.quantile(kept_arr, 0.95)),
                "filt_max": float(kept_arr.max()),
            })

        # 聚合：召回取均值与最差，候选数取均值，宽度取所有切分里的最大
        recalls = [r["recall_cond"] for r in split_rows]
        recalls_e2e = [r["recall_e2e"] for r in split_rows]
        recalls_pert = [r["recall_perturbed"] for r in split_rows if r["n_test_perturbed"] > 0]
        d_stars = [r["d_star"] for r in split_rows]
        filt_mean = float(np.mean([r["filt_mean"] for r in split_rows]))
        base_mean = float(np.mean([r["base_mean"] for r in split_rows]))
        filt_max = float(max(r["filt_max"] for r in split_rows))
        filt_p95 = float(np.mean([r["filt_p95"] for r in split_rows]))

        # 负例：实体身份不相交（不是 label=False）；真目标单独统计，不得混入负例率
        retained_neg: dict[str, float] = {}
        retained_pos: dict[str, float] = {}
        edges = np.array([0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.001])
        labels_bin = [f"{edges[k]:.2f}-{edges[k+1]:.2f}" for k in range(len(edges) - 1)]
        n_bins = len(edges) - 1
        neg_tot = np.zeros(n_bins)
        neg_kept = np.zeros(n_bins)
        pos_tot = np.zeros(n_bins)
        pos_kept = np.zeros(n_bins)
        d_med = int(np.median(d_stars))
        for i in range(len(entries)):
            members = cluster_members(i)
            if members.size == 0:
                continue
            dist = all_dists[(ell, i)]
            rho = q_vec[i] @ db_vec[members].T
            idx = np.clip(np.digitize(rho, edges[1:-1]), 0, n_bins - 1)
            is_neg = db_codes[members] != q_codes[i]
            np.add.at(neg_tot, idx[is_neg], 1.0)
            np.add.at(neg_kept, idx[is_neg], (dist[is_neg] <= d_med).astype(np.float64))
            if labels[i] and target_in_cluster[i]:
                pos = int(np.searchsorted(members, entries[i].target_index))
                np.add.at(pos_tot, idx[pos], 1.0)
                np.add.at(pos_kept, idx[pos], float(dist[pos] <= d_med))
        for k in range(n_bins):
            if neg_tot[k] >= 50:
                retained_neg[labels_bin[k]] = float(neg_kept[k] / neg_tot[k])
            if pos_tot[k] >= 3:
                retained_pos[labels_bin[k]] = float(pos_kept[k] / pos_tot[k])

        results.append({
            "ell": int(ell),
            "d_star_median": d_med,
            "d_star_range": [int(min(d_stars)), int(max(d_stars))],
            "recall_cond_mean": float(np.mean(recalls)),
            "recall_cond_min": float(min(recalls)),
            "recall_e2e_mean": float(np.mean(recalls_e2e)),
            "recall_e2e_min": float(min(recalls_e2e)),
            "recall_perturbed_mean": float(np.mean(recalls_pert)) if recalls_pert else float("nan"),
            "recall_perturbed_min": float(min(recalls_pert)) if recalls_pert else float("nan"),
            "n_pos_test_mean": float(np.mean([r["n_pos_test"] for r in split_rows])),
            "n_in_cluster_mean": float(np.mean([r["n_in_cluster"] for r in split_rows])),
            "baseline_candidates_mean": base_mean,
            "filtered_candidates_mean": filt_mean,
            "filtered_candidates_p95": filt_p95,
            "filtered_candidates_max": filt_max,
            "candidate_reduction_mean": float(1.0 - filt_mean / max(1e-9, base_mean)),
            "tiles_baseline": int(math.ceil(max_size / tile_width)),
            "tiles_filtered_safe": int(math.ceil(filt_max / tile_width)),
            "retained_negatives_by_rho": retained_neg,
            "retained_positives_by_rho": retained_pos,
            "splits": len(split_rows),
        })

    # ---------------- 6. 输出 ----------------
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "cluster_filter.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ell", "d", "dev_recall", "dev_mean_candidates"])
        for row in curve_rows:
            writer.writerow([row["ell"], row["d"], f"{row['dev_recall']:.4f}", f"{row['dev_mean_candidates']:.2f}"])

    with (out_dir / "per_query.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        fields = ["ell", "query_ncid", "perturbed", "d_star", "baseline_candidates",
                  "filtered_candidates", "target_survived", "target_hamming"]
        writer.writerow(fields)
        for row in per_query_rows:
            writer.writerow([row.get(key, "") for key in fields])

    payload = {
        "config": vars(args) | {"data_root": str(args.data_root), "out_dir": str(args.out_dir)},
        "runtime": {"normalizer": _NORMALIZER_SOURCE, "numpy": np.__version__, "kmeans": "numpy 复刻"},
        "clusters": {
            "k": args.clusters,
            "sizes": {"mean": float(cluster_sizes.mean()), "max": max_size, "min": int(cluster_sizes.min())},
            "tile_width": tile_width,
        },
        "cluster_recall": cluster_recall,
        "target_in_cluster": int(target_in_cluster[positives].sum()),
        "positives": int(len(positives)),
        "points": results,
    }
    (out_dir / "cluster_filter.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# 簇内候选过滤评估",
        "",
        f"- 库 {len(db_names)} 条，k={args.clusters}，簇大小 mean {cluster_sizes.mean():.1f} / max {max_size}",
        f"- Top-1 簇召回（真目标落在选中簇）：{cluster_recall:.3f}"
        f"（{int(target_in_cluster[positives].sum())}/{len(positives)}）——这是**上游**损失，与过滤器无关",
        f"- {args.splits} 次分层 dev/test 切分；阈值 d 只在 dev 上按 \"召回 ≥ {args.target_recall}\" 选取，"
        f"所有指标在 test 上汇总",
        f"- test 集正例约 {np.mean([p['n_pos_test_mean'] for p in results]):.0f} 条，"
        f"其中落在选中簇的约 {np.mean([p['n_in_cluster_mean'] for p in results]):.0f} 条"
        f"（阈值由这 {np.mean([p['n_in_cluster_mean'] for p in results]):.0f} 条里最难的扰动正例决定）",
        "",
        "| ell | d* (范围) | 条件召回 mean [min] | **端到端召回** mean [min] | 扰动正例端到端 | 候选 mean | 候选 max | 缩减 | tiles(基线/safe) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for point in results:
        lines.append(
            f"| {point['ell']} | {point['d_star_median']} "
            f"({point['d_star_range'][0]}–{point['d_star_range'][1]}) | "
            f"{point['recall_cond_mean']:.3f} [{point['recall_cond_min']:.3f}] | "
            f"{point['recall_e2e_mean']:.3f} [{point['recall_e2e_min']:.3f}] | "
            f"{point['recall_perturbed_mean']:.3f} | "
            f"{point['baseline_candidates_mean']:.1f} → {point['filtered_candidates_mean']:.1f} | "
            f"{point['filtered_candidates_max']:.0f} | "
            f"{point['candidate_reduction_mean']*100:.1f}% | "
            f"{point['tiles_baseline']} → {point['tiles_filtered_safe']} |"
        )
    for title, key in (
        ("簇内负例保留率（实体身份不相交）", "retained_negatives_by_rho"),
        ("簇内正例（真目标）保留率", "retained_positives_by_rho"),
    ):
        present = sorted({k for point in results for k in point[key]},
                         key=lambda x: float(x.split("-")[0]))
        if not present:
            continue
        lines += ["", f"## {title}", ""]
        lines.append("| ell | " + " | ".join(present) + " |")
        lines.append("|---|" + "---|" * len(present))
        for point in results:
            cells = [
                f"{point[key][label]:.3f}" if label in point[key] else "n/a" for label in present
            ]
            lines.append(f"| {point['ell']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 边界（必须连同数字一起引用）",
        "",
        "- **过滤器自身的成本没有计入。** 上表的 tiles 只算 CKKS 那一半；二值码的生成、"
        "Hamming 比较要么在密文上做（等于把省下的计算又花回去），要么经 PSI（另计通信与泄漏）。",
        f"- **阈值是极值统计量。** 在约 {np.mean([p['n_in_cluster_mean'] for p in results]):.0f} 条"
        f"\"真目标在簇内\"的正例上要求 {args.target_recall:.0%} 覆盖，等价于要求全部覆盖；"
        f"d 由其中最难的扰动正例决定，margin 很薄。{args.splits} 次切分只反映切分波动，"
        "不反映换数据集或换扰动种子的波动。",
        "- **平均候选数下降 ≠ 固定形状工作量下降。** CKKS 必须按最大保留宽度补齐，"
        "所以要看\"候选 max\"与 tiles 两列；mean 的降幅不能直接换算成加速。",
        "- **端到端召回 = 簇召回 × 条件过滤召回**，但两阶段正例集合不同，必须分别报告；",
        "  表中\"端到端\"列的分母是全部 test 正例，可直接与簇召回相乘核对。",
        "- 标准化器只在库侧拟合（与生产 `fit_scaler` 一致），查询复用同一组 mean/scale。",
        "- tiles(safe) 用过滤后候选数的**最大值**（固定形状要求）；按查询变长会泄漏，不能直接用 mean。",
        "- 聚类是 numpy 复刻（无 sklearn），与实际 `run_cosine_kmeans` 可能有 ±若干百分点的差异。",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\noutputs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
