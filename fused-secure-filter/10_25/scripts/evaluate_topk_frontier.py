#!/usr/bin/env python3
"""Top-K 探测曲线：召回 vs 候选成本，在同一套数据/聚类/判据下对照。

上一轮已确认：端到端召回 0.86，最大损失在**目标进入候选集之前**（Top-1 簇召回
0.880，扰动正例更低）。本脚本测"多探几个簇"能把召回推到哪，以及代价是多少。

## 四个对照（同一批查询、同一库侧 scaler、同一最终判据）

  top1_no_filter   基线：Top-1，无过滤
  topk_no_filter   Top-K，无过滤；K = 1,2,3,5,8 与穷举
  topk_filter      Top-K + 二值码过滤（阈值在 dev 上按条件召回选）
  (mixed)          未实现：GC/PSI 成本未计入

## 两个必须分开的成本口径

  tiles_fixed     固定形状：每个探测簇都要按 max_size 补齐，K 个探测 = K × ceil(max_size/T)
  columns_early   部署形状：按质心分降序逐簇扫，撞到命中就停；这是 `multi_attribute`
                  实测使用的口径（B 算完被探到的整簇）。**无命中的查询救不了**，
                  它会扫满 K 个簇（穷举时扫满全库），必须单独报告。

## 用法

    python scripts/evaluate_topk_frontier.py --ks 1 2 3 5 8 0 --ells 256 512

`--ks 0` 表示穷举探测。
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
    NUM_PERMUTATIONS_CLUSTER,
    NUM_PERMUTATIONS_MATCH,
)
from minhash.encoder import batch_encode  # noqa: E402

from scripts.evaluate_binarization_frontier import (  # noqa: E402
    _NORMALIZER_SOURCE,
    l2_normalize,
    read_ncvr,
)
from scripts.evaluate_cluster_conditional_filter import (  # noqa: E402
    cosine_kmeans,
    select_clusters,
)

SLOT_CAPACITY = 4096


def q(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
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
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5, 8, 0],
                        help="探测簇数；0 表示穷举")
    parser.add_argument("--ells", type=int, nargs="+", default=[256, 512],
                        help="Top-K + 过滤 对照使用的码长")
    parser.add_argument("--filter-ks", type=int, nargs="+", default=[1, 3],
                        help="在这些 K 上叠加二值码过滤")
    parser.add_argument("--target-recall", type=float, default=0.98)
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--plane-seed", type=int, default=2026)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts" / "topk_frontier")
    args = parser.parse_args(argv)

    tile_width = SLOT_CAPACITY // args.batch_size

    # ---------------- 数据与编码（与上一轮完全一致） ----------------
    db_names, db_ids, entries = read_ncvr(
        args.data_root, query_limit=args.query_limit, db_limit=args.db_limit,
        fuzzy_ratio=args.fuzzy_ratio, fuzzy_seed=args.fuzzy_seed,
    )
    db_sig200 = batch_encode(db_names, NUM_PERMUTATIONS_CLUSTER)
    q_sig200 = batch_encode([e.name for e in entries], NUM_PERMUTATIONS_CLUSTER)
    db_vec = l2_normalize(db_sig200[:, :NUM_PERMUTATIONS_MATCH])
    q_vec = l2_normalize(q_sig200[:, :NUM_PERMUTATIONS_MATCH])

    # 库侧拟合一次，查询复用（与生产 fit_scaler 一致）
    db_norm200 = l2_normalize(db_sig200)
    q_norm200 = l2_normalize(q_sig200)
    mean = db_norm200.mean(axis=0)
    scale = db_norm200.std(axis=0, ddof=0)
    scale = np.where(scale < 1e-10, 1.0, scale)
    db_std = (db_norm200 - mean) / scale
    q_std = (q_norm200 - mean) / scale

    centroids, assignments = cosine_kmeans(db_std, args.clusters)
    cluster_sizes = np.bincount(assignments, minlength=args.clusters)
    max_size = int(cluster_sizes.max())
    n_db = len(db_names)

    # 查询对所有簇的得分排序（Top-1 只是 K=1 的特例）
    scores = l2_normalize(q_std) @ l2_normalize(centroids).T
    order = np.argsort(-scores, axis=1)  # (m, k) 降序

    labels = np.array([e.label for e in entries], dtype=bool)
    target_idx = np.array(
        [e.target_index if e.target_index is not None else -1 for e in entries], dtype=np.int64
    )
    target_cluster = np.where(labels, assignments[np.maximum(target_idx, 0)], -1)

    id_codes: dict[str, int] = {}
    for value in db_ids + [e.ncid for e in entries]:
        id_codes.setdefault(value, len(id_codes))
    db_codes = np.array([id_codes[v] for v in db_ids], dtype=np.int64)
    q_codes = np.array([id_codes[e.ncid] for e in entries], dtype=np.int64)

    positives = np.flatnonzero(labels)
    n_pos = int(len(positives))

    # ---------------- 逐 K 统计 ----------------
    rows = []
    for K in args.ks:
        k_eff = args.clusters if K <= 0 else min(K, args.clusters)
        exhaustive = K <= 0

        probe_hit = np.zeros(len(entries), dtype=bool)
        union_sizes = np.zeros(len(entries), dtype=np.int64)
        early_cols = np.zeros(len(entries), dtype=np.int64)
        rank_of_target = np.zeros(len(entries), dtype=np.int64)

        for i in range(len(entries)):
            probed = order[i, :k_eff]
            members_mask = np.isin(assignments, probed)
            union_sizes[i] = int(members_mask.sum())

            if labels[i] and target_cluster[i] >= 0:
                hit_pos = int(np.flatnonzero(probed == target_cluster[i])[0]) if np.any(probed == target_cluster[i]) else -1
                if hit_pos >= 0:
                    probe_hit[i] = True
                    rank_of_target[i] = hit_pos
                    # 早停：扫完前 hit_pos+1 个簇；最后一个簇只扫到目标所在列
                    before = int(cluster_sizes[probed[:hit_pos]].sum())
                    within = int((assignments == target_cluster[i])[: target_idx[i]].sum()) + 1
                    early_cols[i] = before + within
                else:
                    early_cols[i] = int(cluster_sizes[probed].sum())
            else:
                # 无命中查询：早停无从触发
                early_cols[i] = n_db if exhaustive else int(cluster_sizes[probed].sum())

        probe_recall = float(probe_hit[positives].mean())
        rows.append({
            "k": args.clusters if exhaustive else K,
            "exhaustive": exhaustive,
            "probe_recall": probe_recall,
            "e2e_recall": probe_recall,  # 无过滤时二者相同
            "union_candidates": q(union_sizes.astype(float)),
            # 规则 A：K 次独立 selector 通过，每次按全局 max_size 补齐（当前 kernel 的形态）
            "tiles_rule_a": int(k_eff * math.ceil(max_size / tile_width)),
            # 规则 B：把 K 个探测簇的并集当成一个候选表，按并集最大宽度补齐
            "tiles_rule_b": int(math.ceil(union_sizes.max() / tile_width)),
            "columns_early_stop": q(early_cols.astype(float)),
            "no_hit_full_scan_queries": int((~probe_hit[positives]).sum()) + int((~labels).sum()),
        })

    # ---------------- Top-K + 二值码过滤 ----------------
    ells = sorted({int(v) for v in args.ells if v > 0})
    planes = np.random.default_rng(args.plane_seed).standard_normal((max(ells), NUM_PERMUTATIONS_MATCH))
    db_bits = (db_vec @ planes.T) >= 0.0
    q_bits = (q_vec @ planes.T) >= 0.0

    def clustered_split(seed: int):
        rng = np.random.default_rng(seed)
        dev_l, test_l = [], []
        for value in (True, False):
            pool = np.flatnonzero(labels == value)
            shuf = rng.permutation(pool)
            half = len(shuf) // 2
            dev_l.extend(shuf[:half].tolist())
            test_l.extend(shuf[half:].tolist())
        return np.array(sorted(dev_l)), np.array(sorted(test_l))

    filter_rows = []
    for K in args.filter_ks:
        k_eff = args.clusters if K <= 0 else min(K, args.clusters)
        for ell in ells:
            d_choices, e2e_list, cand_mean, cand_max = [], [], [], []
            for split_index in range(args.splits):
                dev_idx, test_idx = clustered_split(args.split_seed + 1000 * split_index)

                def dists(i):
                    members = np.flatnonzero(np.isin(assignments, order[i, :k_eff]))
                    if members.size == 0:
                        return members, np.zeros(0, dtype=np.int32)
                    d = (q_bits[i, :ell][None, :] != db_bits[members, :ell]).sum(axis=1)
                    return members, d.astype(np.int32)

                # dev 上定阈值
                dev_pos = [i for i in dev_idx if labels[i] and target_cluster[i] >= 0
                           and np.any(order[i, :k_eff] == target_cluster[i])]
                dev_dists = {i: dists(i) for i in dev_idx}
                dev_tdist = {}
                for i in dev_pos:
                    members, d = dev_dists[i]
                    pos = np.searchsorted(members, target_idx[i])
                    if pos < members.size and members[pos] == target_idx[i]:
                        dev_tdist[i] = int(d[pos])
                d_star = None
                for cand in range(0, min(ell, 200) + 1):
                    recall_dev = (float(np.mean([v <= cand for v in dev_tdist.values()]))
                                  if dev_tdist else float("nan"))
                    if recall_dev >= args.target_recall:
                        d_star = cand
                        break
                if d_star is None:
                    d_star = min(ell, 200)
                d_choices.append(d_star)

                kept, survived, n_in = [], 0, 0
                per_probe_max = 0
                for i in test_idx:
                    members, d = dists(i)
                    kept.append(int((d <= d_star).sum()))
                    # per-probe 口径：每个探测簇单独补齐时的最大保留数
                    for probe in order[i, :k_eff]:
                        mem = np.flatnonzero(assignments == probe)
                        if mem.size == 0:
                            continue
                        dd = (q_bits[i, :ell][None, :] != db_bits[mem, :ell]).sum(axis=1)
                        per_probe_max = max(per_probe_max, int((dd <= d_star).sum()))
                    if labels[i]:
                        ok = False
                        if target_cluster[i] >= 0 and np.any(order[i, :k_eff] == target_cluster[i]):
                            n_in += 1
                            pos = np.searchsorted(members, target_idx[i])
                            ok = bool(pos < members.size and members[pos] == target_idx[i] and d[pos] <= d_star)
                        survived += int(ok)
                n_test_pos = int(sum(1 for i in test_idx if labels[i]))
                e2e_list.append(survived / max(1, n_test_pos))
                kept_arr = np.array(kept, dtype=float)
                cand_mean.append(float(kept_arr.mean()))
                cand_max.append(float(kept_arr.max()))

            filter_rows.append({
                "k": args.clusters if K <= 0 else K,
                "ell": ell,
                "d_star_median": int(np.median(d_choices)),
                "d_star_range": [int(min(d_choices)), int(max(d_choices))],
                "e2e_recall_mean": float(np.mean(e2e_list)),
                "e2e_recall_min": float(min(e2e_list)),
                "candidates_mean": float(np.mean(cand_mean)),
                "candidates_max": float(max(cand_max)),
                "per_probe_max": int(per_probe_max),
                "tiles_rule_a": int(k_eff * math.ceil(per_probe_max / tile_width)),
                "tiles_rule_b": int(math.ceil(max(cand_max) / tile_width)),
            })

    # ---------------- 输出 ----------------
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "topk.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["k", "exhaustive", "probe_recall", "cand_mean", "cand_p95", "cand_max",
                    "tiles_rule_a", "tiles_rule_b", "early_cols_mean", "early_cols_p95", "early_cols_max"])
        for row in rows:
            w.writerow([row["k"], row["exhaustive"], f"{row['probe_recall']:.4f}",
                        f"{row['union_candidates']['mean']:.1f}", f"{row['union_candidates']['p95']:.1f}",
                        f"{row['union_candidates']['max']:.0f}", row["tiles_rule_a"], row["tiles_rule_b"],
                        f"{row['columns_early_stop']['mean']:.1f}",
                        f"{row['columns_early_stop']['p95']:.1f}",
                        f"{row['columns_early_stop']['max']:.0f}"])

    payload = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "runtime": {"normalizer": _NORMALIZER_SOURCE, "numpy": np.__version__, "kmeans": "numpy 复刻"},
        "clusters": {"k": args.clusters, "max_size": max_size, "tile_width": tile_width,
                     "mean_size": float(cluster_sizes.mean())},
        "positives": n_pos,
        "topk": rows,
        "topk_filter": filter_rows,
    }
    (out_dir / "topk.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Top-K 探测曲线",
        "",
        f"- 库 {n_db} 条，k={args.clusters}，簇大小 mean {cluster_sizes.mean():.1f} / max {max_size}",
        f"- 正例 {n_pos} 条；库侧拟合标准化器，查询复用",
        f"- 固定形状 tile：每个探测簇按 max_size 补齐，`tiles_fixed = K × ceil({max_size}/{tile_width})`",
        "",
        "## 无过滤",
        "",
        "| K | 探测召回（=端到端） | 候选 mean | 候选 max | tiles 规则A（每簇补齐） | tiles 规则B（并集补齐） | 早停列数 mean |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        k_label = "穷举" if row["exhaustive"] else str(row["k"])
        lines.append(
            f"| {k_label} | {row['probe_recall']:.3f} | {row['union_candidates']['mean']:.1f} | "
            f"{row['union_candidates']['max']:.0f} | "
            f"{row['tiles_rule_a']} | {row['tiles_rule_b']} | "
            f"{row['columns_early_stop']['mean']:.1f} |"
        )
    lines += [
        "",
        "## Top-K + 二值码过滤",
        "",
        "| K | ell | d* | 端到端召回 mean [min] | 候选 mean | 候选 max | 每簇最大保留 | tiles 规则A | tiles 规则B |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in filter_rows:
        k_label = "穷举" if row["k"] == args.clusters else str(row["k"])
        lines.append(
            f"| {k_label} | {row['ell']} | {row['d_star_median']} | "
            f"{row['e2e_recall_mean']:.3f} [{row['e2e_recall_min']:.3f}] | "
            f"{row['candidates_mean']:.1f} | {row['candidates_max']:.0f} | {row['per_probe_max']} | "
            f"{row['tiles_rule_a']} | {row['tiles_rule_b']} |"
        )
    lines += [
        "",
        "## 口径",
        "",
        "- `probe_recall` 的分母是全部正例（含目标不在任何探测簇的），因此它**就是**无过滤时"
        "的端到端召回，可直接与簇召回对照。",
        "- `tiles_fixed` 是固定形状下的 CKKS 工作量；早停列数只用于部署口径，"
        "**无命中的查询无法早停**，会扫满 K 个簇（穷举时扫满全库）。",
        "- 过滤行里的候选数只统计被探测簇的并集，不含 padding；tiles 仍按固定形状的最大宽度算。",
        "- **规则 A**：K 次独立 selector 通过，每次按全局 `max_size` 补齐 —— 这是当前 "
        "`column_wise_multi_probe_matching` 的形态，tiles = K × ceil(max_size/T)。",
        "- **规则 B**：把 K 个探测簇的并集当作一张候选表，按并集最大宽度补齐 —— 需要 oblivious "
        "compaction，但宽度只需一个常量。两种规则的 tile 数**不可混比**，必须成对照报。",
        "- 本脚本不含任何 GC / PSI 成本，也不含 CKKS 实测时间；墙钟时间需要 TenSEAL 环境。",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\noutputs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
