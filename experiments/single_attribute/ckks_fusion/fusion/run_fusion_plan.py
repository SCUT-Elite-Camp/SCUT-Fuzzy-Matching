#!/usr/bin/env python3
"""驱动：候选筛选 → 执行计划 → 两条路径的成本对照。

    python fusion/run_fusion_plan.py --ell 512 --d 109 --block-sizes 20 32 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.params import NUM_PERMUTATIONS_CLUSTER, NUM_PERMUTATIONS_MATCH  # noqa: E402
from minhash.encoder import batch_encode  # noqa: E402
from scripts.evaluate_binarization_frontier import l2_normalize, read_ncvr  # noqa: E402
from scripts.evaluate_cluster_conditional_filter import cosine_kmeans, select_clusters  # noqa: E402

from fusion import database_layout  # noqa: E402
from fusion.candidate_selection import build_execution_plan, filter_candidates  # noqa: E402
from fusion.evaluation import audit  # noqa: E402
from fusion.evaluation.compare_actual_execution import compare  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--clusters", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--ell", type=int, default=512)
    ap.add_argument("--d", type=int, default=109)
    ap.add_argument("--block-sizes", type=int, nargs="+", default=[20, 32, 64])
    ap.add_argument("--interval", type=int, default=64, help="code 策略中按码值分段排序的粒度")
    args = ap.parse_args()

    db_names, db_ids, entries = read_ncvr(
        args.data_root, query_limit=200, db_limit=10000, fuzzy_ratio=0.3, fuzzy_seed=42
    )
    db200 = batch_encode(db_names, NUM_PERMUTATIONS_CLUSTER)
    q200 = batch_encode([e.name for e in entries], NUM_PERMUTATIONS_CLUSTER)
    db50 = l2_normalize(db200[:, :NUM_PERMUTATIONS_MATCH])
    q50 = l2_normalize(q200[:, :NUM_PERMUTATIONS_MATCH])

    mean = l2_normalize(db200).mean(axis=0)
    scale = l2_normalize(db200).std(axis=0, ddof=0)
    scale = np.where(scale < 1e-10, 1.0, scale)
    db_std = (l2_normalize(db200) - mean) / scale
    q_std = (l2_normalize(q200) - mean) / scale

    idx = database_layout.build_cluster_index(db_std, args.clusters)
    selected = select_clusters(q_std, idx.centroids)

    planes = np.random.default_rng(2026).standard_normal((args.ell, NUM_PERMUTATIONS_MATCH))
    db_codes = (db50 @ planes.T) >= 0
    q_codes = (q50 @ planes.T) >= 0

    survivors = filter_candidates(
        assignments=idx.assignments, query_codes=q_codes, db_codes=db_codes,
        selected_clusters=selected, ell=args.ell, d=args.d,
    )

    labels = np.array([e.label for e in entries], dtype=bool)
    targets = np.array([e.target_index if e.target_index is not None else -1 for e in entries])
    target_in_cluster = np.array([
        bool(labels[i] and targets[i] >= 0 and idx.assignments[targets[i]] == selected[i])
        for i in range(len(entries))
    ])

    print(f"库 {len(db_names)}，k={args.clusters}，max_size={idx.max_size}，"
          f"ell={args.ell}，d={args.d}")
    print(f"逻辑候选: mean {np.mean([s.size for s in survivors]):.1f} / "
          f"max {np.max([s.size for s in survivors])}")
    header = f"{'layout':<9}{'B':>4}{'块总数':>7}{'每查询块(max/mean)':>18}{'铺满槽位':>10}{'tiles':>8}{'vs基线':>8}{'ct-pt':>10}{'召回':>7}"
    print("\n" + header)
    print("-" * len(header))

    rows = []
    for strategy in ("stored", "cluster", "code"):
        for B in args.block_sizes:
            layout = database_layout.build_blocks(
                n=len(db_names), block_size=B, strategy=strategy,
                assignments=idx.assignments, codes=db_codes,
            )
            plan = build_execution_plan(
                survivors=survivors, layout=layout, max_size=idx.max_size,
                leakage_mode="public_index",
            )
            row = compare(
                plan=plan, survivors=survivors, targets=targets,
                target_in_cluster=target_in_cluster, batch_size=args.batch_size,
                clusters=args.clusters, max_size=idx.max_size, dim=NUM_PERMUTATIONS_MATCH,
            )
            row["strategy"] = strategy
            row["block_size"] = B
            rows.append(row)
            print(f"{strategy:<9}{B:>4}{layout.n_blocks:>7}"
                  f"{plan.blocks_executed:>9}/{plan.blocks_per_query.mean():<8.1f}"
                  f"{plan.processed_positions:>10}{row['blockwise_tiles']:>8}"
                  f"{row['tile_ratio']:>7.2f}x{row['blockwise_ct_pt']:>10}{row['target_recall']:>7.3f}")

    print("\n泄漏模式对照（同一日志）：")
    for mode in ("public_index", "fixed_count", "secret_index"):
        info = audit(mode)
        print(f"  {mode:<13} B 可见: {info.get('B_sees', '?')[:38]:<38} 省计算: {info.get('saves','?')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
