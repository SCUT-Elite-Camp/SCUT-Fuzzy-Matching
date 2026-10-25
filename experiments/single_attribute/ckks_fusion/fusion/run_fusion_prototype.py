#!/usr/bin/env python3
"""最小可运行融合原型：A / B / C 可运行，D 明确阻断并给出理由。

    python fusion/run_fusion_prototype.py --ell 512 --d 109 --block-size 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.params import NUM_PERMUTATIONS_CLUSTER, NUM_PERMUTATIONS_MATCH  # noqa: E402
from minhash.encoder import batch_encode  # noqa: E402
from scripts.evaluate_binarization_frontier import l2_normalize, read_ncvr  # noqa: E402
from scripts.evaluate_cluster_conditional_filter import select_clusters  # noqa: E402

from fusion import database_layout  # noqa: E402
from fusion.adapter import AdaptedBatch, CandidateEnvelope, FusionBlocked, adapt  # noqa: E402
from fusion.evaluation import leakage_audit  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--clusters", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--ell", type=int, default=512)
    ap.add_argument("--d", type=int, default=109)
    ap.add_argument("--block-size", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "fusion_prototype")
    args = ap.parse_args()

    db_names, _db_ids, entries = read_ncvr(
        args.data_root, query_limit=200, db_limit=10000, fuzzy_ratio=0.3, fuzzy_seed=42
    )
    db200 = batch_encode(db_names, NUM_PERMUTATIONS_CLUSTER)
    q200 = batch_encode([e.name for e in entries], NUM_PERMUTATIONS_CLUSTER)
    db50 = l2_normalize(db200[:, :NUM_PERMUTATIONS_MATCH])
    q50 = l2_normalize(q200[:, :NUM_PERMUTATIONS_MATCH])

    mu = l2_normalize(db200).mean(axis=0)
    sd = l2_normalize(db200).std(axis=0, ddof=0)
    sd = np.where(sd < 1e-10, 1.0, sd)
    db_std = (l2_normalize(db200) - mu) / sd
    q_std = (l2_normalize(q200) - mu) / sd

    idx = database_layout.build_cluster_index(db_std, args.clusters)
    selected = select_clusters(q_std, idx.centroids)

    planes = np.random.default_rng(2026).standard_normal((args.ell, NUM_PERMUTATIONS_MATCH))
    db_codes = (db50 @ planes.T) >= 0
    q_codes = (q50 @ planes.T) >= 0

    survivors = []
    for qi in range(len(entries)):
        members = np.flatnonzero(idx.assignments == selected[qi])
        dist = (q_codes[qi, : args.ell][None, :] != db_codes[members, : args.ell]).sum(axis=1)
        survivors.append(members[dist <= args.d])

    # 正确的布局：按簇聚集（否则分块比基线更贵，见上一轮实测）
    layout = database_layout.build_blocks(
        n=len(db_names), block_size=args.block_size, strategy="cluster",
        assignments=idx.assignments, codes=db_codes,
    )

    common = dict(layout=layout, max_size=idx.max_size, clusters=args.clusters,
                  dim=NUM_PERMUTATIONS_MATCH, batch_size=args.batch_size)

    results: list[dict] = []

    # A 基线
    base = adapt(CandidateEnvelope("none", "fixed_shape"), **common)
    # B 公开索引
    env_b = CandidateEnvelope("binarization_filter", "plaintext_index", per_query=survivors)
    b = adapt(env_b, **common)
    # C 固定形状（候选不可见）
    c = adapt(CandidateEnvelope("binarization_filter", "fixed_shape"), **common)
    # D PSI 融合
    d_error = None
    try:
        adapt(CandidateEnvelope("approx_psi", "secret_share", share_bits=1), **common)
    except FusionBlocked as exc:
        d_error = str(exc)

    def row(tag: str, batch: AdaptedBatch, note: str = "") -> dict:
        return {
            "path": tag, "mode": batch.mode, "blocks": batch.blocks,
            "positions": batch.processed_positions, "tiles": batch.tiles,
            "ct_pt_mul": batch.ct_pt_mul, "ct_ct_mul": batch.ct_ct_mul,
            "tiles_vs_baseline": batch.tiles / base.tiles if base.tiles else None,
            "leaks_index": batch.leaks_candidate_index, "note": note or batch.note,
        }

    results.append(row("A_baseline", base))
    results.append(row("B_public_index", b))
    results.append(row("C_fixed_shape", c))
    results.append({"path": "D_psi_fusion", "mode": "blocked", "blocks": None,
                    "positions": None, "tiles": None, "ct_pt_mul": None,
                    "ct_ct_mul": None, "tiles_vs_baseline": None,
                    "leaks_index": None, "note": d_error})

    # 逻辑候选 / 召回（算法层面，实测）
    targets = np.array([e.target_index if e.target_index is not None else -1 for e in entries])
    in_cluster = np.array([bool(targets[i] >= 0 and idx.assignments[targets[i]] == selected[i])
                           for i in range(len(entries))])
    n_positive = int(sum(1 for e in entries if e.label))
    considered = int(in_cluster.sum())
    survived = sum(int(np.isin(targets[i], survivors[i])) for i in range(len(entries)) if in_cluster[i])
    logical = np.array([s.size for s in survivors])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(
        json.dumps({"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "rows": results,
                    "logical_candidates": {"mean": float(logical.mean()), "max": int(logical.max())},
                    "recall": {"target_in_cluster": considered / n_positive if n_positive else None,
                               "filter_conditional": survived / considered if considered else None}},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    leakage_audit.write_csv(args.out_dir / "leakage.csv")

    print(f"逻辑候选 mean {logical.mean():.1f} / max {logical.max()}")
    print(f"簇召回 {considered/n_positive:.3f}（{considered}/{n_positive} 条正例），"
          f"簇内过滤召回 {survived/considered:.3f}\n")
    print(f"{'路径':<16}{'blocks':>7}{'positions':>10}{'tiles':>7}{'vs基线':>8}{'泄漏索引':>9}")
    print("-" * 60)
    for row_ in results:
        ratio = "-" if row_["tiles_vs_baseline"] is None else f"{row_['tiles_vs_baseline']:.2f}x"
        cells = [row_["path"], str(row_["blocks"]), str(row_["positions"]),
                 str(row_["tiles"]), ratio, str(row_["leaks_index"])]
        print(f"{cells[0]:<16}{cells[1]:>7}{cells[2]:>10}{cells[3]:>7}{cells[4]:>8}{cells[5]:>9}")
    print(f"\nD 阻断原因：{d_error}")
    print(f"\n输出 -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
