"""把三种数量与两条路径的成本放在一张表里比较。"""
from __future__ import annotations

import numpy as np

from fusion.ckks_matching import baseline_cost, blockwise_cost
from fusion.candidate_selection import ExecutionPlan


def compare(
    *,
    plan: ExecutionPlan,
    survivors: list[np.ndarray],
    targets: np.ndarray,
    target_in_cluster: np.ndarray,
    batch_size: int,
    clusters: int,
    max_size: int,
    dim: int,
) -> dict:
    base = baseline_cost(batch_size=batch_size, clusters=clusters, max_size=max_size, dim=dim)
    block = blockwise_cost(batch_size=batch_size, block_size=plan.block_size,
                           blocks=plan.blocks_executed, dim=dim)

    # 召回：目标是否在幸存集合里（且它本来就在选中簇内）
    recalled = 0
    considered = 0
    for q, survived in enumerate(survivors):
        if targets[q] < 0 or not target_in_cluster[q]:
            continue
        considered += 1
        recalled += int(np.isin(targets[q], survived))

    logical = plan.logical_candidates.astype(float)
    return {
        "leakage_mode": plan.leakage_mode,
        "logical_candidates_mean": float(logical.mean()),
        "logical_candidates_max": float(logical.max()),
        "blocks_per_query_max": int(plan.blocks_executed),
        "blocks_per_query_mean": float(plan.blocks_per_query.mean()),
        "processed_positions": int(plan.processed_positions),
        "baseline_tiles": base.tiles,
        "blockwise_tiles": block.tiles,
        "tile_ratio": block.tiles / base.tiles if base.tiles else float("nan"),
        "baseline_ct_pt": base.ct_pt_mul,
        "blockwise_ct_pt": block.ct_pt_mul,
        "ct_pt_ratio": block.ct_pt_mul / base.ct_pt_mul if base.ct_pt_mul else float("nan"),
        "target_recall": recalled / considered if considered else float("nan"),
        "targets_considered": considered,
    }
