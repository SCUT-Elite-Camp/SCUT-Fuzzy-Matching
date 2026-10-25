"""候选发现：在选中簇内部，用二值码 Hamming <= d 过滤出幸存候选。

不做任何密码学，输出的是**逻辑候选**（db 索引）。它是否泄漏、以及
能否转化为更少的实际计算，由 `build_execution_plan` 与 `leakage_audit` 负责。
"""
from __future__ import annotations

import numpy as np


def filter_candidates(
    *,
    assignments: np.ndarray,
    query_codes: np.ndarray,
    db_codes: np.ndarray,
    selected_clusters: np.ndarray,
    ell: int,
    d: int,
) -> list[np.ndarray]:
    """返回每条查询在其选中簇内的幸存候选（db 索引，按 db 顺序）。"""
    survivors: list[np.ndarray] = []
    for q in range(query_codes.shape[0]):
        members = np.flatnonzero(assignments == selected_clusters[q])
        if members.size == 0:
            survivors.append(members)
            continue
        dist = (query_codes[q, :ell][None, :] != db_codes[members, :ell]).sum(axis=1)
        survivors.append(members[dist <= d])
    return survivors
