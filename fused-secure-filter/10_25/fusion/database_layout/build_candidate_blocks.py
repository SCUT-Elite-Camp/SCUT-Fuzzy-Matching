"""把库切成**公开**的固定容量块。

块是公开边界：块的编号、容量、以及块内记录的存储顺序都不保密。
保密的是"查询触达了哪些块"——这一点由 `candidate_selection` 负责记账。

三种块内排序策略，用于检验"幸存候选是否在块内聚集"：
    stored     库的原始顺序
    cluster    按簇分组，簇内保持原序
    code       按簇分组，簇内再按二值码排序（相似码相邻）
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BlockLayout:
    order: np.ndarray        # (n,) 块内记录顺序：position -> db index
    block_of: np.ndarray     # (n,) db index -> block id
    block_size: int
    n_blocks: int
    strategy: str


def build_blocks(
    *,
    n: int,
    block_size: int,
    strategy: str,
    assignments: np.ndarray | None = None,
    codes: np.ndarray | None = None,
) -> BlockLayout:
    if strategy == "stored":
        order = np.arange(n)
    elif strategy in {"cluster", "code"}:
        if assignments is None:
            raise ValueError("cluster/code strategy needs assignments")
        # 先按簇，再按簇内策略
        cluster_num = np.asarray(assignments, dtype=np.int64)
        if strategy == "code":
            if codes is None:
                raise ValueError("code strategy needs codes")
            # codes 打包成整数用于排序：(n, ell) bool -> 大整数
            weights = 1 << np.arange(codes.shape[1], dtype=object)
            key = (codes.astype(object) * weights).sum(axis=1).astype(object)
            order = np.lexsort((np.array([int(v) for v in key], dtype=object), cluster_num))
            order = np.asarray(order, dtype=np.int64)
        else:
            order = np.argsort(cluster_num, kind="stable")
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    block_of = np.empty(n, dtype=np.int64)
    block_of[order] = np.arange(n) // block_size
    return BlockLayout(order=order, block_of=block_of, block_size=block_size,
                       n_blocks=int(np.ceil(n / block_size)), strategy=strategy)
