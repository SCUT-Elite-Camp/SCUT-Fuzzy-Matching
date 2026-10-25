"""把逻辑候选翻译成**块索引**，并对三种数量分别记账。

三种数量（缺一不可，逻辑候选减少不等于实际计算减少）：

    logical_candidates   粗筛后还剩多少候选
    blocks_executed      CKKS 真正需要处理多少个块（固定形状时取所有查询的最大值）
    processed_positions  块数 × 块容量——这才是 CKKS 实际铺满的槽位量

## 泄漏模式

    public_index   每条查询的块索引明文交给 B —— 省计算，但泄漏访问模式
    fixed_count    块数补齐到全局上限，只泄漏"上限"不泄漏真实个数
    secret_index   索引保密 —— 现有 CKKS 栈无法做到，仅作对照（等价于全扫）

`secret_index` 之所以退化为全扫：B 不知道要算哪些块，只能全部算。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fusion.database_layout.build_candidate_blocks import BlockLayout


@dataclass
class ExecutionPlan:
    leakage_mode: str
    block_size: int
    blocks_per_query: np.ndarray        # (m,) 每条查询实际执行的块数
    blocks_executed: int                # 固定形状下真正执行的块数（= 最大值）
    processed_positions: int            # blocks_executed * block_size
    logical_candidates: np.ndarray      # (m,) 逻辑候选数
    block_sets: list[np.ndarray] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def build_execution_plan(
    *,
    survivors: list[np.ndarray],
    layout: BlockLayout,
    max_size: int,
    leakage_mode: str = "public_index",
) -> ExecutionPlan:
    m = len(survivors)
    logical = np.array([s.size for s in survivors], dtype=np.int64)
    block_sets: list[np.ndarray] = []
    per_query_blocks = np.zeros(m, dtype=np.int64)

    for q, s in enumerate(survivors):
        if s.size:
            blocks = np.unique(layout.block_of[s])
        else:
            blocks = np.zeros(0, dtype=np.int64)
        block_sets.append(blocks)
        per_query_blocks[q] = blocks.size

    notes: list[str] = []
    if leakage_mode == "secret_index":
        # B 不知道块索引 → 只能扫全部块（此处近似为选中簇的完整宽度）
        blocks_executed = int(np.ceil(max_size / layout.block_size))
        notes.append("secret_index: B 无法获知块索引，退化为全扫；"
                     "这是现有 CKKS 栈在隐藏访问模式下的真实代价")
    elif leakage_mode == "fixed_count":
        blocks_executed = int(per_query_blocks.max())
        notes.append("fixed_count: 块数按全局最大值补齐，泄漏'上限'但不泄漏每查询真实个数")
    else:
        blocks_executed = int(per_query_blocks.max())
        notes.append("public_index: 块索引明文可见，省计算但泄漏访问模式（上界估计）")

    return ExecutionPlan(
        leakage_mode=leakage_mode,
        block_size=layout.block_size,
        blocks_per_query=per_query_blocks,
        blocks_executed=blocks_executed,
        processed_positions=blocks_executed * layout.block_size,
        logical_candidates=logical,
        block_sets=block_sets,
        notes=notes,
    )
