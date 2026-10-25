"""分块路径的成本：只处理执行计划点名的块。

与 baseline 的关键差别：块索引公开时，B 直接知道要算哪些记录，
**不再需要 one-hot selector**，每 tile 只剩逐特征 ct-pt 点积。

这正是"省计算"的来源，也是"泄漏访问模式"的来源——两者是同一件事。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

SLOT_CAPACITY = 4096


@dataclass
class BlockwiseCost:
    tile_width: int
    tiles_per_block: int
    blocks: int
    tiles: int
    ct_pt_mul: int
    ct_ct_mul: int


def blockwise_cost(*, batch_size: int, block_size: int, blocks: int, dim: int) -> BlockwiseCost:
    T = SLOT_CAPACITY // batch_size
    tiles_per_block = max(1, math.ceil(block_size / T))
    tiles = tiles_per_block * blocks
    return BlockwiseCost(
        tile_width=T,
        tiles_per_block=tiles_per_block,
        blocks=blocks,
        tiles=tiles,
        ct_pt_mul=tiles * dim,   # 无 selector
        ct_ct_mul=0,
    )
