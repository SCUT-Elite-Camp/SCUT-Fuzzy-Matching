"""现有路径的成本：全局共享矩阵 + one-hot selector + 固定形状。

算子口径与 `party_b/online_responder.py:381-406` 一致：
每 tile 做 k*d 次 ct-pt 乘（selector 选择）+ d 次 ct-ct 乘（点积）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

SLOT_CAPACITY = 4096


@dataclass
class BaselineCost:
    tile_width: int
    tiles_per_probe: int
    probes: int
    tiles: int
    ct_pt_mul: int
    ct_ct_mul: int


def baseline_cost(*, batch_size: int, clusters: int, max_size: int, dim: int, probes: int = 1) -> BaselineCost:
    T = SLOT_CAPACITY // batch_size
    tiles_per_probe = math.ceil(max_size / T)
    tiles = tiles_per_probe * probes
    return BaselineCost(
        tile_width=T,
        tiles_per_probe=tiles_per_probe,
        probes=probes,
        tiles=tiles,
        ct_pt_mul=tiles * clusters * dim,
        ct_ct_mul=tiles * dim,
    )
