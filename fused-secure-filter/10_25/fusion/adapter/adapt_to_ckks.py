"""把候选信封适配成 CKKS 批量匹配的输入布局。

三种 trust 的适配结果完全不同，且**只有前两种可运行**：

  plaintext_index → 块索引批次（Mode B）。可运行，但块索引对 B 可见，
                    等同于泄漏查询触达的记录集合。这是**性能上界**，不是隐私协议。
  fixed_shape     → 退回原始固定形状（Mode C）。可运行且保持簇隐藏，
                    但计算量与 baseline 相同（无收益）。
  secret_share    → **抛 FusionBlocked**（Mode D 不可实现）。

## 为什么 secret_share 必然失败（三条独立理由，任一成立即阻断）

1. **表示不兼容**：Approx-PSI 的输出是每人一份的 XOR 份额
   （`include/secure/ss_psi.h` 的 `flag_share`）。CKKS 只能消费密文或明文；
   份额既不是密文（不能同态运算），也不是明文（不能直接布局）。
2. **转换需要被禁的机制**：把份额变成 CKKS 可用的明文布局，只有两条路——
   交给某一方重构（等于把候选集泄给 B 或把选择结果泄给 A），
   或做同态/MPC 重加密（MPC 被禁用；而同态重加密需要额外乘法深度）。
3. **深度预算不足**：`COEFF_MOD_BIT_SIZES=[60,40,40,60]` 只提供 2 层乘法深度，
   而匹配 kernel 自身已用满 2 层（selector 的 ct-pt 乘 + query 的 ct-ct 乘）。
   任何在匹配之前插入的同态转换都需要额外深度。

此外，Approx-PSI 的实现本身依赖 OT：`CMakeLists.txt:82-84` 的
`find_package(emp-ot REQUIRED)`、PSI 层的 `volePSI::RsCpsi*`、GC 层的
`emp::SH2PCSession`。在"禁用 MPC/OT"的约束下，Approx-PSI 无法被实现——
这不是选择了不用，而是协议本身就是 MPC。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fusion.candidate_selection import ExecutionPlan, build_execution_plan
from fusion.database_layout import BlockLayout

SLOT_CAPACITY = 4096


class FusionBlocked(RuntimeError):
    """在给定约束下无法完成的衔接。必须携带可复核的理由。"""


@dataclass
class AdaptedBatch:
    mode: str                      # "A_baseline" | "B_public_index" | "C_fixed_shape"
    blocks: int                    # 固定形状下每批真正执行的块数
    processed_positions: int
    tiles: int
    ct_pt_mul: int
    ct_ct_mul: int
    leaks_candidate_index: bool
    note: str


def _tiles(positions: int, batch_size: int) -> int:
    return math.ceil(positions / (SLOT_CAPACITY // batch_size)) if positions else 0


def adapt(
    envelope: CandidateEnvelope,  # noqa: F821  (avoid circular import at runtime)
    *,
    layout: BlockLayout,
    max_size: int,
    clusters: int,
    dim: int,
    batch_size: int,
) -> AdaptedBatch:
    if envelope.trust == "plaintext_index":
        plan: ExecutionPlan = build_execution_plan(
            survivors=envelope.per_query, layout=layout, max_size=max_size,
            leakage_mode="public_index",
        )
        tiles = _tiles(plan.processed_positions, batch_size) * max(1, math.ceil(layout.block_size / (SLOT_CAPACITY // batch_size)))
        return AdaptedBatch(
            mode="B_public_index",
            blocks=plan.blocks_executed,
            processed_positions=plan.processed_positions,
            tiles=tiles,
            ct_pt_mul=tiles * dim,
            ct_ct_mul=0,
            leaks_candidate_index=True,
            note="块索引对 B 可见；本轮收益以此换取泄漏，不能称为隐私协议",
        )

    if envelope.trust == "fixed_shape":
        positions = max_size
        tiles = math.ceil(positions / (SLOT_CAPACITY // batch_size))
        return AdaptedBatch(
            mode="C_fixed_shape",
            blocks=math.ceil(max_size / layout.block_size),
            processed_positions=positions,
            tiles=tiles,
            ct_pt_mul=tiles * clusters * dim,
            ct_ct_mul=tiles * dim,
            leaks_candidate_index=False,
            note="保持簇隐藏；计算量与 baseline 相同，候选缩减未转化为计算缩减",
        )

    if envelope.trust == "secret_share":
        raise FusionBlocked(
            "secret_share 无法适配到 CKKS：份额不是密文也不是明文；"
            "重构需要向一方泄漏，重加密需要 MPC（禁用）或额外乘法深度（当前链只有 2 层，"
            "kernel 已用满）。且 Approx-PSI 的实现依赖 emp-ot 与 volePSI，禁用 OT 即无法实现。"
        )

    raise ValueError(f"unknown trust level: {envelope.trust}")
