"""候选发现与 CKKS 精匹配之间的**唯一交接对象**。

设计原则：候选从哪来、以什么信任等级到达，必须显式写在信封上，
不能靠调用方约定。适配层只接受这三种 trust：

    plaintext_index   候选 db 索引明文可用（功能验证 / 性能上界）
    fixed_shape       候选不可见，只能按固定形状全扫（隐私约束下的可达方案）
    secret_share      候选以秘密份额形式存在（Approx-PSI 的真实输出形态）

第三种是本任务要研究的目标，但**在禁用 MPC/OT 的约束下无法实现**，
详见 `adapt_to_ckks.py` 的 FusionBlocked 理由。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CandidateEnvelope:
    source: str                       # "binarization_filter" | "approx_psi"
    trust: str                        # plaintext_index | fixed_shape | secret_share
    per_query: list[np.ndarray] = field(default_factory=list)   # db 索引（逐查询）
    share_bits: int | None = None     # secret_share 模式下每个候选标志的位数
    note: str = ""

    def total_candidates(self) -> int:
        return int(sum(c.size for c in self.per_query))
