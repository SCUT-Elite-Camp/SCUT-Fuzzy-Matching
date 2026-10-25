"""泄漏审计表：逐项标注"谁能看到 / 什么条件下 / 证据等级"。

证据等级只有三档，不允许混用：
    code_audit   —— 读过代码可以确认的行为
    measured     —— 实际运行观察到
    design       —— 设计假设，未实现/未验证
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

VISIBLE = "B 可见"
HIDDEN = "B 不可见"


@dataclass
class Row:
    item: str
    baseline: str
    public_index: str
    fixed_shape: str
    psi_fusion: str
    evidence: str


ROWS = [
    Row("查询身份/特征", HIDDEN, HIDDEN, HIDDEN, HIDDEN, "code_audit: A 侧加密后才发送"),
    Row("簇号", HIDDEN, "间接可见（由块集合反推）", HIDDEN, HIDDEN,
        "code_audit: 基线用加密 one-hot selector；公开块索引等价于暴露簇"),
    Row("候选记录索引", HIDDEN, VISIBLE, HIDDEN, "仅份额，两方都不见明文",
        "code_audit: 基线只返回掩码后分数"),
    Row("块编号", "不适用", VISIBLE, HIDDEN, "设计上不可见，未实现",
        "measured: fusion/run_fusion_plan.py 的 public_index 模式"),
    Row("候选数量", "不泄漏（固定形状）", VISIBLE, HIDDEN, "设计上不可见",
        "design: fixed_count 仅抹平上限，未证实可区分性"),
    Row("实际访问数量", "恒为 max_size 列", VISIBLE, "恒为 max_size 列", "未实现",
        "code_audit: 基线固定 25 tiles"),
    Row("数据库记录身份", HIDDEN, "可见（块内含记录）", HIDDEN, HIDDEN,
        "code_audit: 块是公开划分，索引即记录集合"),
    Row("中间密文", "B 只见密文，A 有私钥", "同基线", "同基线", "需重加密，未实现",
        "code_audit: 公开 context 不含私钥"),
    Row("解密结果", "A 得到逐候选掩码分数", "同基线", "同基线", "设计上只出 keep 位，未实现",
        "code_audit: check_tiled_score_batch 解全部 tile"),
    Row("最终输出", "catch: bool（API 层）", "catch: bool", "catch: bool", "未实现",
        "code_audit: protocol/types.py 的 MatchResult"),
]


def audit() -> list[Row]:
    return ROWS


def write_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["项目", "A 基线", "B 公开索引", "C 固定形状", "D PSI 融合", "证据等级"])
        for row in ROWS:
            writer.writerow([row.item, row.baseline, row.public_index,
                             row.fixed_shape, row.psi_fusion, row.evidence])


def render() -> str:
    widths = [12, 22, 26, 22, 26, 34]
    header = ["项目", "A 基线", "B 公开索引", "C 固定形状", "D PSI 融合", "证据等级"]
    lines = [" | ".join(text.ljust(widths[i]) for i, text in enumerate(header))]
    lines.append("-" * (sum(widths) + 3 * 5))
    for row in ROWS:
        cells = [row.item, row.baseline, row.public_index, row.fixed_shape, row.psi_fusion, row.evidence]
        lines.append(" | ".join(text.ljust(widths[i]) for i, text in enumerate(cells)))
    return "\n".join(lines)
