"""Side-by-side comparison of the two schemes' data preprocessing.

The top block lists what both pipelines share (same rows, same field, same
normalisation, same 2-gram shingles, same SHA-256 hash family); the bottom
block starts where they diverge (feature form, dimensionality, fitted vs
data-independent structures, metric and threshold).

Output: ``data/preprocess_comparison.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

EXPERIMENT = Path(__file__).resolve().parent.parent
OUTPUT = EXPERIMENT / "data" / "preprocess_comparison.png"

INK = "#1F2933"
MUTED = "#5B6773"
RULE = "#D5DBE3"
BAND = "#E9EEF5"
HEADER_FILL = "#F4F7FA"
SCHEME1 = "#2F5D8A"
SCHEME2 = "#2F6F4F"

COLUMNS = [
    ("KMeans聚类+CKKS余弦", SCHEME1),
    ("随机投影分桶+Hamming GC", SCHEME2),
]

SHARED = [
    ("输入数据", "同一份 database.csv（10k 行）与同一批查询集"),
    ("使用字段", "只用 normalized_name；DOB 读入但未参与匹配"),
    ("文本归一化", "NFKC · casefold · 变音转写 · 去符号 · ASCII 化"),
    ("分片粒度", "字符 2-gram"),
    ("哈希原语", "SHA-256 → mod 2^20，a·x + b，seed 42"),
]

DIVERGED = [
    ("特征形态", "200 / 50 维浮点型向量", "256 位二进制串"),
    ("匹配维度", "聚类 200 维，精判取前缀 50 维", "全程 256 位"),
    ("检索结构", "KMeans 簇（离线训练）", "80 轮随机坐标子集 → 桶，8 槽/桶"),
    ("相似度度量", "余弦", "汉明"),
    ("判定阈值", "余弦相似度 > 0.926（+1e-4 噪声余量）", "汉明距离 ≤ 32"),
    ("随机性", "哈希 seed 42，单次确定性", "seed 42 + 3 个投影种子取均值"),
]

NOTES = [
    "注 1  两侧的差别全部发生在签名数值化之后：浮点最小值 → 2 位量化会丢信息，度量也从余弦换成汉明，因此两个阈值不可互相换算。",
    "注 2  0.926 由独立标定集扫出的 F1 最优点确定；32/256 沿用既有默认值，未在本数据集上标定。",
]

X_LABEL = 0.030
X_COL1 = 0.165
X_COL2 = 0.565
X_RIGHT = 0.985
DIVIDER1 = 0.155
DIVIDER2 = 0.555


def _setup_fonts() -> None:
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def build() -> None:
    _setup_fonts()
    title_height = 1.30
    header_height = 0.72
    section_height = 0.42
    row_height = 0.36
    note_height = 0.24 * len(NOTES) + 0.55
    blocks = 2
    rows = len(SHARED) + len(DIVERGED)
    height = (
        title_height + header_height + section_height * blocks
        + row_height * rows + note_height
    )
    width = 12.0

    fig = plt.figure(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor("white")

    def yfrac(y_in: float) -> float:
        return 1.0 - y_in / height

    def text(x: float, y_in: float, value: str, **kwargs) -> None:
        fig.text(x, yfrac(y_in), value, va="top", transform=fig.transFigure, **kwargs)

    def hline(y_in: float, color: str = RULE, lw: float = 0.7) -> None:
        fig.add_artist(
            plt.Line2D([X_LABEL, X_RIGHT], [yfrac(y_in)] * 2,
                       transform=fig.transFigure, color=color, lw=lw)
        )

    def band(x0: float, x1: float, y0: float, y1: float, color: str) -> None:
        fig.add_artist(
            Rectangle((x0, yfrac(y1)), x1 - x0, (y1 - y0) / height,
                      transform=fig.transFigure, facecolor=color,
                      edgecolor="none", zorder=0)
        )

    def vline(x: float, y0: float, y1: float, color: str = RULE) -> None:
        fig.add_artist(
            plt.Line2D([x, x], [yfrac(y1), yfrac(y0)],
                       transform=fig.transFigure, color=color, lw=0.7)
        )

    y = 0.0
    text(X_LABEL, y + 0.10,
         "两方案数据预处理对比（RLdata10000 · 单姓名属性）",
         fontsize=17, fontweight="bold", color=INK)
    text(X_LABEL, y + 0.58,
         "上半：共同环节（同一份数据、同一套归一化与哈希）；下半：特征数值化开始分叉",
         fontsize=10, color=MUTED)
    y += title_height

    # ---------------- header ----------------
    band(X_LABEL, X_RIGHT, y, y + header_height, HEADER_FILL)
    hline(y, INK, 0.9)
    text(X_LABEL, y + 0.20, "环节", fontsize=10.5, fontweight="bold", color=INK)
    for (label, colour), x in zip(COLUMNS, (X_COL1, X_COL2)):
        text(x, y + 0.20, label, fontsize=10.5, fontweight="bold", color=colour)
    y += header_height
    hline(y, INK, 0.9)
    top = y

    def section(title: str, y_in: float) -> float:
        band(X_LABEL, X_RIGHT, y_in, y_in + section_height, "#EEF2F7")
        fig.text((X_LABEL + X_RIGHT) / 2, yfrac(y_in + 0.11), title,
                 ha="center", va="top", transform=fig.transFigure,
                 fontsize=9.4, fontweight="bold", color=MUTED)
        out = y_in + section_height
        hline(out, "#E7EBF0", 0.5)
        return out

    # ---------------- shared block ----------------
    y = section("共同环节", y)
    for index, (label, value) in enumerate(SHARED):
        if index % 2 == 1:
            band(X_LABEL, X_RIGHT, y, y + row_height, BAND)
        text(X_LABEL, y + 0.09, label, fontsize=9.2, color=INK)
        text(X_COL1, y + 0.09, value, fontsize=9.2, color=INK)
        y += row_height

    # ---------------- diverged block ----------------
    y = section("分叉环节", y)
    for index, (label, left, right) in enumerate(DIVERGED):
        if index % 2 == 0:
            band(X_LABEL, X_RIGHT, y, y + row_height, BAND)
        text(X_LABEL, y + 0.09, label, fontsize=9.2, color=INK)
        text(X_COL1, y + 0.09, left, fontsize=9.2, color=INK)
        text(X_COL2, y + 0.09, right, fontsize=9.2, color=INK)
        y += row_height

    # column separators: the scheme divider only spans the diverged block
    diverge_start = top + section_height + row_height * len(SHARED)
    vline(DIVIDER1, diverge_start, y)
    vline(DIVIDER2, diverge_start, y)

    y += 0.20
    hline(y, INK, 0.9)
    y += 0.14
    for note in NOTES:
        text(X_LABEL, y, note, fontsize=8.4, color=MUTED)
        y += 0.24

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    build()
