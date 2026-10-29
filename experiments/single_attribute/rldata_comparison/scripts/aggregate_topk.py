"""Aggregate the Top-k comparison table (2 schemes x 6 scenarios).

Scenarios
---------
* fuzzy  -- ``data/queries.csv``: 100 positives with edits / native umlauts,
            100 recombination negatives;
* exact  -- ``data/queries_exact.csv``: 100 verbatim positives, 100
            recombination negatives.

Rows (12): scheme 1 (KMeans + CKKS) and scheme 2 (ApproxPSI + EMP GC) for
Top-1/2/3 on both query sets.  Every row is a full ciphertext run.  The two
schemes share the "Top-k" name but not its meaning: scheme 1 probes the k most
similar clusters with early stop (``--top-k`` = probe count), scheme 2 returns
the k nearest identities of its single PSI candidate group.  Scheme 2's fuzzy
Top-1 reuses the original ``results/secure_e2e.json`` (same pipeline, three
projection seeds); every other row comes from ``results/topk/``.

Outputs (``results/topk/``): ``topk_summary.csv``, ``topk_summary.md`` and
``topk_summary.png``.  Metrics are means over the available repetitions/seeds;
the PNG and Markdown add the sample standard deviation.
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

EXPERIMENT = Path(__file__).resolve().parent.parent
RESULTS = EXPERIMENT / "results"
TOPKDIR = RESULTS / "topk"

INK = "#1F2933"
MUTED = "#5B6773"
RULE = "#D5DBE3"
BAND = "#E9EEF5"
HEADER_FILL = "#F4F7FA"
SCHEME1_COLOR = "#2F5D8A"
SCHEME2_COLOR = "#2F6F4F"
MB = 1e6

SCENARIOS = [
    ("fuzzy", "模糊姓名", 1, "topk/baseline_fuzzy_top1.json", "secure_e2e.json"),
    ("fuzzy", "模糊姓名", 2, "topk/baseline_fuzzy_top2.json", "topk/secure_fuzzy_top2/secure_e2e.json"),
    ("fuzzy", "模糊姓名", 3, "topk/baseline_fuzzy_top3.json", "topk/secure_fuzzy_top3/secure_e2e.json"),
    ("exact", "精确姓名", 1, "topk/baseline_exact_top1.json", "topk/secure_exact_top1/secure_e2e.json"),
    ("exact", "精确姓名", 2, "topk/baseline_exact_top2.json", "topk/secure_exact_top2/secure_e2e.json"),
    ("exact", "精确姓名", 3, "topk/baseline_exact_top3.json", "topk/secure_exact_top3/secure_e2e.json"),
]

METRIC_KEYS = ("accuracy", "precision", "recall", "f1")


def _mean(values):
    return sum(values) / len(values)


def _std(values):
    if len(values) < 2:
        return 0.0
    centre = _mean(values)
    return (sum((value - centre) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def load_scheme1(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    reps = payload["repetitions"]
    return {
        "runs": len(reps),
        "metrics": {
            key: [float(rep[key]) for rep in reps] for key in METRIC_KEYS
        },
        "latency": [float(rep["online_total_sec"]) for rep in reps],
        "resource": [float(rep["total_bytes"]) / MB for rep in reps],
        "top_k": int(payload["config"].get("top_k", 1)),
    }


def load_scheme2(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload["runs"]
    stats = [run["client_stats"] for run in runs]
    return {
        "runs": len(runs),
        "metrics": {
            key: [float(run["metrics"][key]) for run in runs] for key in METRIC_KEYS
        },
        "latency": [float(item["total_sec"]) for item in stats],
        "resource": [
            (
                item["psi_bytes_sent"]
                + item["psi_bytes_received"]
                + item["emp_bytes_sent"]
                + item["emp_bytes_received"]
            )
            / MB
            for item in stats
        ],
        "top_k": int(payload["config"].get("top_k", 1)),
        "identical": all(run["predictions_identical"] for run in runs),
    }


def build_rows() -> list[dict]:
    rows: list[dict] = []
    for scenario, label, top_k, one_path, two_path in SCENARIOS:
        for scheme, name, loader, colour in (
            ("scheme1", "方案1 baseline_Integration", load_scheme1, SCHEME1_COLOR),
            ("scheme2", "方案2 fused-secure-filter", load_scheme2, SCHEME2_COLOR),
        ):
            path = RESULTS / (one_path if scheme == "scheme1" else two_path)
            data = loader(path)
            if data["top_k"] != top_k:
                raise SystemExit(
                    f"{path}: expected top_k={top_k}, found {data['top_k']}"
                )
            rows.append(
                {
                    "scheme": name,
                    "colour": colour,
                    "scenario": scenario,
                    "scenario_label": label,
                    "top_k": top_k,
                    "runs": data["runs"],
                    "metrics": data["metrics"],
                    "latency": data["latency"],
                    "resource": data["resource"],
                    "source": str(path.relative_to(EXPERIMENT)),
                }
            )
    return rows


def numeric_summary(rows: list[dict]) -> list[dict]:
    return [
        {
            "scheme": row["scheme"],
            "query_set": row["scenario"],
            "top_k": row["top_k"],
            **{key: _mean(row["metrics"][key]) for key in METRIC_KEYS},
            "latency_sec": _mean(row["latency"]),
            "resource_mb": _mean(row["resource"]),
            "runs": row["runs"],
        }
        for row in rows
    ]


def _fmt_metric(values: list[float]) -> str:
    spread = _std(values)
    if spread > 0:
        return f"{_mean(values):.4f} ± {spread:.4f}"
    return f"{_mean(values):.4f}"


def _fmt_span(values: list[float], digits: int) -> str:
    low, high = min(values), max(values)
    if round(high - low, digits) == 0:
        return f"{_mean(values):.{digits}f}"
    return f"{low:.{digits}f}–{high:.{digits}f}"


def display_rows(rows: list[dict]) -> list[dict]:
    """Collapse the table for the figure.

    Scheme 2's metrics are k-invariant, so each query set contributes one row
    for it (Top-k blank, latency/traffic as the k=1/2/3 span) after the three
    scheme-1 rows.  The CSV/Markdown keep all 12 rows.
    """
    cells: list[dict] = []
    block = 0
    for scenario in ("fuzzy", "exact"):
        mine = [row for row in rows if row["scenario"] == scenario]
        scheme1 = [row for row in mine if row["scheme"].startswith("方案1")]
        scheme2 = [row for row in mine if row["scheme"].startswith("方案2")]
        for row in scheme1:
            cells.append(
                {
                    "scheme": row["scheme"],
                    "colour": row["colour"],
                    "scenario_label": row["scenario_label"],
                    "top_k_text": f"Top-{row['top_k']}",
                    "metrics": [_fmt_metric(row["metrics"][key]) for key in METRIC_KEYS],
                    "latency_text": f"{_mean(row['latency']):.2f}",
                    "resource_text": f"{_mean(row['resource']):.1f}",
                    "block": block,
                }
            )
        merged = scheme2[0]
        cells.append(
            {
                "scheme": merged["scheme"],
                "colour": merged["colour"],
                "scenario_label": merged["scenario_label"],
                "top_k_text": "\\",
                "metrics": [_fmt_metric(merged["metrics"][key]) for key in METRIC_KEYS],
                "latency_text": _fmt_span(
                    [_mean(r["latency"]) for r in scheme2], 2
                ),
                "resource_text": _fmt_span(
                    [_mean(r["resource"]) for r in scheme2], 1
                ),
                "block": block,
            }
        )
        block += 1
    return cells


def write_csv(rows: list[dict], path: Path) -> None:
    summary = numeric_summary(rows)
    fields = [
        "scheme", "query_set", "top_k", "accuracy", "precision", "recall",
        "f1", "latency_sec", "resource_mb", "runs",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summary:
            writer.writerow(
                {
                    **item,
                    **{key: f"{item[key]:.6f}" for key in METRIC_KEYS},
                    "latency_sec": f"{item['latency_sec']:.3f}",
                    "resource_mb": f"{item['resource_mb']:.3f}",
                }
            )


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = [
        "| 方案 | 查询集 | Top-k | Accuracy | Precision | Recall | F1 | 总时延 (s) | 资源开销 (MB) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        cells = [
            row["scheme"],
            row["scenario_label"],
            f"Top-{row['top_k']}",
        ]
        for key in METRIC_KEYS:
            values = row["metrics"][key]
            spread = _std(values)
            if spread > 0:
                cells.append(f"{_mean(values):.4f} ± {spread:.4f}")
            else:
                cells.append(f"{_mean(values):.4f}")
        cells.append(f"{_mean(row['latency']):.2f}")
        cells.append(f"{_mean(row['resource']):.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "> 指标为均值；有多个样本时附 ± 样本标准差（方案 1 为一次完整运行，"
        "方案 2 为 3 个投影种子）。总时延 = 在线两轮协议耗时；资源开销 = 双向通信量。"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# PNG table
# ---------------------------------------------------------------------------

COLUMNS = [
    ("方案", 0.030, 0.225),
    ("查询集", 0.225, 0.315),
    ("Top-k", 0.315, 0.375),
    ("Accuracy", 0.375, 0.480),
    ("Precision", 0.480, 0.585),
    ("Recall", 0.585, 0.690),
    ("F1", 0.690, 0.785),
    ("总时延 (s)", 0.785, 0.885),
    ("资源开销 (MB)", 0.885, 0.985),
]
TEXT_X = [0.037, 0.232, 0.322, 0.382, 0.487, 0.592, 0.697, 0.792, 0.892]

FOOTNOTES = [
    "注 1  每一行均走完完整密文流程：方案 1 为 TenSEAL CKKS 两轮密文匹配（第一轮加密簇心探测、第二轮 tiled 候选评分），"
    "方案 2 为 volePSI ApproxPSI + EMP 混淆电路两方端到端管线；明文只用于最后的指标统计。",
    "注 2  Top-k 口径不同：方案 1 = 按簇心相似度依次探测前 k 个簇、簇内命中即早停并返回单个身份，"
    "探针按「查询数 × k」批量加密；方案 2 = 在 PSI 候选组（80 轮投影桶并成一组）内返回距离最小的 k 个身份。",
    "注 3  方案 2 的四项指标不随 k 变化（失效点在候选生成与汉明距离阈值 32，而非组内排名），故每个查询集只占一行、"
    "Top-k 记为「\\」，时延与资源开销给出 k=1/2/3 的范围；完整 12 行见 CSV/MD。",
    "注 4  指标按查询级评估：正查询当且仅当返回身份等于期望身份记为命中（方案 2 为 k 个身份中任一），负查询只要返回任意身份即记为 FP。"
    "表中为均值，方案 1 每行一次完整运行、方案 2 每行为 3 个投影种子的均值，有多个样本时附样本标准差。",
    "注 5  总时延 = 在线协议耗时；资源开销 = 双向通信量。方案 1 六行同一 runner 依次执行、未重复采样，"
    "时延只读 k 的趋势（约每多一个探针 ×1.8–2）；方案 2 通信量随 k 增长，时延由 PSI 阶段主导。",
    "注 6  模糊姓名集 = data/queries.csv（100 正 / 100 负，含编辑与变音变体）；精确姓名集 = data/queries_exact.csv"
    "（100 条逐字正查询 / 100 条重组负查询）。",
]
NOTE_WIDTH = 132
NOTE_INDENT = "      "


def _setup_fonts() -> None:
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def render(rows: list[dict], output: Path) -> None:
    notes: list[str] = []
    for note in FOOTNOTES:
        notes.extend(
            textwrap.wrap(note, width=NOTE_WIDTH, subsequent_indent=NOTE_INDENT)
        )
    row_height = 0.40
    header_height = 0.85
    title_height = 1.35
    note_height = 0.21 * len(notes) + 0.5
    height = title_height + header_height + row_height * len(rows) + note_height
    width = 15.0

    fig = plt.figure(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor("white")

    def yfrac(y_in: float) -> float:
        return 1.0 - y_in / height

    def text(x: float, y_in: float, value: str, **kwargs) -> None:
        fig.text(x, yfrac(y_in), value, va="top", transform=fig.transFigure, **kwargs)

    def hline(y_in: float, color: str = RULE, lw: float = 0.7) -> None:
        fig.add_artist(
            plt.Line2D([0.030, 0.985], [yfrac(y_in)] * 2,
                       transform=fig.transFigure, color=color, lw=lw)
        )

    def band(x0: float, x1: float, y0: float, y1: float, color: str) -> None:
        fig.add_artist(
            Rectangle((x0, yfrac(y1)), x1 - x0, (y1 - y0) / height,
                      transform=fig.transFigure, facecolor=color,
                      edgecolor="none", zorder=0)
        )

    y = 0.0
    text(0.030, y + 0.10,
         "表 2  RLdata10000 单姓名 Top-k 检索：两方案全密文对比",
         fontsize=17, fontweight="bold", color=INK)
    text(0.030, y + 0.55,
         "RLdata10000 · 仅姓名属性 · 200 条查询/行 · 每行均走完完整加密流程",
         fontsize=10, color=MUTED)
    y += title_height

    band(0.030, 0.985, y, y + header_height, HEADER_FILL)
    hline(y, INK, 0.9)
    for (label, _x0, _x1), tx in zip(COLUMNS, TEXT_X):
        text(tx, y + 0.22, label, fontsize=10, fontweight="bold", color=INK)
    y += header_height
    hline(y, INK, 0.9)

    top = y
    for row in rows:
        if row["block"] % 2 == 0:
            band(0.030, 0.985, y, y + row_height, BAND)

        cells = [
            (row["scheme"], row["colour"]),
            (row["scenario_label"], INK),
            (row["top_k_text"], INK),
            *((value, INK) for value in row["metrics"]),
            (row["latency_text"], INK),
            (row["resource_text"], INK),
        ]

        for (value, colour), tx in zip(cells, TEXT_X):
            text(tx, y + 0.10, value, fontsize=8.6, color=colour)
        y += row_height
        hline(y, "#E7EBF0", 0.5)

    for x in [column[2] for column in COLUMNS[:-1]]:
        fig.add_artist(
            plt.Line2D([x, x], [yfrac(y), yfrac(top)],
                       transform=fig.transFigure, color=RULE, lw=0.7)
        )

    y += 0.18
    hline(y, INK, 0.9)
    y += 0.12
    for note in notes:
        text(0.030, y, note, fontsize=8.0, color=MUTED)
        y += 0.21

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=TOPKDIR)
    args = parser.parse_args()

    _setup_fonts()
    rows = build_rows()
    output = args.output_dir / "topk_summary"
    write_csv(rows, output.with_suffix(".csv"))
    write_markdown(rows, output.with_suffix(".md"))
    render(display_rows(rows), output)
    print(f"wrote {output.with_suffix('.csv')}")
    print(f"wrote {output.with_suffix('.md')}")
    print(f"wrote {output.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
