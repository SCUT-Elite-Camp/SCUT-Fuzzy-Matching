"""Render the scheme-1 vs scheme-2 comparison as a text-only table figure.

Both schemes are measured on their **full ciphertext pipeline**:

* scheme 1 -- ``baseline_he_test.json``: three independent repetitions of the
  real TenSEAL CKKS protocol (round 1 cluster probe + round 2 Top-1 tiles);
* scheme 2 -- ``secure_e2e.json``: three projection seeds of the end-to-end
  volePSI + EMP garbled-circuit pipeline.

Plaintext simulations, calibration runs and ablations are deliberately absent:
the figure only reports what each encrypted pipeline actually produced.

Run::

    python experiments/single_attribute/rldata_comparison/scripts/make_figure.py
    python experiments/single_attribute/rldata_comparison/scripts/make_figure.py \
        --output results/scheme_comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

EXPERIMENT = Path(__file__).resolve().parent.parent
ARTIFACTS = EXPERIMENT / "results"
FIGURES = EXPERIMENT / "results"

INK = "#1F2933"
MUTED = "#5B6773"
RULE = "#D5DBE3"
BAND = "#E9EEF5"
HEADER_FILL = "#F4F7FA"
SCHEME1_COLOR = "#2F5D8A"
SCHEME2_COLOR = "#2F6F4F"

MILLION = 1e6
MB = 1e6  # bytes -> decimal MB, matching the HE counters


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _setup_fonts() -> None:
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _mean(values):
    return sum(values) / len(values)


def _std(values):
    if len(values) < 2:
        return 0.0
    centre = _mean(values)
    return (sum((value - centre) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _range(values, digits: int) -> str:
    return f"{min(values):.{digits}f} – {max(values):.{digits}f}"


def _pm(values, digits: int) -> str:
    return f"{_mean(values):.{digits}f} ± {_std(values):.{digits}f}"


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def load_scheme1(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    reps = payload["repetitions"]
    offline = payload["offline"]["offline_sec"]
    wall = [rep["online_total_sec"] + offline for rep in reps]
    return {
        "config": payload["config"],
        "reps": len(reps),
        "offline_sec": offline,
        "offline_mb": payload["offline"]["offline_memory_peak_mb"],
        "round1": [rep["round1_sec"] for rep in reps],
        "round2": [rep["round2_sec"] for rep in reps],
        "online": [rep["online_total_sec"] for rep in reps],
        "wall": wall,
        "sent": [rep["sent_bytes"] for rep in reps],
        "recv": [rep["recv_bytes"] for rep in reps],
        "total": [rep["total_bytes"] for rep in reps],
        "context": [rep["public_context_bytes"] for rep in reps],
        "rss": [rep["process_rss_peak_mb"] for rep in reps],
        "tp": [rep["tp"] for rep in reps],
        "fp": [rep["fp"] for rep in reps],
        "fn": [rep["fn"] for rep in reps],
        "tn": [rep["tn"] for rep in reps],
        "precision": [rep["precision"] for rep in reps],
        "recall": [rep["recall"] for rep in reps],
        "f1": [rep["f1"] for rep in reps],
        "accuracy": [rep["accuracy"] for rep in reps],
    }


def load_scheme2(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload["runs"]
    config = payload["config"]
    stats = [run["client_stats"] for run in runs]
    totals = [
        item["psi_bytes_sent"] + item["psi_bytes_received"]
        + item["emp_bytes_sent"] + item["emp_bytes_received"]
        for item in stats
    ]
    psi = [item["psi_bytes_sent"] + item["psi_bytes_received"] for item in stats]
    emp = [item["emp_bytes_sent"] + item["emp_bytes_received"] for item in stats]
    return {
        "config": config,
        "seeds": payload["seeds"],
        "runs": runs,
        "rounds": config["rounds"],
        "slots": config["slots"],
        "distance": config["distance_threshold"],
        "probability": config["probability"],
        "export": [run["export_sec"]["server"] for run in runs],
        "phase1": [item["phase1_sec"] for item in stats],
        "phase3": [item["phase3_sec"] for item in stats],
        "total": [item["total_sec"] for item in stats],
        "wall": [run["wall_sec"] for run in runs],
        "sent": [item["psi_bytes_sent"] + item["emp_bytes_sent"] for item in stats],
        "recv": [item["psi_bytes_received"] + item["emp_bytes_received"] for item in stats],
        "total_bytes": totals,
        "psi": psi,
        "emp": emp,
        "candidates": [item["candidates"] for item in stats],
        "groups": [item["groups"] for item in stats],
        "and_gates": [item["and_gates"] for item in stats],
        "psi_bins": [item["psi_bins"] for item in stats],
        "hit_bins": [item["hit_bins"] for item in stats],
        "psi_items_server": [run["server_stats"]["psi_input_items"] for run in runs],
        "psi_items_client": [item["psi_input_items"] for item in stats],
        "precision": [run["metrics"]["precision"] for run in runs],
        "recall": [run["metrics"]["recall"] for run in runs],
        "f1": [run["metrics"]["f1"] for run in runs],
        "accuracy": [run["metrics"]["accuracy"] for run in runs],
        "tp": [run["metrics"]["tp"] for run in runs],
        "fp": [run["metrics"]["fp"] for run in runs],
        "fn": [run["metrics"]["fn"] for run in runs],
        "tn": [run["metrics"]["tn"] for run in runs],
        "identical": all(run["predictions_identical"] for run in runs),
        "queries": 200,
    }


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------


def build_rows(one: dict, two: dict) -> list[dict]:
    """Rows as ``{"kind": "section"|"row", ...}`` in display order."""
    cfg1 = one["config"]
    queries = two["queries"]

    s1_wall = _mean(one["wall"])
    s2_wall = _mean(two["wall"])
    speedup = s1_wall / s2_wall
    s1_rss = _mean(one["rss"])
    s2_rss = 72.4  # measured peak of the slower party (client), see footnote 4
    mem_gain = s1_rss / s2_rss
    comm_gain = _mean(two["total_bytes"]) / _mean(one["total"])

    rows: list[dict] = []

    def section(label: str) -> None:
        rows.append({"kind": "section", "label": label})

    def row(category: str, metric: str, left: str, right: str) -> None:
        rows.append({
            "kind": "row", "category": category, "metric": metric,
            "left": left, "right": right,
        })

    # ---- I. configuration ------------------------------------------------
    section("I. 系统配置与实验设置")
    row("数据规模",
        "样本库 / 查询数",
        f"RLdata10000，10,000 条记录\n{queries} 条查询",
        f"RLdata10000，10,000 条记录\n{queries} 条查询")
    row("匹配属性与编码",
        "字段 / 字符 n-gram / 向量",
        "仅姓名；2-gram MinHash 签名\n（200 维聚类 / 50 维评分）",
        "仅姓名；2-gram；256 位 b-bit MinHash\n（128 置换 × 2 位）")
    row("密码学后端",
        "安全计算原语",
        f"TenSEAL CKKS：N={cfg1['poly_modulus_degree']}，\n"
        f"模 {cfg1['coeff_mod_bit_sizes']}，scale 2^40，\n"
        f"批次 {cfg1['he_batch_size']}",
        "volePSI ApproxPSI + EMP 混淆电路\n（两方半诚实模型）")
    row("匹配参数",
        "候选与判定阈值",
        f"KMeans k={cfg1['k']}；相似度阈值 τ={cfg1['tau']}；\n"
        f"shingle={cfg1['shingle_size']}",
        f"投影保留率 p={two['probability']}；{two['rounds']} 轮；\n"
        f"槽位 S={two['slots']}；汉明阈值 d={two['distance']}")
    row("结果可见性",
        "Top-1 身份归属",
        "仅查询方获得 Top-1 身份",
        "仅查询方获得 Top-1 身份\n（服务端输出不含预测）")

    # ---- II. accuracy ----------------------------------------------------
    section("II. 精度与混淆矩阵评估")
    row("混淆矩阵",
        "TP / FP / FN / TN",
        " / ".join(f"{_mean(one[key]):.1f}" for key in ("tp", "fp", "fn", "tn")),
        " / ".join(f"{_mean(two[key]):.1f}" for key in ("tp", "fp", "fn", "tn")))
    row("精确率", "Precision",
        _pm(one["precision"], 4), _pm(two["precision"], 4))
    row("召回率", "Recall",
        _pm(one["recall"], 4), _pm(two["recall"], 4))
    row("F1 分数", "F1 = 2PR / (P + R)",
        _pm(one["f1"], 4), _pm(two["f1"], 4))
    row("准确率", "Accuracy = (TP + TN) / N",
        _pm(one["accuracy"], 4), _pm(two["accuracy"], 4))
    row("重复间稳定性",
        "多次运行的一致性",
        f"{one['reps']} 次重复 TP = "
        + " / ".join(str(value) for value in one["tp"])
        + f"；F1 极差 {max(one['f1']) - min(one['f1']):.4f}",
        f"{len(two['seeds'])} 个种子 200/200 预测\n与槽位参考逐条一致"
        if two["identical"] else "存在不一致预测")

    # ---- III. timing and resources --------------------------------------
    section("III. 端到端时延与资源开销")
    row("离线准备",
        "耗时 / 内存",
        f"{one['offline_sec']:.2f} s（Python 峰值 {one['offline_mb']:.1f} MB）",
        f"{_pm(two['export'], 2)} s（本地编码导出）")
    row("第一轮",
        "候选生成",
        f"{_pm(one['round1'], 2)} s\n（{_range(one['round1'], 2)}）",
        f"{_pm(two['phase1'], 2)} s\n（{two['rounds']} 轮 PSI）")
    row("第二轮",
        "密文验证",
        f"{_pm(one['round2'], 2)} s\n（{_range(one['round2'], 2)}）",
        f"{_pm(two['phase3'], 2)} s\n（每候选组一次 GC）")
    row("在线协议总时延",
        "两轮合计",
        f"{_pm(one['online'], 2)} s\n（{_range(one['online'], 2)}）",
        f"{_pm(two['total'], 2)} s\n（{_range(two['total'], 2)}）")
    row("完整流程墙钟",
        "离线 + 在线总耗时",
        f"{_pm(one['wall'], 2)} s",
        f"{_pm(two['wall'], 2)} s")
    row("双向通信量",
        "发送 + 接收",
        f"{_mean(one['total']) / MB:.2f} MB\n"
        f"（发 {_mean(one['sent']) / MB:.2f} / 收 {_mean(one['recv']) / MB:.2f}）",
        f"{_mean(two['total_bytes']) / MB:.2f} MB\n"
        f"（PSI {_mean(two['psi']) / MB:.2f} / "
        f"GC {_mean(two['emp']) / MB:.2f}）")
    row("公共参考上下文",
        "与查询无关的公开数据",
        f"{_mean(one['context']) / MB:.2f} MB",
        "0（无公共上下文）")
    row("单进程峰值内存",
        "最大 RSS",
        f"{_pm(one['rss'], 1)} MB\n（{_range(one['rss'], 1)}）",
        f"{s2_rss:.1f} MB\n（服务端 67.5 / 客户端 72.4）")

    # ---- IV. work and gains ---------------------------------------------
    section("IV. 计算工作量与优化效果")
    row("每查询在线时延",
        "在线总时延 / 查询数",
        f"{_mean(one['online']) / queries:.3f} s",
        f"{_mean(two['total']) / queries * 1e3:.1f} ms")
    row("每查询通信量",
        "双向总流量 / 查询数",
        f"{_mean(one['total']) / queries / MB:.3f} MB",
        f"{_mean(two['total_bytes']) / queries / MB:.3f} MB")
    row("密文计算工作量",
        "每查询参与比较的数据",
        f"第一轮 {cfg1['k']} 个簇心；第二轮对选中簇\n"
        f"逐条做 Top-1 密文评分（批次 {cfg1['he_batch_size']}）",
        f"候选对 {_mean(two['candidates']):.1f}；\n"
        f"混淆电路 AND 门 {_mean(two['and_gates']) / MILLION:.2f} M")
    row("候选生成规模",
        "PSI 输入与命中",
        "不适用（密文聚类直接给出候选簇）",
        f"输入项 {_mean(two['psi_items_server']):,.0f}（库方） / "
        f"{_mean(two['psi_items_client']):,.0f}（查询方）；\n"
        f"桶 {_mean(two['psi_bins']):,.0f}；命中桶 {_mean(two['hit_bins']):,.0f}")
    row("候选组规模",
        "验证阶段的电路数",
        "不适用",
        f"{_mean(two['candidates']):.1f} 个候选对，\n"
        f"{_mean(two['groups']):.0f} 个候选组")
    row("相对方案 1 的整体变化",
        "完整流程实测比值",
        "基准（1.00×）",
        f"时延 ↓{speedup:.1f}×；内存 ↓{mem_gain:.1f}×；\n"
        f"通信 ↑{comm_gain:.2f}×")
    return rows


FOOTNOTES = [
    "注 1  两方案均只统计完整密文流程的实测结果；明文模拟、明文标定与消融实验数据均未列入本表。",
    "注 2  方案 1 为 3 次独立重复的 TenSEAL CKKS 全密文运行；方案 2 为 3 个投影种子"
    "（20261023 / 20261040 / 20261057）的 volePSI + EMP 全流程运行。",
    "注 3  数值为均值 ± 样本标准差，括号内为最小值 – 最大值；TP / FP / FN / TN 为 200 条查询上的计数（均值）。",
    "注 4  方案 2 的峰值内存为 seed 20261023 单次全流程经 /usr/bin/time -v 测得："
    "服务端 67.5 MB、客户端 72.4 MB，表中取两端较大值。",
    "注 5  方案 2 每个种子的 200 条 Top-1 预测均与 S=8 槽位参考逐条一致，"
    "故表内指标即端到端安全流程的输出指标。",
    "注 6  加速比 = 方案 1 完整流程墙钟均值 / 方案 2 墙钟均值；"
    "内存下降 = 方案 1 RSS 均值 / 方案 2 峰值；通信增幅 = 方案 2 / 方案 1。",
]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

COLUMNS = [
    ("类别 / 维度", 0.035, 0.150, INK),
    ("具体指标 / 参数", 0.150, 0.445, INK),
    ("方案 1：KMeans + CKKS", 0.445, 0.710, SCHEME1_COLOR),
    ("方案 2：ApproxPSI + EMP GC", 0.710, 0.975, SCHEME2_COLOR),
]
TEXT_X = [0.047, 0.162, 0.457, 0.722]

ROW_UNITS = 0.435      # inches per single-line row
LINE_UNITS = 0.175     # extra inches per additional line
SECTION_UNITS = 0.395
HEADER_UNITS = 1.42
FOOTNOTE_LINE = 0.185


def _lines(text: str) -> int:
    return text.count("\n") + 1


def render(rows: list[dict], output: Path) -> None:
    total_units = HEADER_UNITS + 0.62
    for item in rows:
        if item["kind"] == "section":
            total_units += SECTION_UNITS
        else:
            extra = max(_lines(item[key]) for key in ("category", "metric", "left", "right")) - 1
            total_units += ROW_UNITS + extra * LINE_UNITS
    total_units += 0.46 + FOOTNOTE_LINE * len(FOOTNOTES) + 0.28

    width = 13.6
    height = total_units
    fig = plt.figure(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor("white")

    def yfrac(y_in: float) -> float:
        return 1.0 - y_in / height

    def text(x: float, y_in: float, value: str, **kwargs) -> None:
        fig.text(x, yfrac(y_in), value, va="top", transform=fig.transFigure, **kwargs)

    def hline(y_in: float, color: str = RULE, lw: float = 0.7) -> None:
        fig.add_artist(plt.Line2D([0.035, 0.975], [yfrac(y_in)] * 2,
                                  transform=fig.transFigure, color=color, lw=lw))

    def band(x0: float, x1: float, y0: float, y1: float, color: str) -> None:
        fig.add_artist(Rectangle((x0, yfrac(y1)), x1 - x0, (y1 - y0) / height,
                                 transform=fig.transFigure, facecolor=color,
                                 edgecolor="none", zorder=0))

    y = 0.0

    # ---- title -----------------------------------------------------------
    text(0.035, y + 0.06,
         "表 1  两方案端到端密文姓名匹配综合评估",
         fontsize=18, fontweight="bold", color=INK)
    text(0.035, y + 0.52,
         "数据集 RLdata10000 · 仅姓名属性 · 200 条查询 · "
         "两方案均已走完完整密文匹配流程",
         fontsize=10.5, color=MUTED)
    text(0.035, y + 0.86,
         "表中仅含密文全流程实测数据；明文模拟与明文标定结果未列入",
         fontsize=9.5, color="#9A4B1E", style="italic")
    y += HEADER_UNITS

    # ---- column header ---------------------------------------------------
    band(0.035, 0.975, y, y + 0.62, HEADER_FILL)
    hline(y, INK, 0.9)
    for (label, _x0, _x1, color), tx in zip(COLUMNS, TEXT_X):
        text(tx, y + 0.17, label, fontsize=10.5, fontweight="bold", color=color)
    y += 0.62
    hline(y, INK, 0.9)

    # ---- body ------------------------------------------------------------
    table_top = y
    for item in rows:
        if item["kind"] == "section":
            band(0.035, 0.975, y, y + SECTION_UNITS, BAND)
            text(0.047, y + 0.10, item["label"],
                 fontsize=11, fontweight="bold", color=INK)
            y += SECTION_UNITS
            hline(y, "#AEB8C4", 0.8)
            continue

        extra = max(_lines(item[key]) for key in
                    ("category", "metric", "left", "right")) - 1
        row_height = ROW_UNITS + extra * LINE_UNITS
        text(TEXT_X[0], y + 0.13, item["category"], fontsize=9.2, color=INK)
        text(TEXT_X[1], y + 0.13, item["metric"], fontsize=9.2, color=INK)
        text(TEXT_X[2], y + 0.13, item["left"], fontsize=9.2, color=INK)
        text(TEXT_X[3], y + 0.13, item["right"], fontsize=9.2, color=INK)
        y += row_height
        hline(y, "#E7EBF0", 0.55)

    # ---- vertical column rules ------------------------------------------
    for x in (COLUMNS[1][1], COLUMNS[2][1], COLUMNS[3][1]):
        fig.add_artist(plt.Line2D([x, x], [yfrac(y), yfrac(table_top)],
                                  transform=fig.transFigure,
                                  color=RULE, lw=0.8))

    # ---- footnotes -------------------------------------------------------
    y += 0.18
    hline(y, INK, 0.9)
    y += 0.14
    for note in FOOTNOTES:
        text(0.035, y, note, fontsize=8.4, color=MUTED)
        y += FOOTNOTE_LINE

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=200,
                facecolor="white", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
                        default=ARTIFACTS / "baseline_he_test.json")
    parser.add_argument("--secure", type=Path,
                        default=ARTIFACTS / "secure_e2e.json")
    parser.add_argument("--output", type=Path,
                        default=FIGURES / "scheme_comparison")
    args = parser.parse_args()

    _setup_fonts()
    one = load_scheme1(args.baseline)
    two = load_scheme2(args.secure)
    render(build_rows(one, two), args.output)
    print(f"wrote {args.output.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
