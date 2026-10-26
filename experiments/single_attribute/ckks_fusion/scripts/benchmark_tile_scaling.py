#!/usr/bin/env python3
"""tile 成本标定：减少 tile 数能否在现有 CKKS 路径中带来可测的端到端加速。

## 审计结论（本脚本据此修正）

对 `scripts/benchmark_he_batching.py:run_tiled_benchmark` 的静态审计发现四处问题：

  A1  无预热。每次调用都新建 context + keygen，首次运行含冷启动开销，原脚本未丢弃。
  A2  `round2_b_kernel` **包含 tile 序列化**（`serialize_output=True`，生成器内
      `score_tile.serialize()` 落在计时区间内）。所以它是"kernel + 序列化"。
  A3  `online_total` **包含 context 与 keygen**（`total_start` 在 context 创建之前）。
      截距 a 因此吸收了一次性 setup，不能当成"在线固定开销"。
  A4  无逐次原始记录、无环境信息、无离散度，只报中位数；无法复核也看不到噪声。

同时确认三件"公平性"事实：
  ✓ `columns → tiles` 映射正确且被脚本自检（`expected_r2_tiles` 断言）；
    T = 4096//200 = 20 → 20/100/200/483 列对应 1/5/10/25 tiles，与生产 max_size→L 一致。
  ✓ 各规模共用同一 seed，query/centroid 向量相同，只有 cluster_matrix 的宽度变化。
  ✓ kernel 消费全部 tile（`list(...)`），无早退。

## 计时层次

  kernel_tile_avg    = round2_b_kernel / tiles     每 tile 平均（含序列化，见 A2）
  round2_b_kernel    多 tile kernel 总时间
  online_total       完整在线阶段（含 setup，见 A3）
  online_ex_setup    = online_total - context_and_public_key   扣除 keygen 的派生量

## 顺序效应

交错测量只让同一 repeat 内的各规模共享噪声，**不能消除顺序偏差**（温度、缓存、
系统负载随位置漂移）。因此默认对每个 repeat **轮换测量顺序**（`--rotate-order`，
循环左移 `repeat % len(columns)` 位），并把每次运行的 `sequence_index` 与
`timestamp` 逐条写入 `runs.csv`，便于事后检查顺序与耗时的相关性。

轮换只是缓解，不是消除。k 个规模、n 次重复时：n 为 k 的倍数则每个规模在每一位
出现次数完全相同；否则近似平衡（如 k=4, n=5 时第 0 位多出现一次）。
脚本同时输出 `order_effect`（`sequence_index` 与耗时的 Pearson r，以及各位置的中位数），
用于事后判断是否存在顺序漂移。

## n 的性质

`--repeats 3` 只是最低可运行配置；建议起步 `--repeats 5`。
**5 次是起步，不是保证得出结论的次数**：结果接近门限、差值波动大、或 kernel 与
端到端判断不一致时，应继续采样，而不是调整阈值。

## 用法

    python scripts/benchmark_tile_scaling.py --repeats 5            # 建议入口
    python scripts/benchmark_tile_scaling.py --self-test            # 只验证统计/判定逻辑
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from datetime import datetime
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

STAGES = ("round2_b_kernel", "round2_a_decrypt", "online_total",
          "context_and_public_key", "round1_b_kernel")

# 双侧 97.5% 的 t 分位数（df = n-1）。n=3 时是 4.303，不是 2。
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060,
         30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t975(df: int) -> float:
    """双侧 97.5% t 分位数；df>120 用正态近似 1.960。"""
    if df <= 0:
        return float("nan")
    for key in sorted(_T975):
        if df <= key:
            return _T975[key]
    return 1.960


# ---------------------------------------------------------------------------
# 纯函数：统计、拟合、判定（可在无 TenSEAL 环境下自检）
# ---------------------------------------------------------------------------


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else float("nan")
    return {
        "n": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "std": std,
        "cov": (std / mean) if mean > 0 and len(values) > 1 else float("nan"),
        "min": min(values),
        "max": max(values),
    }


def fit_linear(xs: list[float], ys: list[float]) -> dict:
    """t = a + b·tiles。明确标注为经验拟合，不是运行规律。"""
    if len(xs) < 2 or float(np.ptp(xs)) == 0.0:
        return {"applicable": False, "reason": "需要至少两个不同的 tile 数"}
    b, a = np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)
    pred = a + b * np.asarray(xs, float)
    residuals = [float(y - p) for y, p in zip(ys, pred)]
    ss_res = float(((np.asarray(ys) - pred) ** 2).sum())
    ss_tot = float(((np.asarray(ys) - np.mean(ys)) ** 2).sum())
    return {
        "applicable": True,
        "a_intercept": float(a),
        "b_slope_per_tile": float(b),
        "r_squared": (1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "residuals": residuals,
        "intercept_negative": bool(a < 0),
        "note": "有限测量点的经验拟合；R² 不能单独作为线性关系成立的证据",
    }


def decide(per_size_online: dict[int, list[float]],
           per_size_kernel: dict[int, list[float]]) -> dict:
    """三层判定。25 与 10 tiles 用**配对差**（同一 repeat 内交替测量）。

    门槛（全部为**预先设定的工程阈值**，不是统计定律）：
      - CoV <= 5% 且每规模 n >= 3；
      - |配对差均值| > t(0.975, n-1) x SE_d；
      - 相对改善 > 5%。
    任一不满足都不落入"支持继续研究"。
    """
    tiles = sorted(per_size_online)
    if len(tiles) < 2:
        return {"verdict": "暂不确定", "reason": "规模点不足"}

    stats = {t: summarize(per_size_online[t]) for t in tiles}
    n_min = min(stats[t]["n"] for t in tiles)
    cov_max = max((stats[t]["cov"] for t in tiles if not math.isnan(stats[t]["cov"])),
                  default=float("nan"))
    kernel_fit = fit_linear(tiles, [summarize(per_size_kernel[t])["median"] for t in tiles])

    if not (25 in stats and 10 in stats):
        return {"verdict": "暂不确定", "reason": "缺少 25 或 10 tiles 的测量点",
                "kernel_fit": kernel_fit}

    hi, lo = per_size_online[25], per_size_online[10]
    paired = len(hi) == len(lo) and len(hi) >= 2
    if paired:
        diffs = [a - b for a, b in zip(hi, lo)]
        d_mean = statistics.fmean(diffs)
        d_sd = statistics.stdev(diffs)
        d_se = d_sd / math.sqrt(len(diffs))
        df = len(diffs) - 1
        tcrit = t975(df)
        threshold = tcrit * d_se
        design = "paired"
    else:
        # 非配对后备：Welch 标准误，并在结论里标明设计
        d_mean = stats[25]["median"] - stats[10]["median"]
        s1 = 0.0 if math.isnan(stats[25]["std"]) else stats[25]["std"]
        s2 = 0.0 if math.isnan(stats[10]["std"]) else stats[10]["std"]
        d_se = math.sqrt(s1 ** 2 / max(1, stats[25]["n"]) + s2 ** 2 / max(1, stats[10]["n"]))
        df = min(stats[25]["n"], stats[10]["n"]) - 1
        tcrit = t975(max(1, df))
        threshold = tcrit * d_se
        design = "unpaired_welch"

    relative = d_mean / stats[25]["median"] if stats[25]["median"] else float("nan")
    base = {"design": design, "delta_seconds": d_mean, "relative": relative,
            "se": d_se, "t_crit": tcrit, "threshold": threshold,
            "n_pairs": len(hi) if paired else None, "kernel_fit": kernel_fit,
            "thresholds": "CoV<=5%, n>=3, |delta|>t*SE, relative>5% 均为预设工程阈值"}

    if n_min < 3 or (not math.isnan(cov_max) and cov_max > 0.05):
        return {**base, "verdict": "暂不确定",
                "reason": f"样本不足或波动过大（n_min={n_min}, 最大 CoV={cov_max:.3f}）；"
                          f"先增加重复次数，不作方向性结论"}

    if design == "unpaired_welch":
        return {**base, "verdict": "暂不确定",
                "reason": "两次测量未能配对（样本数不等），无法计算配对差的标准误"}

    if d_mean > threshold and relative > 0.05:
        return {**base, "verdict": "支持继续研究",
                "reason": f"配对差 {d_mean:.3f}s（{relative*100:.1f}%），超过 "
                          f"t({df})×SE = {threshold:.3f}s"}

    if relative < 0.05:
        return {**base, "verdict": "当前实现下收益有限",
                "reason": f"测量稳定但端到端仅减少 {relative*100:.1f}%（<5% 预设门槛），"
                          f"减少 tile 未有效传递到在线阶段"}

    return {**base, "verdict": "暂不确定",
            "reason": f"配对差 {d_mean:.3f}s 与门限 {threshold:.3f}s 接近，证据不足以定向"}


# ---------------------------------------------------------------------------
# 自检：用合成数据验证统计与判定逻辑（不涉及 CKKS）
# ---------------------------------------------------------------------------


def self_test() -> int:
    import random
    rng = random.Random(0)

    def synth(a: float, b: float, tiles: list[int], noise: float, n: int = 5) -> dict[int, list[float]]:
        return {t: [a + b * t + rng.gauss(0, noise) for _ in range(n)] for t in tiles}

    cases = [
        ("支持继续研究", synth(0.4, 0.12, [1, 5, 10, 25], 0.01)),
        ("暂不确定", synth(0.4, 0.12, [1, 5, 10, 25], 0.9)),
        ("当前实现下收益有限", synth(9.0, 0.004, [1, 5, 10, 25], 0.005)),
        # 关键判别用例（确定性数据）：配对差均值 0.6s，但 sd=0.3 → SE=0.173，
        # t(2)×SE=0.745 > 0.6 → 暂不确定。旧的 2×SE 判据（0.346）会误判为"支持"。
        ("暂不确定", {25: [9.7, 10.0, 10.3], 10: [9.4, 9.4, 9.4],
                      5: [9.1, 9.1, 9.1], 1: [9.0, 9.0, 9.0]}),
    ]
    ok = True
    for expected, data in cases:
        got = decide(data, data)["verdict"]
        status = "OK " if got == expected else "FAIL"
        ok &= got == expected
        print(f"  [{status}] 期望 {expected:<12} 得到 {got}")

    fit = fit_linear([1, 5, 10, 25], [0.52, 1.0, 1.6, 3.4])
    print(f"  [{'OK ' if abs(fit['b_slope_per_tile'] - 0.115) < 0.02 else 'FAIL'}] 线性拟合斜率="
          f"{fit['b_slope_per_tile']:.4f} R²={fit['r_squared']:.4f}")
    summary = summarize([1.0, 2.0, 3.0])
    print(f"  [{'OK ' if summary['n'] == 3 and abs(summary['median'] - 2.0) < 1e-9 else 'FAIL'}] "
          f"统计 n={summary['n']} median={summary['median']}")
    print("\n注意：以上为**逻辑测试结果**（合成数据），不是性能实测，不能用于判断 CKKS 快慢。")
    print("自检通过" if ok else "自检失败")
    return 0 if ok else 1


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--columns", type=int, nargs="+", default=[20, 100, 200, 483])
    ap.add_argument("--clusters", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1, help="每个规模先跑并丢弃的次数")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "tile_scaling")
    ap.add_argument("--rotate-order", action="store_true", default=True,
                    help="每个 repeat 轮换测量顺序，缓解顺序效应（默认开启）")
    ap.add_argument("--no-rotate-order", dest="rotate_order", action="store_false")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "tenseal": None,
        "coeff_mod_bit_sizes": None,
        "poly_modulus_degree": None,
    }
    try:
        import tenseal
        env["tenseal"] = getattr(tenseal, "__version__", "unknown")
    except ImportError:
        print("tenseal 未安装：未做任何测量。以下仅为代码审计结论与可执行实验。")
        print(f"环境: python={env['python']}, numpy={env['numpy']}")
        return 2

    from config.params import COEFF_MOD_BIT_SIZES, POLY_MODULUS_DEGREE, SCALE
    env.update(coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
               poly_modulus_degree=POLY_MODULUS_DEGREE, scale_bits=SCALE.bit_length() - 1)

    from benchmark_he_batching import run_tiled_benchmark

    raw: list[dict] = []
    by_size_online: dict[int, list[float]] = {}
    by_size_kernel: dict[int, list[float]] = {}

    # 先统一预热（每个规模各丢弃 --warmup 次）
    for columns in args.columns:
        for _ in range(args.warmup):
            run_tiled_benchmark(batch_size=args.batch_size, clusters=args.clusters, columns=columns)

    # 交错运行 + 每个 repeat 轮换顺序，缓解顺序效应
    orders: list[list[int]] = []
    for repeat in range(args.repeats):
        order = list(args.columns)
        if args.rotate_order and order:
            shift = repeat % len(order)
            order = order[shift:] + order[:shift]
        orders.append(order)
        for sequence_index, columns in enumerate(order):
            expected_tiles = math.ceil(columns / (4096 // args.batch_size))
            result = run_tiled_benchmark(
                batch_size=args.batch_size, clusters=args.clusters, columns=columns,
            )
            tiles = result["dimensions"]["round2_tiles"]
            if tiles != expected_tiles:
                raise SystemExit(f"tile 映射不符：columns={columns} 得到 {tiles}，期望 {expected_tiles}")
            numerics = result["numerics"]
            if numerics["sign_consistency_rate"] < 1.0:
                raise SystemExit(f"数值一致性不达标：{numerics['sign_consistency_rate']} @ columns={columns}")
            row = {"columns": columns, "tiles": tiles, "repeat": repeat,
                   "sequence_index": sequence_index,
                   "timestamp": datetime.now().isoformat(timespec="seconds"),
                   "warmup": args.warmup,
                   "max_abs_error": numerics["max_abs_error"],
                   "p99_abs_error": numerics["p99_abs_error"],
                   "sign_consistency_rate": numerics["sign_consistency_rate"]}
            for stage in STAGES:
                row[stage] = result["timing_sec"][stage]
            row["online_ex_setup"] = row["online_total"] - row["context_and_public_key"]
            row["kernel_tile_avg"] = row["round2_b_kernel"] / tiles
            raw.append(row)
            by_size_online.setdefault(tiles, []).append(row["online_ex_setup"])
            by_size_kernel.setdefault(tiles, []).append(row["round2_b_kernel"])

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)

    summary_rows = []
    for stage in STAGES + ("online_ex_setup", "kernel_tile_avg"):
        for tiles in sorted({r["tiles"] for r in raw}):
            values = [r[stage] for r in raw if r["tiles"] == tiles]
            stats = summarize(values)
            summary_rows.append({"stage": stage, "tiles": tiles, **stats})
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    tiles_sorted = sorted(by_size_online)
    fit_kernel = fit_linear([t for t in tiles_sorted],
                            [summarize(by_size_kernel[t])["median"] for t in tiles_sorted])
    fit_online = fit_linear([t for t in tiles_sorted],
                            [summarize(by_size_online[t])["median"] for t in tiles_sorted])
    decision = decide(by_size_online, by_size_kernel)

    order_effect = {
        "pearson_sequence_vs_online": pearson(
            [float(r["sequence_index"]) for r in raw],
            [r["online_ex_setup"] for r in raw],
        ),
        "median_by_position": {
            pos: statistics.median([r["online_ex_setup"] for r in raw if r["sequence_index"] == pos])
            for pos in sorted({r["sequence_index"] for r in raw})
        },
        "note": "r 接近 0 说明未见明显顺序漂移；样本少时该值本身不稳定，只作提示",
    }

    payload = {"environment": env,
               "config": {"columns": args.columns, "clusters": args.clusters,
                          "batch_size": args.batch_size, "repeats": args.repeats,
                          "warmup": args.warmup, "rotate_order": args.rotate_order,
                          "measured_orders": orders},
               "definitions": {
                   "round2_b_kernel": "B 侧 kernel + tile 序列化（serialize_output=True）",
                   "online_total": "context 创建之前 → round2 解密结束；含 setup",
                   "online_ex_setup": "online_total - context_and_public_key；"
                                      "保留 query_encrypt、两轮序列化、两轮 kernel、两轮解密，"
                                      "扣除 context/keygen/公开 context 序列化",
               },
               "caveats": [
                   "round2_b_kernel 含 tile 序列化（serialize_output=True）",
                   "online_total 含 context 与 keygen；因此另报 online_ex_setup",
                   "拟合是有限点经验模型，不是运行规律",
               ],
               "fit_kernel": fit_kernel, "fit_online": fit_online,
               "order_effect": order_effect,
               "decision": decision}
    (out / "fit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'tiles':>6}{'kernel(med)':>13}{'online_ex_setup(med)':>22}{'n':>4}")
    for tiles in tiles_sorted:
        print(f"{tiles:>6}{summarize(by_size_kernel[tiles])['median']:>13.3f}"
              f"{summarize(by_size_online[tiles])['median']:>22.3f}{len(by_size_online[tiles]):>4}")
    print(f"\n判定: {decision['verdict']}\n理由: {decision['reason']}")
    print(f"\noutputs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
