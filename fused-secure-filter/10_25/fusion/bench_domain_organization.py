#!/usr/bin/env python3
"""融合假设检验：CKKS 成本取决于「候选如何组织」还是「填充后总量」？

## 假设

当前 kernel 每 tile 做 k·d 次密文×明文乘法，tile 数为 ceil(L/T)。
故总乘法数 ∝ (L/T)·k·d = (K·L)·d/T。

若该式成立，则**固定 K·L、改变 k 与 L 的搭配**时，耗时应当基本不变。
这意味着 Approx-PSI 无论怎样重新组织候选（分更多簇/更少簇、重排块），
只要填充后总量不变，CKKS 成本就不变。

## 设计

三组配置，K·L ≈ 24,150（= 当前 50×483）：
    (k=50,  L=483)  当前基线
    (k=100, L=242)
    (k=25,  L=966)

预测：三者耗时接近（差异 < 15%）。

    python fusion/bench_domain_organization.py --repeats 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BATCH = 200
CONFIGS = [(50, 483), (100, 242), (25, 966)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "domain_organization" / "report.json")
    args = ap.parse_args()

    try:
        import tenseal  # noqa: F401
    except ImportError:
        print("tenseal 未安装")
        return 2

    from scripts.benchmark_he_batching import run_tiled_benchmark

    print(f"{'k':>5}{'L':>7}{'K*L':>8}{'tiles':>7}{'kernel 中位数':>16}{'相对基线':>10}")
    results = []
    base = None
    for k, L in CONFIGS:
        times = []
        for _ in range(args.repeats):
            res = run_tiled_benchmark(batch_size=BATCH, clusters=k, columns=L)
            times.append(res["timing_sec"]["round2_b_kernel"])
        med = statistics.median(times)
        if base is None:
            base = med
        tiles = res["dimensions"]["round2_tiles"]
        print(f"{k:>5}{L:>7}{k*L:>8}{tiles:>7}{med:>15.2f}s{med/base:>10.3f}")
        results.append({"clusters": k, "columns": L, "padded_size": k * L,
                        "tiles": tiles, "kernel_median_s": med,
                        "relative_to_baseline": med / base, "runs": times})

    vals = [r["kernel_median_s"] for r in results]
    spread = (max(vals) - min(vals)) / min(vals)
    print(f"\n最大相对差异 = {spread*100:.1f}%  →  "
          f"{'支持「成本∝填充总量」' if spread < 0.15 else '不支持该预测'}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "batch": BATCH, "repeats": args.repeats, "configs": results,
        "spread": spread, "prediction_supported": spread < 0.15,
        "note": "固定 K*L=24,150，改变簇数与列宽的搭配。",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"outputs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
