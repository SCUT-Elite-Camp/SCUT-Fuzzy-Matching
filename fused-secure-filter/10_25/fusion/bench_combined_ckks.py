#!/usr/bin/env python3
"""组合对照：原 kernel / kernel_v2 / kernel_v2+截断，同一 session 交错测量。

三组：
  A  原 kernel，c=483（25 tiles）—— 基线
  B  kernel_v2， c=483（25 tiles）—— 实现优化单独收益
  C  kernel_v2， c=400（20 tiles）—— 实现优化 × 截断（组合）

同 mask、同输入、交错运行。C 与 A 在前 400 列上逐槽位比对符号。
"""

from __future__ import annotations
import json, statistics, sys, time
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
M = 200
C_FULL, C_TRUNC = 483, 400

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", type=Path, default=ROOT/"artifacts"/"combined_ckks"/"report.json")
    args = ap.parse_args()

    import tenseal as ts  # noqa: F401
    from fusion.bench_kernel_v2 import build_case, make_sampler
    from fusion.kernel_v2 import tiled_batch_matching_v2
    from party_b.online_responder import tiled_batch_matching
    from ckks.tiling import decrypt_tiled_score_tile, iter_tile_slices, make_slot_tile_layout
    from config.params import SIMILARITY_THRESHOLD

    ctx, pub, matrix, req2 = build_case(C_FULL)
    layout = make_slot_tile_layout(M)
    sampler, reset = make_sampler()

    def run(kernel, cols):
        reset()
        sub = matrix[:, :cols, :]
        t0 = time.perf_counter()
        tiles = list(kernel(sub, req2, pub, tau=SIMILARITY_THRESHOLD, serialize_output=True))
        dt = time.perf_counter() - t0
        dec = np.concatenate([decrypt_tiled_score_tile(b, ctx, layout, w)
                              for b, (_, _, w) in zip(tiles, iter_tile_slices(cols, layout))], axis=1)
        return len(tiles), dt, dec

    with patch.object(sys.modules["party_b.online_responder"],
                      "_sample_positive_mask_matrix", side_effect=sampler):
        runs = {"A": [], "B": [], "C": []}
        decA = decC = None
        for r in range(args.repeats):
            order = ["A", "B", "C"] if r % 2 == 0 else ["C", "B", "A"]
            for g in order:
                if g == "A": t, dt, d = run(tiled_batch_matching, C_FULL); decA = d
                elif g == "B": t, dt, d = run(tiled_batch_matching_v2, C_FULL)
                else: t, dt, d = run(tiled_batch_matching_v2, C_TRUNC); decC = d
                runs[g].append({"tiles": t, "seconds": dt})

    med = {g: statistics.median([x["seconds"] for x in v]) for g, v in runs.items()}
    tiles = {g: runs[g][0]["tiles"] for g in runs}
    sign_mis = int(np.sum(np.sign(decA[:, :C_TRUNC]) != np.sign(decC)))
    print(f"{'组':<4}{'tiles':>7}{'selector乘':>12}{'耗时中位数':>12}{'相对A':>8}")
    for g, name in (("A","原kernel"),("B","v2"),("C","v2+截断")):
        print(f"{g:<4}{tiles[g]:>7}{tiles[g]*50*50:>12,}{med[g]:>11.2f}s{med[g]/med['A']:>8.3f}")
    print(f"\nv2 单独:      {med['A']/med['B']:.3f}×")
    print(f"组合 C/A:     {med['A']/med['C']:.3f}×")
    print(f"前 {C_TRUNC} 列符号不一致: {sign_mis} / {decA[:, :C_TRUNC].size}")

    import platform, tenseal
    env = {"python": platform.python_version(), "tenseal": tenseal.__version__,
           "numpy": np.__version__, "machine": platform.machine()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {"m": M, "k": 50, "d": 50, "c_full": C_FULL, "c_trunc": C_TRUNC, "repeats": args.repeats},
        "runs": runs,
        "median_seconds": med,
        "speedup": {"v2_alone": med["A"]/med["B"], "combined": med["A"]/med["C"]},
        "tiles": tiles,
        "sign_mismatch_on_common_cols": sign_mis,
        "note": "同一 session、交错顺序、固定 mask；C 与 A 仅在前 400 列可比",
        "environment": env,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"outputs -> {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
