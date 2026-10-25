#!/usr/bin/env python3
"""R3 数值探针：验证 XOR 份额在 CKKS 里能否可靠重构为 0/1 标志。

只做数值与成本测量，不接任何协议流程，不改生产基线。

形状取自生产布局：m=200 查询 × max_size=483 候选 = 96,600 个标志；
tile 宽度 4000（= m × T，T=4096//200=20），因此需要 25 个 tile。

每个 tile：
    Enc(z) = Enc(a) + Enc(b) - 2 · Enc(a)·Enc(b)
    z = a XOR b  （a,b ∈ {0,1}，与 secure_approx_psi.cpp:523 的 flag_a ^ flag_b 一致）

判据：解密后四舍五入到 0/1，与真实 XOR 逐位比较。
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ckks.context import create_ckks_context  # noqa: E402
from config.params import COEFF_MOD_BIT_SIZES, POLY_MODULUS_DEGREE, SCALE  # noqa: E402

M, MAX_SIZE, T = 200, 483, 4096 // 200
SLOTS = M * T
TILES = int(np.ceil(M * MAX_SIZE / SLOTS))


def main() -> int:
    try:
        import tenseal as ts
    except ImportError:
        print("tenseal 未安装：无法做 CKKS 实测。")
        return 2

    rng = np.random.default_rng(2026)
    total_bits = M * MAX_SIZE
    a_bits = rng.integers(0, 2, size=TILES * SLOTS).astype(np.float64)
    b_bits = rng.integers(0, 2, size=TILES * SLOTS).astype(np.float64)
    # 让最后一点超出 total_bits 的槽位也有确定值（不做统计）
    truth = np.logical_xor(a_bits, b_bits).astype(np.float64)

    ctx = create_ckks_context()
    per_tile = []
    errs = []
    wrong = 0

    for tile in range(TILES):
        sl = slice(tile * SLOTS, (tile + 1) * SLOTS)
        a = a_bits[sl].tolist()
        b = b_bits[sl].tolist()
        t0 = time.perf_counter()
        ea = ts.ckks_vector(ctx, a)
        eb = ts.ckks_vector(ctx, b)
        # z = a + b - 2ab ；Enc(a)·Enc(b) 是一次密文-密文乘法（1 层深度 + relin + rescale）
        ez = ea + eb - (ea * eb) * 2.0
        z = np.array(ez.decrypt())
        dt = time.perf_counter() - t0
        rounded = np.rint(z)
        err = np.abs(z - rounded)
        valid = min((tile + 1) * SLOTS, total_bits) - tile * SLOTS
        wrong += int(np.sum(rounded[:valid] != truth[sl][:valid]))
        errs.append(float(err[:valid].max()))
        per_tile.append(dt)
        print(f"  tile {tile+1:2d}/{TILES}: max|err|={errs[-1]:.3e}  {dt:.3f}s")

    valid_total = total_bits
    payload = {
        "config": {"poly_modulus_degree": POLY_MODULUS_DEGREE,
                   "coeff_mod_bit_sizes": COEFF_MOD_BIT_SIZES,
                   "scale_bits": SCALE.bit_length() - 1,
                   "m": M, "max_size": MAX_SIZE, "tile_width_T": T,
                   "slots_per_tile": SLOTS, "tiles": TILES},
        "formula": "Enc(z) = Enc(a) + Enc(b) - 2*Enc(a)*Enc(b)",
        "result": {
            "bits_tested": int(valid_total),
            "bit_errors": wrong,
            "bit_accuracy": 1.0 - wrong / valid_total,
            "max_abs_error": max(errs),
            "median_tile_seconds": statistics.median(per_tile),
            "total_seconds": sum(per_tile),
            "ct_ct_mults": TILES,
        },
        "note": "仅重构 PSI 碰撞位；电路里的 valid 与 ham_ok 两道门没有被重构。",
    }
    print(f"\nbits={valid_total} errors={wrong} accuracy={payload['result']['bit_accuracy']:.6f}")
    print(f"max|err|={payload['result']['max_abs_error']:.3e}  "
          f"median/tile={payload['result']['median_tile_seconds']:.3f}s  "
          f"total={payload['result']['total_seconds']:.2f}s  ct_ct={TILES}")

    out = ROOT / "artifacts" / "r3_probe" / "numeric.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"outputs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
