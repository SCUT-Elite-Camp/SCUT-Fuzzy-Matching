#!/usr/bin/env python3
"""单 tile 时间构成拆解：明文准备 / selector 乘法 / 密文加法 / 点积 / 序列化。"""
from __future__ import annotations
import sys, time, statistics
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
M, K, D = 200, 50, 50
import argparse

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--d", type=int, default=50)
    args = ap.parse_args()
    import tenseal as ts
    from ckks.context import create_ckks_context
    from ckks.tiling import iter_tile_slices, make_slot_tile_layout, pack_candidate_tile
    import party_b.online_responder as R
    from config.params import RANDOM_MASK_MAX, RANDOM_MASK_MIN

    ctx = create_ckks_context()
    layout = make_slot_tile_layout(M)
    T = layout.tile_width
    matrix = np.random.default_rng(0).uniform(-1, 1, (K, 483, D))
    enc_sel = [ts.ckks_vector(ctx, np.random.rand(M * T).tolist()) for _ in range(K)]
    enc_q = [ts.ckks_vector(ctx, np.random.rand(M * T).tolist()) for _ in range(args.d)]
    _, start, vw = next(iter(iter_tile_slices(483, layout)))
    rng = np.random.default_rng(1)

    t_prep = t_mul = t_add = t_dot = t_ser = 0.0
    for _ in range(3):
        s0 = time.perf_counter()
        Rm = rng.uniform(RANDOM_MASK_MIN, RANDOM_MASK_MAX, size=(M, T)); Rm[:, vw:] = 1.0
        B_tile = pack_candidate_tile(matrix, layout, start, vw)
        s1 = time.perf_counter(); t_prep += s1 - s0

        sel_feats = []
        for f in range(args.d):
            acc = None
            for i in range(K):
                plain = (B_tile[i, :, f] * Rm).reshape(-1)
                pt = ts.plain_tensor(plain, dtype="float")
                s2 = time.perf_counter(); term = enc_sel[i] * pt; s3 = time.perf_counter()
                t_mul += s3 - s2
                if acc is None:
                    acc = term
                else:
                    s4 = time.perf_counter(); acc += term; t_add += time.perf_counter() - s4
            sel_feats.append(acc)
        s5 = time.perf_counter()
        score = enc_q[0] * sel_feats[0]
        for f in range(1, args.d):
            score += enc_q[f] * sel_feats[f]
        s6 = time.perf_counter(); t_dot += s6 - s5
        score.serialize()
        t_ser += time.perf_counter() - s6

    tot = t_prep + t_mul + t_add + t_dot + t_ser
    print(f"单 tile 合计 {tot/3:.3f}s（3 次平均）")
    for name, v in (("明文准备(掩码+打包)", t_prep), ("selector ct-pt 乘 (2,500)", t_mul),
                    ("selector 累加 (2,450)", t_add), ("点积 ct-ct (50)", t_dot),
                    ("序列化", t_ser)):
        print(f"  {name:<26}{v/3:>8.3f}s{ v/tot*100:>7.1f}%")
    print(f"\n单次 selector 乘法 = {t_mul/(3*K*D)*1000:.3f} ms")
    print(f"单次点积 ct-ct    = {t_dot/(3*args.d)*1000:.3f} ms")
    print(f"单次密文加法      = {t_add/(3*D*(K-1))*1000:.3f} ms")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
