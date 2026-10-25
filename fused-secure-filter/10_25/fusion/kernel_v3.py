"""计算图重写实验：把随机掩码 R 从内层提到外层。

## 代数推导

当前 kernel（每 tile）：
    selected_f = Σ_i  Enc(sel_i) ⊙ (R ⊙ B[i,:,f])
    score      = Σ_f  Enc(q_f) ⊙ selected_f − τ·R

因为 R 是**明文**且与 i,f 无关，可以提出来：
    Σ_f Enc(q_f) ⊙ ( Σ_i Enc(sel_i) ⊙ (R ⊙ B[i,:,f]) )
  = R ⊙ Σ_f Enc(q_f) ⊙ ( Σ_i Enc(sel_i) ⊙ B[i,:,f] )
  = R ⊙ ( Σ_f Enc(q_f) ⊙ selected'_f )

故 score = R ⊙ (sim − τ)，其中 sim = Σ_f Enc(q_f) ⊙ selected'_f，selected'_f 不含 R。

## 代价变化

- 密文-明文乘法：仍然 k·d 次（内层），另有 1 次用于外层 R（新增 1 次/tile）
- 明文 NumPy 运算：从 k·d 次 4000 元素乘法降为 1 次
- 输出：逐位等价的**正比关系**不是逐位相等：score_v3 = R·(sim−τ)，score_v2 = Σ_f q·(R·B) − τ·R
  两者在实数上相等，密文层因运算顺序不同会有浮点差异，需实测确认。

    python fusion/kernel_v3.py --columns 20 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

M, K, D = 200, 50, 50


def tiled_batch_matching_v3(cluster_matrix, request, public_context, tau, *, serialize_output=False):
    import tenseal as ts
    import party_b.online_responder as R
    from ckks.tiling import iter_tile_slices, pack_candidate_tile

    matrix = np.asarray(cluster_matrix, dtype=np.float64)
    k, L, d = matrix.shape
    layout = request.layout
    m, T = layout.batch_size, layout.tile_width
    ctx = R._load_tiled_public_context(public_context)
    eq = R._load_tiled_ciphertexts(request.encrypted_query_50, ctx, feature_count=50, layout=layout)
    enc_sel = R._load_tiled_ciphertexts(request.encrypted_selectors, ctx, feature_count=k, layout=layout)

    for _, start, vw in iter_tile_slices(L, layout):
        Rm = R._sample_positive_mask_matrix(m, T)
        Rm[:, vw:] = 1.0
        B_tile = pack_candidate_tile(matrix, layout, start, vw)
        sim = None
        for f in range(d):
            acc = None
            for i in range(k):
                # 不再乘 R：plain 只是 B 的切片广播
                plain = np.repeat(B_tile[i, :, f][None, :], m, axis=0).reshape(-1)
                term = enc_sel[i] * ts.plain_tensor(plain, dtype="float")
                acc = term if acc is None else acc + term
            term_f = eq[f] * acc
            sim = term_f if sim is None else sim + term_f
        # 外层一次：R ⊙ (sim − τ)
        score = sim + ts.plain_tensor((-tau * np.ones(1)).repeat(1)[0:1].tolist() * (m * T), dtype="float")
        score = score * ts.plain_tensor(Rm.reshape(-1), dtype="float")
        yield score.serialize() if serialize_output else score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--columns", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "kernel_v3" / "compare.json")
    args = ap.parse_args()

    try:
        import tenseal as ts
    except ImportError:
        print("tenseal 未安装")
        return 2

    from fusion.bench_kernel_v2 import build_case, make_sampler
    from fusion.kernel_v2 import tiled_batch_matching_v2
    from ckks.tiling import decrypt_tiled_score_tile, iter_tile_slices, make_slot_tile_layout
    from config.params import SIMILARITY_THRESHOLD
    from unittest.mock import patch

    ctx, pub, matrix, req2 = build_case(args.columns)
    sampler, reset = make_sampler()
    layout = make_slot_tile_layout(M)

    def run(fn, **kw):
        reset()
        return list(fn(matrix, req2, pub, tau=SIMILARITY_THRESHOLD, serialize_output=True, **kw))

    def decrypt(tiles):
        return np.concatenate([
            decrypt_tiled_score_tile(b, ctx, layout, w)
            for b, (_, _, w) in zip(tiles, iter_tile_slices(args.columns, layout))
        ], axis=1)

    with patch.object(sys.modules["party_b.online_responder"],
                      "_sample_positive_mask_matrix", side_effect=sampler):
        t2, t3 = [], []
        for _ in range(args.repeats):
            s = time.perf_counter(); a = run(tiled_batch_matching_v2); t2.append(time.perf_counter() - s)
            s = time.perf_counter(); b = run(tiled_batch_matching_v3); t3.append(time.perf_counter() - s)
        da, db = decrypt(a), decrypt(b)

    # 两者都等于 R·(sim − τ)，但运算顺序不同；比较前先检查数值
    diff = float(np.max(np.abs(da - db)))
    scale = float(np.max(np.abs(da))) or 1.0
    sign_agree = float(np.mean(np.sign(da) == np.sign(db)))
    payload = {
        "columns": args.columns, "repeats": args.repeats,
        "v2_median_s": statistics.median(t2), "v3_median_s": statistics.median(t3),
        "v2_all": t2, "v3_all": t3,
        "max_abs_diff": diff, "relative_diff": diff / scale, "sign_agreement": sign_agree,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"tiles={len(a)}  v2={statistics.median(t2):.3f}s  v3={statistics.median(t3):.3f}s  "
          f"比值={statistics.median(t2)/statistics.median(t3):.3f}×")
    print(f"max|diff|={diff:.3e} (相对 {diff/scale:.2e})  符号一致={sign_agree:.4f}")
    print(f"outputs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
