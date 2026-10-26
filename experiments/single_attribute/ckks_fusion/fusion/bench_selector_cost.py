#!/usr/bin/env python3
"""量化"隐藏选中簇"的成本：加密 selector vs 公开簇（仅作测量，不提议泄露）。

两个版本在数学上完全等价——one-hot selector 选择的就是那条记录。
差别只在于 B 是否知道"选了哪个簇"。因此这个比值就是该隐私属性的价格。

    python fusion/bench_selector_cost.py --columns 20 --repeats 3

输出：两版的中位耗时、比值、以及输出一致性（max|diff| 与符号一致率）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

M, K, D = 200, 50, 50


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--columns", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "kernel_v2" / "selector_cost.json")
    args = ap.parse_args()

    try:
        import tenseal as ts
    except ImportError:
        print("tenseal 未安装：无法测量。")
        return 2

    from ckks.context import create_ckks_context
    from ckks.keys import serialize_public_context
    from ckks.tiling import (
        decrypt_tiled_score_tile, encrypt_tiled_feature_batch, iter_tile_slices,
        make_slot_tile_layout, pack_candidate_tile,
    )
    from config.params import RANDOM_MASK_MAX, RANDOM_MASK_MIN, SIMILARITY_THRESHOLD
    from party_a.online_querier import choose_clusters_and_build_tiled_request
    from party_b.online_responder import compare_tiled_batch_to_centroids, tiled_batch_matching
    from protocol.transport import (
        serialize_tiled_first_round_request, serialize_tiled_second_round_request,
    )
    from protocol.types import TiledFirstRoundRequest, TiledPartyALocalState

    layout = make_slot_tile_layout(M)
    rng = np.random.default_rng(args.seed)
    q200, q50 = rng.uniform(-1, 1, (M, 200)), rng.uniform(-1, 1, (M, 50))
    cent, mat = rng.uniform(-1, 1, (K, 200)), rng.uniform(-1, 1, (K, args.columns, 50))

    ctx = create_ckks_context()
    ctx.generate_relin_keys()
    pub = serialize_public_context(ctx)
    state = TiledPartyALocalState(
        secret_context=ctx, encrypted_query_50=encrypt_tiled_feature_batch(q50, ctx, layout), layout=layout)
    wire1 = serialize_tiled_first_round_request(TiledFirstRoundRequest(
        public_context_bytes=pub,
        encrypted_query_200=encrypt_tiled_feature_batch(q200, ctx, layout), layout=layout))
    scores = compare_tiled_batch_to_centroids(wire1, cent, serialize_output=True)
    req2, debug = choose_clusters_and_build_tiled_request(scores, state, k=K)
    wire2 = serialize_tiled_second_round_request(req2)
    selected = debug.selected_clusters

    def sampler_factory(seed: int = 777):
        s = {"rng": np.random.default_rng(seed)}

        def draw(m: int, t: int) -> np.ndarray:
            return s["rng"].uniform(RANDOM_MASK_MIN, RANDOM_MASK_MAX, size=(m, t))

        def reset() -> None:
            s["rng"] = np.random.default_rng(seed)

        return draw, reset

    sampler, reset = sampler_factory()

    def kernel_public_cluster(matrix, request, public_context, sel, tau=SIMILARITY_THRESHOLD):
        import party_b.online_responder as R
        c = R._load_tiled_public_context(public_context)
        eq = R._load_tiled_ciphertexts(request.encrypted_query_50, c, feature_count=50, layout=layout)
        m, T = layout.batch_size, layout.tile_width
        for _, start, vw in iter_tile_slices(matrix.shape[1], layout):
            Rm = R._sample_positive_mask_matrix(m, T)
            Rm[:, vw:] = 1.0
            Bt = pack_candidate_tile(matrix, layout, start, vw)
            rows = sel[:, None]
            acc = None
            for f in range(D):
                cand = Bt[rows, np.arange(T)[None, :], f] * Rm
                term = eq[f] * ts.plain_tensor(cand.reshape(-1), dtype="float")
                acc = term if acc is None else acc + term
            yield acc + ts.plain_tensor((Rm.reshape(-1) * -tau).astype(np.float64), dtype="float")

    def run_encrypted():
        reset()
        return list(tiled_batch_matching(mat, wire2, pub, tau=SIMILARITY_THRESHOLD, serialize_output=True))

    def run_public():
        reset()
        return list(kernel_public_cluster(mat, wire2, pub, selected, tau=SIMILARITY_THRESHOLD))

    def decrypt(tiles):
        return np.concatenate([
            decrypt_tiled_score_tile(b, ctx, layout, w)
            for b, (_, _, w) in zip(tiles, iter_tile_slices(args.columns, layout))
        ], axis=1)

    with patch.object(sys.modules["party_b.online_responder"],
                      "_sample_positive_mask_matrix", side_effect=sampler):
        t_enc, t_pub = [], []
        for _ in range(args.repeats):
            t0 = time.perf_counter(); a = run_encrypted(); t_enc.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); b = run_public(); t_pub.append(time.perf_counter() - t0)
        da, db = decrypt(a), decrypt(b)

    me, mp = statistics.median(t_enc), statistics.median(t_pub)
    payload = {
        "config": {"columns": args.columns, "repeats": args.repeats, "m": M, "k": K, "d": D},
        "encrypted_selector_median_s": me,
        "public_cluster_median_s": mp,
        "ratio": me / mp,
        "numerics": {"max_abs_diff": float(np.max(np.abs(da - db))),
                     "sign_agreement": float(np.mean(np.sign(da) == np.sign(db)))},
        "note": "公开簇版本等价于把「选了哪一簇」告诉 B；仅用于测量该隐私属性的成本，不是可用方案。",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"columns={args.columns} tiles={len(a)} repeats={args.repeats}")
    print(f"加密 selector = {me:.3f}s   公开簇 = {mp:.3f}s   比值 = {me/mp:.2f}×")
    print(f"输出 max|diff|={payload['numerics']['max_abs_diff']:.3e}  "
          f"符号一致={payload['numerics']['sign_agreement']:.4f}")
    print(f"outputs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
