#!/usr/bin/env python3
"""基线 kernel 与 kernel_v2 的对照：数值一致性与真实耗时。

两版使用**同一组 mask**（通过 patch 采样器 + 每次运行前重置种子），
因此差异只来自明文操作数的传递方式。

    python fusion/bench_kernel_v2.py --columns 20  --repeats 3
    python fusion/bench_kernel_v2.py --columns 483 --repeats 3
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

from config.params import RANDOM_MASK_MAX, RANDOM_MASK_MIN, SIMILARITY_THRESHOLD  # noqa: E402

M, K, D = 200, 50, 50


def build_case(columns: int, seed: int = 42):
    import tenseal as ts
    from ckks.context import create_ckks_context
    from ckks.keys import serialize_public_context
    from ckks.tiling import make_slot_tile_layout, encrypt_tiled_feature_batch
    from party_a.online_querier import choose_clusters_and_build_tiled_request
    from party_b.online_responder import compare_tiled_batch_to_centroids
    from protocol.transport import (
        serialize_tiled_first_round_request,
        serialize_tiled_second_round_request,
    )
    from protocol.types import TiledFirstRoundRequest, TiledPartyALocalState

    layout = make_slot_tile_layout(M)
    rng = np.random.default_rng(seed)
    q200 = rng.uniform(-1, 1, size=(M, 200))
    q50 = rng.uniform(-1, 1, size=(M, 50))
    centroids = rng.uniform(-1, 1, size=(K, 200))
    cluster_matrix = rng.uniform(-1, 1, size=(K, columns, 50))

    ctx = create_ckks_context()
    ctx.generate_relin_keys()
    pub = serialize_public_context(ctx)
    enc200 = encrypt_tiled_feature_batch(q200, ctx, layout)
    enc50 = encrypt_tiled_feature_batch(q50, ctx, layout)
    state = TiledPartyALocalState(secret_context=ctx, encrypted_query_50=enc50, layout=layout)
    req1 = TiledFirstRoundRequest(public_context_bytes=pub, encrypted_query_200=enc200, layout=layout)
    wire1 = serialize_tiled_first_round_request(req1)
    scores = compare_tiled_batch_to_centroids(wire1, centroids, serialize_output=True)
    req2, _ = choose_clusters_and_build_tiled_request(scores, state, k=K)
    wire2 = serialize_tiled_second_round_request(req2)
    return ctx, pub, cluster_matrix, wire2


def make_sampler(seed: int = 777):
    state = {"rng": np.random.default_rng(seed)}

    def sampler(m: int, t: int) -> np.ndarray:
        return state["rng"].uniform(RANDOM_MASK_MIN, RANDOM_MASK_MAX, size=(m, t))

    def reset() -> None:
        state["rng"] = np.random.default_rng(seed)

    return sampler, reset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--columns", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "kernel_v2" / "compare.json")
    args = ap.parse_args()

    try:
        import tenseal as ts
    except ImportError:
        print("tenseal 未安装：无法测量。")
        return 2

    from party_b.online_responder import tiled_batch_matching
    from fusion.kernel_v2 import tiled_batch_matching_v2

    ctx, pub, matrix, req2 = build_case(args.columns)
    sampler, reset = make_sampler()

    def run(fn):
        reset()
        return list(fn(matrix, req2, pub, tau=SIMILARITY_THRESHOLD, serialize_output=True))

    def decrypt(tiles):
        from ckks.tiling import decrypt_tiled_score_tile, iter_tile_slices, make_slot_tile_layout
        layout = make_slot_tile_layout(M)
        out = [decrypt_tiled_score_tile(b, ctx, layout, w)
               for b, (_, _, w) in zip(tiles, iter_tile_slices(args.columns, layout))]
        return np.concatenate(out, axis=1)

    with patch.object(sys.modules["party_b.online_responder"],
                      "_sample_positive_mask_matrix", side_effect=sampler):
        t0 = time.perf_counter(); base_tiles = run(tiled_batch_matching); t_base0 = time.perf_counter() - t0
        t0 = time.perf_counter(); v2_tiles = run(tiled_batch_matching_v2); t_v2_0 = time.perf_counter() - t0
        base_scores, v2_scores = decrypt(base_tiles), decrypt(v2_tiles)

        max_diff = float(np.max(np.abs(base_scores - v2_scores)))
        same_sign = float(np.mean(np.sign(base_scores) == np.sign(v2_scores)))

        times_base, times_v2 = [t_base0], [t_v2_0]
        for _ in range(args.repeats):
            t0 = time.perf_counter(); run(tiled_batch_matching); times_base.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); run(tiled_batch_matching_v2); times_v2.append(time.perf_counter() - t0)

    mb, mv = statistics.median(times_base), statistics.median(times_v2)
    payload = {
        "config": {"columns": args.columns, "repeats": args.repeats + 1, "m": M, "k": K, "d": D},
        "numerics": {"max_abs_diff": max_diff, "sign_agreement": same_sign,
                     "tiles": [len(base_tiles), len(v2_tiles)]},
        "timing": {"baseline_median_s": mb, "v2_median_s": mv,
                   "speedup": mb / mv, "baseline_all": times_base, "v2_all": times_v2},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"columns={args.columns}  tiles={len(base_tiles)}  repeats={args.repeats + 1}")
    print(f"数值: max|diff|={max_diff:.3e}  符号一致率={same_sign:.4f}")
    print(f"基线 中位数={mb:.3f}s    v2 中位数={mv:.3f}s    加速={mb/mv:.3f}×")
    print(f"outputs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
