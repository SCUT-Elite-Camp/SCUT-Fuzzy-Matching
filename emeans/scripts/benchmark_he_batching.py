"""Benchmark the production V2 query-by-candidate CKKS tiled protocol."""

# Project imports follow the direct-execution path bootstrap below.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ckks.context import create_ckks_context
from ckks.keys import serialize_public_context
from ckks.tiling import (
    decrypt_tiled_score_tile,
    encrypt_tiled_feature_batch,
    iter_tile_slices,
    make_slot_tile_layout,
    tile_count,
)
from config.params import SIMILARITY_THRESHOLD
from party_a.online_querier import choose_clusters_and_build_tiled_request
from party_b import online_responder
from party_b.online_responder import (
    compare_tiled_batch_to_centroids,
    tiled_batch_matching,
)
from protocol.transport import (
    serialize_tiled_first_round_request,
    serialize_tiled_second_round_request,
)
from protocol.types import TiledFirstRoundRequest, TiledPartyALocalState


def _serialized_size(items) -> int:
    return sum(
        len(item if isinstance(item, bytes) else item.serialize()) for item in items
    )


def run_tiled_benchmark(
    *,
    batch_size: int = 200,
    clusters: int = 50,
    columns: int = 483,
    tau: float = SIMILARITY_THRESHOLD,
    seed: int = 42,
) -> dict:
    """Run one complete, non-projected V2 HE pass with synthetic vectors."""
    if clusters <= 0 or columns <= 0:
        raise ValueError("clusters and columns must be positive")

    layout = make_slot_tile_layout(batch_size)
    rng = np.random.default_rng(seed)
    q200 = rng.uniform(-1.0, 1.0, size=(batch_size, 200))
    q50 = rng.uniform(-1.0, 1.0, size=(batch_size, 50))
    centroids = rng.uniform(-1.0, 1.0, size=(clusters, 200))
    cluster_matrix = rng.uniform(-1.0, 1.0, size=(clusters, columns, 50))

    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    start = time.perf_counter()
    secret_context = create_ckks_context()
    secret_context.generate_relin_keys()
    public_context_bytes = serialize_public_context(secret_context)
    timings["context_and_public_key"] = time.perf_counter() - start

    start = time.perf_counter()
    encrypted_q200 = encrypt_tiled_feature_batch(q200, secret_context, layout)
    encrypted_q50 = encrypt_tiled_feature_batch(q50, secret_context, layout)
    timings["query_encrypt"] = time.perf_counter() - start

    request1 = TiledFirstRoundRequest(
        public_context_bytes=public_context_bytes,
        encrypted_query_200=encrypted_q200,
        layout=layout,
    )
    state = TiledPartyALocalState(
        secret_context=secret_context,
        encrypted_query_50=encrypted_q50,
        layout=layout,
    )

    start = time.perf_counter()
    wire_request1 = serialize_tiled_first_round_request(request1)
    timings["round1_request_serialize"] = time.perf_counter() - start

    start = time.perf_counter()
    round1_tiles = compare_tiled_batch_to_centroids(
        wire_request1, centroids, serialize_output=True
    )
    timings["round1_b_kernel"] = time.perf_counter() - start

    start = time.perf_counter()
    request2, cluster_debug = choose_clusters_and_build_tiled_request(
        round1_tiles, state, k=clusters
    )
    timings["round1_a_decrypt_and_selector_encrypt"] = time.perf_counter() - start

    start = time.perf_counter()
    wire_request2 = serialize_tiled_second_round_request(request2)
    timings["round2_request_serialize"] = time.perf_counter() - start

    captured_masks: list[np.ndarray] = []
    original_mask_sampler = online_responder._sample_positive_mask_matrix

    def record_mask(m: int, t: int) -> np.ndarray:
        mask = original_mask_sampler(m, t)
        captured_masks.append(mask.copy())
        return mask

    start = time.perf_counter()
    with patch(
        "party_b.online_responder._sample_positive_mask_matrix",
        side_effect=record_mask,
    ):
        round2_tiles = list(
            tiled_batch_matching(
                cluster_matrix,
                wire_request2,
                public_context_bytes,
                tau=tau,
                serialize_output=True,
            )
        )
    timings["round2_b_kernel"] = time.perf_counter() - start

    start = time.perf_counter()
    decrypted_round2 = []
    for encrypted_tile, (_, _, valid_width) in zip(
        round2_tiles, iter_tile_slices(columns, layout), strict=True
    ):
        decrypted_round2.append(
            decrypt_tiled_score_tile(
                encrypted_tile, secret_context, layout, valid_width
            )
        )
    scores = np.concatenate(decrypted_round2, axis=1)
    timings["round2_a_decrypt"] = time.perf_counter() - start
    timings["online_total"] = time.perf_counter() - total_start

    selected = cluster_debug.selected_clusters
    expected_tiles = []
    for mask, (_, start_col, valid_width) in zip(
        captured_masks, iter_tile_slices(columns, layout), strict=True
    ):
        candidate = cluster_matrix[
            selected[:, None],
            np.arange(start_col, start_col + valid_width)[None, :],
            :,
        ]
        margin = np.einsum("qd,qtd->qt", q50, candidate) - tau
        expected_tiles.append(mask[:, :valid_width] * margin)
    expected = np.concatenate(expected_tiles, axis=1)
    abs_error = np.abs(scores - expected)
    clear_zone = np.abs(expected) >= 5e-4
    sign_consistency = (
        float(np.mean(np.sign(scores[clear_zone]) == np.sign(expected[clear_zone])))
        if np.any(clear_zone)
        else 1.0
    )

    expected_r1_tiles = tile_count(clusters, layout)
    expected_r2_tiles = tile_count(columns, layout)
    if len(round1_tiles) != expected_r1_tiles or len(round2_tiles) != expected_r2_tiles:
        raise AssertionError("tiled kernel emitted an unexpected ciphertext count")

    sent_bytes = (
        len(public_context_bytes)
        + _serialized_size(wire_request1.encrypted_query_200)
        + _serialized_size(wire_request2.encrypted_query_50)
        + _serialized_size(wire_request2.encrypted_selectors)
    )
    received_bytes = _serialized_size(round1_tiles) + _serialized_size(round2_tiles)

    return {
        "dimensions": {
            "batch_size": batch_size,
            "clusters": clusters,
            "columns": columns,
            "tile_width": layout.tile_width,
            "active_slots": layout.active_slots,
            "round1_tiles": len(round1_tiles),
            "round2_tiles": len(round2_tiles),
        },
        "timing_sec": timings,
        "communication_bytes": {
            "sent": sent_bytes,
            "received": received_bytes,
            "total": sent_bytes + received_bytes,
        },
        "numerics": {
            "max_abs_error": float(np.max(abs_error)),
            "p99_abs_error": float(np.percentile(abs_error, 99)),
            "sign_consistency_rate": sign_consistency,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a complete V2 query-by-candidate CKKS tiled benchmark."
    )
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--clusters", type=int, default=50)
    parser.add_argument("--columns", type=int, default=483)
    parser.add_argument("--tau", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "artifacts" / "benchmark_he_batching_v2.json"),
    )
    args = parser.parse_args()

    result = run_tiled_benchmark(
        batch_size=args.batch_size,
        clusters=args.clusters,
        columns=args.columns,
        tau=args.tau,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    dims = result["dimensions"]
    timings = result["timing_sec"]
    numerics = result["numerics"]
    print(json.dumps(result, indent=2))
    print(
        f"V2 complete: m={dims['batch_size']}, T={dims['tile_width']}, "
        f"R1={dims['round1_tiles']} tiles, R2={dims['round2_tiles']} tiles, "
        f"online={timings['online_total']:.3f}s, "
        f"max_error={numerics['max_abs_error']:.3e}"
    )
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
