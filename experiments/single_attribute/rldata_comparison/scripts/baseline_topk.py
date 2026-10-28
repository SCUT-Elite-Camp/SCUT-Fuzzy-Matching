"""Scheme 1 (KMeans baseline) Top-k retrieval on RLdata10000.

Top-k follows the project's cluster-probe semantics (``--probes`` in the
mainline): round 1 decrypts the centroid scores, keeps the ``k`` most similar
clusters, and round 2 matches inside those clusters.  The probes are scored in
descending centroid similarity and the client stops at the first probe whose
best candidate clears the acceptance floor (``cosine > tau``), returning that
one identity.  Everything stays in ciphertext: the expanded batch of
``queries x k`` probes is encrypted and matched by Party B in one round, and
only Party A decrypts the score tiles.

Because the probes are batched, the garbled evaluation still computes every
probe; "early stop" is a selection rule on the decrypted scores, not a saving
in ciphertext work.  Round-2 cost, request size and traffic therefore grow with
``k``.

Evaluation is the Top-1-style query-level confusion matrix over the single
returned identity: a positive is correct when it equals the expected identity,
a negative is a false positive when any identity is returned.  ``k = 1`` is the
original Top-1 protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
ROOT = EXPERIMENT.parents[2]
BASELINE = ROOT / "baseline_Integration"
for entry in (HERE, ROOT, BASELINE):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import config.params as params  # noqa: E402
from ckks.keys import serialize_public_context  # noqa: E402
from ckks.tiling import decrypt_tiled_score_tile, iter_tile_slices  # noqa: E402
from party_a.local_prep import (  # noqa: E402
    create_ckks_context,
    prepare_tiled_query_batch,
)
from party_a.online_querier import choose_clusters_and_build_tiled_request  # noqa: E402
from party_b.online_responder import (  # noqa: E402
    compare_tiled_batch_to_centroids,
    tiled_batch_matching,
)
from protocol.transport import (  # noqa: E402
    serialize_tiled_first_round_request,
    serialize_tiled_second_round_request,
)

from baseline_top1 import build_offline, RssSampler, set_shingle_size  # noqa: E402
from names import normalize_name  # noqa: E402

DEFAULT_DATA = EXPERIMENT / "data"


def load_data(data_dir: Path, queries_file: Path):
    database = pd.read_csv(data_dir / "database.csv")
    database["canonical"] = database["normalized_name"].fillna("").map(normalize_name)
    database["identity"] = database["identity"].astype(int)
    database["dob"] = database["dob"].fillna("")
    queries = pd.read_csv(queries_file)
    queries["canonical"] = queries["normalized_name"].fillna("").map(normalize_name)
    queries["label_bool"] = queries["label"].astype(str).str.lower() == "true"
    queries["expected_identity"] = queries["expected_identity"].fillna(-1).astype(int)
    return database, queries


def metrics_from_single(
    labels: np.ndarray, expected: np.ndarray, predicted: np.ndarray
) -> dict:
    tp = int(np.sum(labels & (predicted == expected)))
    fp = int(np.sum(~labels & (predicted != -1)))
    fn = int(np.sum(labels & (predicted != expected)))
    tn = int(np.sum(~labels & (predicted == -1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(labels),
    }


def probe_predictions(
    scores: np.ndarray,
    *,
    probe_count: int,
    eps: float,
    probed_clusters: np.ndarray,
    assignments: np.ndarray,
    identities: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """Early-stop rule over the probes of each query.

    ``scores`` has one row per (query, probe) in query-major order.  Probes are
    given in descending centroid similarity; the first probe whose best
    candidate exceeds ``eps`` decides the answer.
    """
    m = scores.shape[0] // probe_count
    predicted = np.full(m, -1, dtype=np.int64)
    details: list[dict] = []
    for row in range(m):
        chosen = -1
        used = 0
        for probe in range(probe_count):
            used += 1
            cluster = int(probed_clusters[row * probe_count + probe])
            members = np.where(assignments == cluster)[0]
            if len(members) == 0:
                continue
            cluster_scores = scores[row * probe_count + probe, : len(members)]
            best = int(np.argmax(cluster_scores))
            if float(cluster_scores[best]) > eps:
                chosen = int(identities[members[best]])
                break
        predicted[row] = chosen
        details.append({"probes_used": used, "identity": chosen})
    return predicted, details


def run_he_topk(
    database: pd.DataFrame,
    queries: pd.DataFrame,
    artifacts,
    *,
    tau: float,
    shingle_size: int,
    top_k: int,
    eps: float,
    repetition: int = 0,
) -> dict:
    set_shingle_size(shingle_size)
    names = queries["canonical"].tolist()
    labels = queries["label_bool"].to_numpy()
    expected = queries["expected_identity"].to_numpy()
    assignments = np.asarray(artifacts.cluster_assignments)
    cluster_matrix = np.asarray(artifacts.cluster_matrix)
    k = int(artifacts.centroids.shape[0])
    logical_width = int(cluster_matrix.shape[1])

    secret_context = create_ckks_context()
    secret_context.generate_relin_keys()
    public_context_bytes = serialize_public_context(secret_context)
    sent_bytes = len(public_context_bytes)
    recv_bytes = 0

    with RssSampler() as sampler:
        online_start = time.perf_counter()

        # ---------------- Round 1: encrypted cluster probe ----------------
        r1_start = time.perf_counter()
        request1, state = prepare_tiled_query_batch(
            names,
            artifacts.scaler_mean,
            artifacts.scaler_scale,
            context=secret_context,
            public_context_bytes=public_context_bytes,
        )
        wire1 = serialize_tiled_first_round_request(request1)
        sent_bytes += sum(len(item) for item in wire1.encrypted_query_200)

        centroid_tiles = compare_tiled_batch_to_centroids(
            wire1, artifacts.centroids, serialize_output=True
        )
        recv_bytes += sum(len(item) for item in centroid_tiles)
        probe_count = min(top_k, k)
        request2, cluster_debug = choose_clusters_and_build_tiled_request(
            centroid_tiles, state, k=k, probes=probe_count
        )
        layout = request2.layout
        wire2 = serialize_tiled_second_round_request(request2)
        sent_bytes += sum(len(item) for item in wire2.encrypted_query_50)
        sent_bytes += sum(len(item) for item in wire2.encrypted_selectors)
        r1_time = time.perf_counter() - r1_start

        # ---------------- Round 2: encrypted candidate scores --------------
        r2_start = time.perf_counter()
        score_iter = tiled_batch_matching(
            cluster_matrix,
            wire2,
            public_context_bytes,
            tau=tau,
            serialize_output=True,
            mask_scope="per_query",
        )

        m = len(names)
        scores = np.full((layout.batch_size, logical_width), -np.inf, dtype=np.float64)
        for (_, start, valid_width), encrypted_tile in zip(
            iter_tile_slices(logical_width, layout), score_iter
        ):
            recv_bytes += len(encrypted_tile)
            tile_scores = decrypt_tiled_score_tile(
                encrypted_tile, secret_context, layout, valid_width
            )
            scores[:, start : start + valid_width] = tile_scores
        for extra in score_iter:  # pragma: no cover - protocol guard
            recv_bytes += len(extra)
            raise ValueError("received extra encrypted score tile")
        r2_time = time.perf_counter() - r2_start
        online_total = time.perf_counter() - online_start

    probed = np.asarray(cluster_debug.probed_clusters)
    if probed.size != m * probe_count:
        raise ValueError("probe layout does not match queries x probes")
    selected = np.asarray(cluster_debug.selected_clusters)
    identities = database["identity"].to_numpy()
    predicted, details = probe_predictions(
        scores,
        probe_count=probe_count,
        eps=eps,
        probed_clusters=probed,
        assignments=assignments,
        identities=identities,
    )
    per_query = []
    for row, detail in enumerate(details):
        per_query.append(
            {
                "query_id": str(queries["query_id"].iat[row]),
                "label": bool(labels[row]),
                "expected_identity": int(expected[row]),
                "selected_cluster": int(selected[row]),
                "probed_clusters": [int(c) for c in probed[row * probe_count : (row + 1) * probe_count]],
                "probes_used": int(detail["probes_used"]),
                "identity": int(detail["identity"]),
            }
        )

    metrics = metrics_from_single(labels, expected, predicted)
    return {
        "repetition": repetition,
        "round1_sec": r1_time,
        "round2_sec": r2_time,
        "online_total_sec": online_total,
        "sent_bytes": sent_bytes,
        "recv_bytes": recv_bytes,
        "total_bytes": sent_bytes + recv_bytes,
        "public_context_bytes": len(public_context_bytes),
        "process_rss_peak_mb": sampler.peak_mb,
        "probe_count": probe_count,
        "mean_probes_used": float(np.mean([d["probes_used"] for d in details])),
        **metrics,
        "per_query": per_query,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--queries-file", type=Path,
                        default=DEFAULT_DATA / "queries.csv")
    parser.add_argument("--k", type=int, default=150)
    parser.add_argument("--shingle-size", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.926)
    parser.add_argument("--top-k", type=int, default=2,
                        help="clusters probed per query (Top-k cluster search)")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--query-limit", type=int, default=0,
                        help="smoke-test subset; 0 runs every query")
    parser.add_argument("--eps", type=float, default=params.BATCH_DECRYPT_EPS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    database, queries = load_data(args.data, args.queries_file)
    if args.query_limit > 0:
        queries = queries.iloc[: args.query_limit].reset_index(drop=True)
    artifacts, offline_stats = build_offline(database, args.k, args.shingle_size)

    runs = [
        run_he_topk(
            database,
            queries,
            artifacts,
            tau=args.tau,
            shingle_size=args.shingle_size,
            top_k=args.top_k,
            eps=args.eps,
            repetition=repetition,
        )
        for repetition in range(args.repetitions)
    ]
    payload = {
        "scheme": "baseline_Integration",
        "queries_file": str(args.queries_file),
        "repetitions": runs,
        "offline": offline_stats,
        "config": {
            "k": int(artifacts.centroids.shape[0]),
            "tau": args.tau,
            "shingle_size": args.shingle_size,
            "top_k": args.top_k,
            "batch_size": len(queries),
            "eps": args.eps,
            "poly_modulus_degree": params.POLY_MODULUS_DEGREE,
            "coeff_mod_bit_sizes": list(params.COEFF_MOD_BIT_SIZES),
            "scale": params.SCALE,
            "mask_scope": "per_query",
            "round2_policy": f"probes={args.top_k}_early_stop",
        },
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    summary = {key: value for key, value in runs[0].items() if key != "per_query"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
