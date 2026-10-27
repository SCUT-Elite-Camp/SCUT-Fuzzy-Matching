"""Scheme 1 (KMeans baseline) with a Top-1 second round on RLdata10000.

The production baseline returns a binary hit/miss over every candidate column.
The comparison requested here returns exactly one record per query, so this
runner extends the protocol in two small, explicit ways:

* Party B masks each query's candidate scores with **one** positive mask
  (``mask_scope="per_query"``) instead of an independent mask per pair.  The
  masked score remains a positive multiple of ``cosine - tau``, so Party A can
  rank candidates within a query, while the absolute cosine stays hidden.
* Party A decrypts all round-2 tiles and returns ``argmax`` over the selected
  cluster, rejecting the query when the best masked score is not positive.

The rest is the repository's real CKKS/TenSEAL protocol.  Calibration (choice
of ``k``, shingle size and ``tau``) is done in plaintext, mirroring the exact
arithmetic of the HE path, and the final measurement is run with real HE.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
import minhash.encoder as encoder  # noqa: E402
from ckks.keys import serialize_public_context  # noqa: E402
from party_a.local_prep import (  # noqa: E402
    create_ckks_context,
    encode_query_batch,
    prepare_tiled_query_batch,
)
from party_a.online_querier import (  # noqa: E402
    check_tiled_score_top1_debug,
    choose_clusters_and_build_tiled_request,
)
from party_b.offline_prep import prepare_party_b_offline  # noqa: E402
from party_b.online_responder import (  # noqa: E402
    compare_tiled_batch_to_centroids,
    tiled_batch_matching,
)
from protocol.transport import (  # noqa: E402
    serialize_tiled_first_round_request,
    serialize_tiled_second_round_request,
)

from names import normalize_name  # noqa: E402

DEFAULT_DATA = EXPERIMENT / "data"


@dataclass
class RssSampler:
    """Sample process RSS in a background thread while the online phase runs."""

    interval: float = 0.02
    peak_mb: float = 0.0
    _stop: threading.Event | None = None
    _thread: threading.Thread | None = None

    def __enter__(self) -> "RssSampler":
        try:
            import psutil
        except ImportError:
            return self
        process = psutil.Process()
        self.peak_mb = process.memory_info().rss / 1024**2
        self._stop = threading.Event()

        def loop() -> None:
            assert self._stop is not None
            while not self._stop.is_set():
                self.peak_mb = max(self.peak_mb, process.memory_info().rss / 1024**2)
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return None


def set_shingle_size(size: int) -> None:
    if size <= 0:
        raise ValueError(f"shingle size must be positive, got {size}")
    params.SHINGLE_SIZE = size
    encoder.SHINGLE_SIZE = size


def load_data(data_dir: Path = DEFAULT_DATA, split: str = "test"):
    database = pd.read_csv(data_dir / "database.csv")
    database["canonical"] = database["normalized_name"].fillna("").map(normalize_name)
    database["identity"] = database["identity"].astype(int)
    database["dob"] = database["dob"].fillna("")
    filename = "queries.csv" if split == "test" else "calibration_queries.csv"
    queries = pd.read_csv(data_dir / filename)
    queries["canonical"] = queries["normalized_name"].fillna("").map(normalize_name)
    queries["label_bool"] = queries["label"].astype(str).str.lower() == "true"
    queries["expected_identity"] = queries["expected_identity"].fillna(-1).astype(int)
    queries["expected_record_id"] = queries["expected_record_id"].fillna("")
    queries["dob"] = queries["dob"].fillna("")
    return database, queries


def build_offline(database: pd.DataFrame, k: int, shingle_size: int):
    set_shingle_size(shingle_size)
    tracemalloc.start()
    start = time.perf_counter()
    artifacts = prepare_party_b_offline(database["canonical"].tolist(), k_mode=str(k))
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return artifacts, {"offline_sec": elapsed, "offline_memory_peak_mb": peak / 1024**2}


def plaintext_scores(
    database: pd.DataFrame,
    queries: pd.DataFrame,
    artifacts,
    *,
    shingle_size: int,
) -> dict:
    """Mirror the HE arithmetic without encryption.

    Round 1 is the *dot product* used by ``compare_tiled_batch_to_centroids``
    (the standardized query is not re-normalised), and round 2 is the dot
    product of two L2-normalised 50-dim vectors.
    """
    set_shingle_size(shingle_size)
    query_200_std, query_50_norm = encode_query_batch(
        queries["canonical"].tolist(),
        artifacts.scaler_mean,
        artifacts.scaler_scale,
    )
    centroids = np.asarray(artifacts.centroids, dtype=np.float64)
    round1_scores = query_200_std @ centroids.T
    selected = np.argmax(round1_scores, axis=1)

    cluster_50 = np.asarray(artifacts.cluster_matrix, dtype=np.float64)
    assignments = np.asarray(artifacts.cluster_assignments)
    best_score = np.full(len(queries), -2.0, dtype=np.float64)
    best_global = np.full(len(queries), -1, dtype=np.int64)
    cluster_recall = np.zeros(len(queries), dtype=bool)
    expected = queries["expected_identity"].to_numpy()
    identities = database["identity"].to_numpy()

    for row in range(len(queries)):
        cluster = int(selected[row])
        members = np.where(assignments == cluster)[0]
        candidates = cluster_50[cluster, : len(members), :]
        scores = candidates @ query_50_norm[row]
        local = int(np.argmax(scores))
        best_score[row] = float(scores[local])
        best_global[row] = int(members[local])
        if queries["label_bool"].iat[row]:
            cluster_recall[row] = bool(np.any(identities[members] == expected[row]))

    return {
        "selected_clusters": selected,
        "best_scores": best_score,
        "best_global_indices": best_global,
        "cluster_recall": cluster_recall,
    }


def metrics_from_scores(
    database: pd.DataFrame,
    queries: pd.DataFrame,
    best_scores: np.ndarray,
    best_global: np.ndarray,
    threshold: float,
) -> dict:
    identities = database["identity"].to_numpy()
    predicted = np.where(best_scores >= threshold, identities[best_global], -1)
    labels = queries["label_bool"].to_numpy()
    expected = queries["expected_identity"].to_numpy()
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


def calibrate(
    database: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    k: int,
    shingle_size: int,
    thresholds: Sequence[float] | None = None,
) -> dict:
    artifacts, offline_stats = build_offline(database, k, shingle_size)
    scores = plaintext_scores(
        database, queries, artifacts, shingle_size=shingle_size
    )
    if thresholds is None:
        thresholds = np.round(np.arange(0.500, 1.0001, 0.002), 4)
    sweep = []
    for tau in thresholds:
        sweep.append(
            {
                "tau": float(tau),
                **metrics_from_scores(
                    database,
                    queries,
                    scores["best_scores"],
                    scores["best_global_indices"],
                    float(tau),
                ),
            }
        )
    best = max(sweep, key=lambda item: (item["f1"], item["accuracy"]))
    positive = queries["label_bool"].to_numpy()
    top1_identity = float(
        np.mean(
            database["identity"].to_numpy()[scores["best_global_indices"]][positive]
            == queries["expected_identity"].to_numpy()[positive]
        )
    )
    cluster_sizes = np.bincount(artifacts.cluster_assignments, minlength=k)
    return {
        "k": k,
        "shingle_size": shingle_size,
        "offline": offline_stats,
        "cluster_sizes": {
            "mean": float(cluster_sizes.mean()),
            "max": int(cluster_sizes.max()),
            "min": int(cluster_sizes.min()),
        },
        "cluster_recall": float(scores["cluster_recall"][positive].mean()),
        "top1_identity_accuracy": top1_identity,
        "best": best,
        "sweep": sweep,
    }


def _map_top1(
    database: pd.DataFrame,
    assignments: np.ndarray,
    selected_clusters: np.ndarray,
    best_columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    identities = database["identity"].to_numpy()
    predicted = np.full(len(selected_clusters), -1, dtype=np.int64)
    for row, (cluster, column) in enumerate(zip(selected_clusters, best_columns)):
        if column < 0:
            continue
        members = np.where(assignments == int(cluster))[0]
        if column >= len(members):
            continue  # zero-padded column of a smaller cluster
        predicted[row] = int(identities[members[int(column)]])
    return predicted


def run_he(
    database: pd.DataFrame,
    queries: pd.DataFrame,
    artifacts,
    *,
    k: int,
    tau: float,
    shingle_size: int,
    repetitions: int = 3,
) -> dict:
    set_shingle_size(shingle_size)
    names = queries["canonical"].tolist()
    labels = queries["label_bool"].to_numpy()
    expected = queries["expected_identity"].to_numpy()
    assignments = np.asarray(artifacts.cluster_assignments)
    k = int(artifacts.centroids.shape[0])

    runs = []
    for repetition in range(repetitions):
        secret_context = create_ckks_context()
        secret_context.generate_relin_keys()
        public_context_bytes = serialize_public_context(secret_context)
        sent_bytes = len(public_context_bytes)
        recv_bytes = 0

        with RssSampler() as sampler:
            # tracemalloc is deliberately not enabled here: tracing every
            # Python float created by ``tolist()`` inflates the measured
            # encryption work by several times.  The process RSS sampled above
            # is the HE memory peak; the offline phase keeps its own
            # tracemalloc measurement.
            online_start = time.perf_counter()

            # ---------------- Round 1 ----------------
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
            request2, cluster_debug = choose_clusters_and_build_tiled_request(
                centroid_tiles, state, k=k
            )
            wire2 = serialize_tiled_second_round_request(request2)
            sent_bytes += sum(len(item) for item in wire2.encrypted_query_50)
            sent_bytes += sum(len(item) for item in wire2.encrypted_selectors)
            r1_time = time.perf_counter() - r1_start

            # ---------------- Round 2 ----------------
            r2_start = time.perf_counter()
            score_iter = tiled_batch_matching(
                artifacts.cluster_matrix,
                wire2,
                public_context_bytes,
                tau=tau,
                serialize_output=True,
                mask_scope="per_query",
            )

            def counted(iterator=score_iter):
                nonlocal recv_bytes
                for item in iterator:
                    recv_bytes += len(item)
                    yield item

            match_result, match_debug, best_scores, best_columns = (
                check_tiled_score_top1_debug(
                    counted(),
                    secret_context,
                    layout=state.layout,
                    logical_width=artifacts.cluster_matrix.shape[1],
                )
            )
            online_total = time.perf_counter() - online_start
            r2_time = time.perf_counter() - r2_start

        predicted = _map_top1(
            database,
            assignments,
            cluster_debug.selected_clusters,
            best_columns,
        )
        tp = int(np.sum(labels & (predicted == expected)))
        fp = int(np.sum(~labels & (predicted != -1)))
        fn = int(np.sum(labels & (predicted != expected)))
        tn = int(np.sum(~labels & (predicted == -1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        runs.append(
            {
                "repetition": repetition,
                "round1_sec": r1_time,
                "round2_sec": r2_time,
                "online_total_sec": online_total,
                "sent_bytes": sent_bytes,
                "recv_bytes": recv_bytes,
                "total_bytes": sent_bytes + recv_bytes,
                "public_context_bytes": len(public_context_bytes),
                "process_rss_peak_mb": sampler.peak_mb,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": (tp + tn) / len(labels),
                "agreement_with_plaintext": None,
            }
        )
    return {
        "repetitions": runs,
        "config": {
            "k": k,
            "tau": tau,
            "shingle_size": shingle_size,
            "batch_size": len(names),
            "he_batch_size": len(names),
            "poly_modulus_degree": params.POLY_MODULUS_DEGREE,
            "coeff_mod_bit_sizes": list(params.COEFF_MOD_BIT_SIZES),
            "scale": params.SCALE,
            "mask_scope": "per_query",
            "round2_policy": "top1",
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--mode", choices=["calibrate", "run"], default="run")
    parser.add_argument("--split", choices=["calibration", "test"], default="test")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--shingle-size", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    database, queries = load_data(args.data, args.split)
    if args.query_limit > 0:
        queries = queries.iloc[: args.query_limit].reset_index(drop=True)
    if args.mode == "calibrate":
        payload = calibrate(
            database,
            queries,
            k=args.k,
            shingle_size=args.shingle_size,
        )
    else:
        artifacts, offline_stats = build_offline(database, args.k, args.shingle_size)
        payload = run_he(
            database,
            queries,
            artifacts,
            k=args.k,
            tau=args.tau,
            shingle_size=args.shingle_size,
            repetitions=args.repetitions,
        )
        payload["offline"] = offline_stats
        if args.calibration and args.calibration.exists():
            calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
            payload["calibration_reference"] = calibration.get("best")
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
