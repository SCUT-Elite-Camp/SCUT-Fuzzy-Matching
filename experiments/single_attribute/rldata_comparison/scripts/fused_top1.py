"""Scheme 2: random-projection candidate filter + Top-1 Hamming verification.

This is the application-layer adaptation of ``fused-secure-filter`` to the
RLdata10000 name-matching task.  The paper's filter has two stages:

1. **Candidate generation (round 1).**  Each side projects its binary vector on
   a fresh random subset of coordinates.  Elements that land in the same bucket
   are candidate pairs.  Repeating this ``k`` times gives an approximate PSI.
2. **Verification (round 2).**  Candidate pairs are compared with a secure
   Hamming circuit; a pair is a match when its distance is at most ``d``.

The repository's C++ implementation keeps *one random representative per
bucket* (Algorithm 1, line 4).  That is faithful to the paper but loses recall
when buckets are large.  For Top-1 retrieval we also provide an ``all`` mode
that keeps every member of a matched bucket.  Both modes are simulated here;
the secure backend (EMP + volePSI) is not installed in this environment, so
timings are plaintext and communication is estimated separately from the
repository's measured per-candidate cost (see ``README.md``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from encoding import (  # noqa: E402
    BitEncoding,
    encode_names,
    hamming_packed,
    pack_bits,
)
from names import name_variants, normalize_name  # noqa: E402

DEFAULT_DATA = EXPERIMENT / "data"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectionParams:
    """Paper-style parameter derivation (``src/params.cpp``).

    ``n`` is the size of the server's set (10,000).  ``d`` is the Hamming
    threshold, ``t`` the bucket-size gap and ``lambda`` the statistical
    parameter.  The derived ``p`` and ``rounds`` are exposed so callers can
    override them for a Top-1 cost/recall trade-off.
    """

    n: int
    length: int
    d: int
    t: float
    lam: float

    @property
    def q(self) -> float:
        return 1.0 / ((self.n * self.t) ** (1.0 / (self.t - 1.0)))

    @property
    def p(self) -> float:
        return 1.0 - self.q ** (1.0 / self.d)

    @property
    def alpha(self) -> float:
        return 1.0 - self.q + self.n * (self.q**self.t)

    @property
    def rounds(self) -> int:
        return int(math.ceil((self.lam + math.log2(self.n)) / math.log2(1.0 / self.alpha)))

    @property
    def coordinates(self) -> int:
        return int(round(self.p * self.length))


@dataclass
class FusedConfig:
    encoding: BitEncoding = field(default_factory=lambda: BitEncoding(128, 2, 2))
    projection_probability: float = 0.10
    rounds: int = 32
    distance_threshold: int = 40
    candidate_mode: str = "all"  # "all" or "representative"
    seed: int = 20261023
    shingle_size: int | None = None  # None -> encoding.shingle_size
    use_variants: bool = False
    dob_weight: float = 0.0
    dob_missing_penalty_years: float = 5.0
    dob_cap_years: float = 10.0

    def resolved_encoding(self) -> BitEncoding:
        if self.shingle_size is None or self.shingle_size == self.encoding.shingle_size:
            return self.encoding
        return BitEncoding(
            permutations=self.encoding.permutations,
            bits_per_permutation=self.encoding.bits_per_permutation,
            shingle_size=self.shingle_size,
            seed=self.encoding.seed,
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["encoding"] = asdict(self.resolved_encoding())
        payload["encoding"]["length"] = self.resolved_encoding().length
        return payload


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_database(path: Path = DEFAULT_DATA) -> pd.DataFrame:
    frame = pd.read_csv(path / "database.csv")
    frame["canonical"] = frame["normalized_name"].fillna("").map(normalize_name)
    frame["identity"] = frame["identity"].astype(int)
    frame["dob"] = frame["dob"].fillna("")
    return frame


def load_queries(
    path: Path = DEFAULT_DATA,
    split: str = "test",
    queries_file: Path | None = None,
) -> pd.DataFrame:
    if queries_file is None:
        filename = "queries.csv" if split == "test" else "calibration_queries.csv"
        queries_file = path / filename
    frame = pd.read_csv(queries_file)
    frame["canonical"] = frame["normalized_name"].fillna("").map(normalize_name)
    frame["label_bool"] = frame["label"].astype(str).str.lower() == "true"
    frame["expected_identity"] = frame["expected_identity"].fillna(-1).astype(int)
    frame["expected_record_id"] = frame["expected_record_id"].fillna("")
    frame["dob"] = frame["dob"].fillna("")
    frame["identity"] = frame["expected_identity"]
    return frame


# ---------------------------------------------------------------------------
# Projection and bucketing
# ---------------------------------------------------------------------------


def sample_projections(
    length: int,
    probability: float,
    rounds: int,
    seed: int,
) -> list[np.ndarray]:
    """Sample ``rounds`` independent Bernoulli coordinate sets."""
    if not (0.0 < probability < 1.0):
        raise ValueError(f"projection probability must be in (0,1), got {probability}")
    if rounds <= 0:
        raise ValueError(f"rounds must be positive, got {rounds}")
    rng = np.random.RandomState(seed)
    projections: list[np.ndarray] = []
    for _ in range(rounds):
        indices = np.flatnonzero(rng.rand(length) < probability)
        if indices.size == 0:
            indices = np.array([int(rng.randint(length))], dtype=np.int64)
        projections.append(indices.astype(np.int64))
    return projections


def _pack_word(selected: np.ndarray) -> np.ndarray:
    powers = np.uint64(1) << np.arange(selected.shape[1], dtype=np.uint64)
    return selected.astype(np.uint64) @ powers


def _keys_for(bits: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Pack the selected bits of every row into one 64-bit key.

    Up to 63 coordinates are packed exactly.  Wider projections are folded with
    a 64-bit multiply/xor; the collision probability for 10k rows is below
    2**-32, and a collision only adds a candidate (the Hamming check still
    decides), so it cannot create a false match.
    """
    if indices.size <= 63:
        return _pack_word(bits[:, indices])
    first = indices[:63]
    rest = indices[63:]
    key = _pack_word(bits[:, first])
    if rest.size:
        tail = _pack_word(bits[:, rest])
        key = key ^ (tail * np.uint64(0x9E3779B97F4A7C15))
    return key


def _bucket_candidates_all(
    db_keys: np.ndarray,
    query_keys: np.ndarray,
    max_candidates: int | None = None,
) -> tuple[list[np.ndarray], int]:
    """Return, for each query, all DB indices sharing its key."""
    order = np.argsort(db_keys, kind="stable")
    sorted_keys = db_keys[order]
    output: list[np.ndarray] = []
    intersections = 0
    for key in query_keys:
        left = int(np.searchsorted(sorted_keys, key, side="left"))
        right = int(np.searchsorted(sorted_keys, key, side="right"))
        if right <= left:
            output.append(np.empty(0, dtype=np.int64))
            continue
        members = order[left:right]
        if max_candidates is not None and members.size > max_candidates:
            members = members[:max_candidates]
        intersections += 1
        output.append(members.astype(np.int64))
    return output, intersections


def _bucket_candidates_representative(
    db_keys: np.ndarray,
    query_keys: np.ndarray,
    rng: np.random.RandomState,
) -> tuple[list[np.ndarray], int]:
    """Keep one random DB representative per key (Algorithm 1, line 4)."""
    buckets: dict[int, list[int]] = {}
    for index, key in enumerate(db_keys):
        buckets.setdefault(int(key), []).append(index)
    representatives = {
        key: int(members[rng.randint(len(members))]) for key, members in buckets.items()
    }
    output: list[np.ndarray] = []
    intersections = 0
    for key in query_keys:
        representative = representatives.get(int(key))
        if representative is None:
            output.append(np.empty(0, dtype=np.int64))
        else:
            intersections += 1
            output.append(np.array([representative], dtype=np.int64))
    return output, intersections


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _dob_penalty_years(
    query_dob: str,
    db_dob: Sequence[str],
    *,
    missing_penalty: float,
    cap: float,
) -> np.ndarray:
    import datetime as dt

    def to_days(value: str) -> float:
        if not value:
            return float("nan")
        return float((dt.date.fromisoformat(value) - dt.date(1900, 1, 1)).days)

    query_days = to_days(query_dob)
    db_days = np.array([to_days(value) for value in db_dob], dtype=np.float64)
    penalty = np.abs(db_days - query_days) / 365.25
    penalty = np.minimum(penalty, cap)
    both_missing = math.isnan(query_days) & np.isnan(db_days)
    one_missing = math.isnan(query_days) ^ np.isnan(db_days)
    penalty[both_missing] = 0.0
    penalty[one_missing] = missing_penalty
    return penalty


def verify_top1(
    query_bits: np.ndarray,
    db_bits: np.ndarray,
    packed_db: np.ndarray,
    candidates: list[np.ndarray],
    *,
    distance_threshold: int,
    identities: np.ndarray,
    query_dobs: Sequence[str] | None = None,
    db_dobs: Sequence[str] | None = None,
    dob_weight: float = 0.0,
    dob_missing_penalty: float = 5.0,
    dob_cap: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (predicted identity, best distance, effective score) per query.

    ``distance_threshold`` applies to the *combined* score.  Without a DOB
    weight this is exactly the Hamming distance.
    """
    predicted = np.full(len(query_bits), -1, dtype=np.int64)
    best_distance = np.full(len(query_bits), -1, dtype=np.int64)
    best_score = np.full(len(query_bits), np.inf, dtype=np.float64)
    for row, candidate_indices in enumerate(candidates):
        if candidate_indices.size == 0:
            continue
        packed_query = pack_bits(query_bits[row : row + 1])
        packed_candidates = packed_db[candidate_indices]
        distances = hamming_packed(packed_query, packed_candidates)[0]
        if dob_weight > 0.0:
            if query_dobs is None or db_dobs is None:
                raise ValueError("DOB weight requires query_dobs and db_dobs")
            penalty = _dob_penalty_years(
                query_dobs[row],
                [db_dobs[index] for index in candidate_indices],
                missing_penalty=dob_missing_penalty,
                cap=dob_cap,
            )
            scores = distances.astype(np.float64) + dob_weight * penalty
        else:
            scores = distances.astype(np.float64)
        local = int(np.argmin(scores))
        if scores[local] <= distance_threshold:
            predicted[row] = int(identities[candidate_indices[local]])
            best_distance[row] = int(distances[local])
            best_score[row] = float(scores[local])
    return predicted, best_distance, best_score


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class FusedResult:
    predictions: np.ndarray
    distances: np.ndarray
    scores: np.ndarray
    candidate_counts: np.ndarray
    stats: dict
    config: dict

    def metrics(self, labels: np.ndarray, expected: np.ndarray) -> dict:
        predicted = self.predictions
        tp = int(np.sum(labels & (predicted == expected)))
        fp = int(np.sum(~labels & (predicted != -1)))
        fn = int(np.sum(labels & (predicted != expected)))
        tn = int(np.sum(~labels & (predicted == -1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }


def _query_forms(raw: str, canonical: str, use_variants: bool) -> list[str]:
    if not use_variants:
        return [canonical]
    forms = name_variants(raw)
    if canonical not in forms:
        forms.insert(0, canonical)
    return forms


def run_fused(
    database: pd.DataFrame,
    queries: pd.DataFrame,
    config: FusedConfig,
    *,
    seed_offset: int = 0,
) -> FusedResult:
    encoding = config.resolved_encoding()
    seed = config.seed + seed_offset

    # ---------------- Offline (Party B) ----------------
    tracemalloc.start()
    offline_start = time.perf_counter()
    db_bits = encode_names(database["canonical"].tolist(), encoding)
    packed_db = pack_bits(db_bits)
    offline_time = time.perf_counter() - offline_start
    _, offline_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    projections = sample_projections(
        encoding.length, config.projection_probability, config.rounds, seed
    )

    # ---------------- Online round 1: candidate generation ----------------
    query_forms = [
        _query_forms(raw, canonical, config.use_variants)
        for raw, canonical in zip(queries["name"].tolist(), queries["canonical"].tolist())
    ]
    flat_forms = [form for forms in query_forms for form in forms]
    form_offsets = np.cumsum([0] + [len(forms) for forms in query_forms])

    tracemalloc.start()
    round1_start = time.perf_counter()
    form_bits = encode_names(flat_forms, encoding)

    rng = np.random.RandomState(seed + 1)
    candidate_sets: list[set[int]] = [set() for _ in range(len(queries))]
    intersections = 0
    total_coordinates = 0
    for indices in projections:
        total_coordinates += int(indices.size)
        db_keys = _keys_for(db_bits, indices)
        form_keys = _keys_for(form_bits, indices)
        if config.candidate_mode == "all":
            buckets, hit = _bucket_candidates_all(db_keys, form_keys)
        elif config.candidate_mode == "representative":
            buckets, hit = _bucket_candidates_representative(db_keys, form_keys, rng)
        else:
            raise ValueError(f"unknown candidate mode: {config.candidate_mode}")
        intersections += hit
        for form_index, members in enumerate(buckets):
            if members.size == 0:
                continue
            query_index = int(np.searchsorted(form_offsets, form_index, side="right") - 1)
            if form_index >= form_offsets[query_index + 1]:
                continue
            candidate_sets[query_index].update(int(member) for member in members)
    round1_time = time.perf_counter() - round1_start
    _, round1_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    candidates = [np.fromiter(values, dtype=np.int64) for values in candidate_sets]

    # ---------------- Online round 2: Top-1 verification ----------------
    tracemalloc.start()
    round2_start = time.perf_counter()
    # Verification uses the canonical form.  Variant search affects candidate
    # generation only; the canonical form is the one both parties agree on.
    canonical_bits = form_bits[form_offsets[:-1]]
    predictions, distances, scores = verify_top1(
        canonical_bits,
        db_bits,
        packed_db,
        candidates,
        distance_threshold=config.distance_threshold,
        identities=database["identity"].to_numpy(),
        query_dobs=queries["dob"].tolist(),
        db_dobs=database["dob"].tolist(),
        dob_weight=config.dob_weight,
        dob_missing_penalty=config.dob_missing_penalty_years,
        dob_cap=config.dob_cap_years,
    )
    round2_time = time.perf_counter() - round2_start
    _, round2_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    candidate_counts = np.array([len(items) for items in candidates], dtype=np.int64)
    stats = {
        "offline_sec": offline_time,
        "round1_sec": round1_time,
        "round2_sec": round2_time,
        "online_total_sec": round1_time + round2_time,
        "offline_memory_peak_mb": offline_peak / 1024**2,
        "online_memory_peak_mb": max(round1_peak, round2_peak) / 1024**2,
        "rounds": config.rounds,
        "total_projection_coordinates": total_coordinates,
        "mean_coordinates_per_round": total_coordinates / config.rounds,
        "projection_intersections": intersections,
        "candidate_pairs": int(candidate_counts.sum()),
        "candidate_mean": float(candidate_counts.mean()),
        "candidate_max": int(candidate_counts.max()) if candidate_counts.size else 0,
        "candidate_min": int(candidate_counts.min()) if candidate_counts.size else 0,
        "unique_database_buckets": None,
    }
    return FusedResult(
        predictions=predictions,
        distances=distances,
        scores=scores,
        candidate_counts=candidate_counts,
        stats=stats,
        config=config.to_dict(),
    )


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def sweep_distance_threshold(
    result: FusedResult,
    labels: np.ndarray,
    expected: np.ndarray,
    thresholds: Iterable[int],
) -> list[dict]:
    """Re-threshold a finished run without recomputing candidates.

    ``result.scores`` already stores the best combined score per query, so the
    sweep is exact for the candidates that were generated.
    """
    output = []
    for threshold in thresholds:
        predictions = np.where(
            (result.scores <= threshold) & (result.scores < np.inf),
            result.predictions,
            -1,
        )
        candidate = FusedResult(
            predictions=predictions,
            distances=result.distances,
            scores=result.scores,
            candidate_counts=result.candidate_counts,
            stats=result.stats,
            config=result.config,
        )
        output.append({"threshold": int(threshold), **candidate.metrics(labels, expected)})
    return output


def oracle_top1(
    database: pd.DataFrame,
    queries: pd.DataFrame,
    encoding: BitEncoding,
    thresholds: Iterable[int],
) -> list[dict]:
    """Best possible Top-1 over the whole database (no filter).

    This establishes the encoding's ceiling, so a filter's recall loss can be
    separated from the encoding's own errors.
    """
    db_bits = encode_names(database["canonical"].tolist(), encoding)
    query_bits = encode_names(queries["canonical"].tolist(), encoding)
    distances = hamming_packed(pack_bits(query_bits), pack_bits(db_bits))
    nearest = distances.argmin(axis=1)
    nearest_distance = distances[np.arange(len(queries)), nearest]
    identities = database["identity"].to_numpy()
    labels = queries["label_bool"].to_numpy()
    expected = queries["expected_identity"].to_numpy()
    output = []
    for threshold in thresholds:
        predictions = np.where(nearest_distance <= threshold, identities[nearest], -1)
        tp = int(np.sum(labels & (predictions == expected)))
        fp = int(np.sum(~labels & (predictions != -1)))
        fn = int(np.sum(labels & (predictions != expected)))
        tn = int(np.sum(~labels & (predictions == -1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        output.append(
            {
                "threshold": int(threshold),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": (tp + tn) / len(labels),
            }
        )
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=["calibration", "test"], default="calibration")
    parser.add_argument("--permutations", type=int, default=128)
    parser.add_argument("--bits-per-permutation", type=int, default=2)
    parser.add_argument("--shingle-size", type=int, default=2)
    parser.add_argument("--projection-probability", type=float, default=0.10)
    parser.add_argument("--rounds", type=int, default=32)
    parser.add_argument("--distance-threshold", type=int, default=40)
    parser.add_argument("--candidate-mode", choices=["all", "representative"], default="all")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--variants", action="store_true")
    parser.add_argument("--dob-weight", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--oracle", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    database = load_database(args.data)
    queries = load_queries(args.data, args.split)
    encoding = BitEncoding(
        permutations=args.permutations,
        bits_per_permutation=args.bits_per_permutation,
        shingle_size=args.shingle_size,
    )
    config = FusedConfig(
        encoding=encoding,
        projection_probability=args.projection_probability,
        rounds=args.rounds,
        distance_threshold=args.distance_threshold,
        candidate_mode=args.candidate_mode,
        seed=args.seed,
        use_variants=args.variants,
        dob_weight=args.dob_weight,
    )
    labels = queries["label_bool"].to_numpy()
    expected = queries["expected_identity"].to_numpy()

    result = run_fused(database, queries, config)
    payload = {
        "split": args.split,
        "config": result.config,
        "stats": result.stats,
        "metrics": result.metrics(labels, expected),
        "threshold_sweep": sweep_distance_threshold(
            result, labels, expected, range(0, encoding.length + 1, 2)
        ),
        "per_query": [
            {
                "query_id": row.query_id,
                "name": row.name,
                "label": bool(row.label_bool),
                "expected_identity": int(row.expected_identity),
                "perturbation": row.perturbation,
                "predicted_identity": int(result.predictions[index]),
                "best_distance": int(result.distances[index]),
                "best_score": (
                    None if not np.isfinite(result.scores[index]) else float(result.scores[index])
                ),
                "candidates": int(result.candidate_counts[index]),
            }
            for index, row in enumerate(queries.itertuples())
        ],
    }
    if args.oracle:
        payload["oracle"] = oracle_top1(
            database, queries, encoding, range(0, encoding.length + 1, 2)
        )
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
