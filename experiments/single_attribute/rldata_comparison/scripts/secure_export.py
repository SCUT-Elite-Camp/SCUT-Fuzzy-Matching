"""Export the two party inputs for the end-to-end secure scheme-2 pipeline.

This is the only Python step in the secure path.  Each party encodes its own
names locally with the shared b-bit MinHash encoder and writes a small binary
file; the C++ ``name_secure_pipeline`` binary then runs PSI candidate
generation and the garbled-circuit Top-1 stage.

Security split
--------------
* ``--party server`` reads only ``database.csv``.  The output contains row
  indices, identities, opaque tokens and bit vectors - never query data,
  labels or predictions.
* ``--party client`` reads only ``queries.csv``.  The output contains query
  indices and bit vectors - never labels, expected identities or perturbations.
* ``--party reference`` is an evaluation-only mode.  It reproduces the secure
  protocol in plaintext (slot expansion S=8, Hamming threshold, tie-break on
  ``(distance, row index)``) so the C++ results can be checked query by query.

Run::

    python experiments/single_attribute/rldata_comparison/scripts/secure_export.py \
        --party server --out results/secure/server.bin
    python experiments/single_attribute/rldata_comparison/scripts/secure_export.py \
        --party client --out results/secure/client.bin
    python experiments/single_attribute/rldata_comparison/scripts/secure_export.py \
        --party reference --out results/secure/reference.json

Binary layout (little endian)::

    header   : <9IQII  magic, version, party, ell_bits, ell_bytes,
                        distance, rounds, slots, shingle, seed, count, nproj
    proj     : rounds x (u32 n + n x u32 coordinate)
    server   : count x (u32 index, u32 identity, u64 token, ell_bytes bits)
    client   : count x (u32 index, ell_bytes bits)

Bits are packed with ``numpy.packbits`` (big-endian within each byte): bit
``j`` lives in byte ``j // 8`` at position ``7 - j % 8``.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from encoding import BitEncoding, encode_names, hamming_packed, pack_bits  # noqa: E402
from fused_top1 import (  # noqa: E402
    DEFAULT_DATA,
    _keys_for,
    load_database,
    load_queries,
    sample_projections,
)

MAGIC = 0x4E505349  # "NPSI"
VERSION = 1
PARTY_SERVER = 1
PARTY_CLIENT = 2
ENCODING = BitEncoding(128, 2, 2)
DEFAULT_PROBABILITY = 0.11
DEFAULT_ROUNDS = 80
DEFAULT_SLOTS = 8
DEFAULT_DISTANCE = 32
DEFAULT_SEED = 20261023
DEFAULT_TOP_K = 1


# ---------------------------------------------------------------------------
# Binary writing
# ---------------------------------------------------------------------------


def write_header(handle, *, party: int, seed: int, count: int, nproj: int, args) -> None:
    handle.write(
        struct.pack(
            "<9IQII",
            MAGIC,
            VERSION,
            party,
            ENCODING.length,
            ENCODING.length // 8,
            args.distance,
            args.rounds,
            args.slots,
            ENCODING.shingle_size,
            seed,
            count,
            nproj,
        )
    )


def write_projections(handle, projections: list[np.ndarray]) -> None:
    for indices in projections:
        handle.write(struct.pack("<I", int(indices.size)))
        handle.write(indices.astype("<u4").tobytes())


def unique_tokens(count: int, seed: int) -> np.ndarray:
    """Deterministic, unpredictable-to-the-client 64-bit record tokens."""
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, 2**64, size=count, dtype=np.uint64, endpoint=False)
    if np.unique(tokens).size != count:  # probability ~1e-11 for 10k rows
        raise RuntimeError("token collision; change the token seed")
    return tokens


def _subset(frame, limit: int):
    if limit <= 0:
        return frame
    return frame.iloc[:limit].reset_index(drop=True)


def export_server(args) -> dict:
    database = _subset(load_database(args.data), args.limit_records)
    bits = encode_names(database["canonical"].tolist(), ENCODING)
    packed = pack_bits(bits).astype(np.uint8)
    identities = database["identity"].to_numpy(dtype=np.uint32)
    tokens = unique_tokens(len(database), args.seed + 0x5EED)
    projections = sample_projections(
        ENCODING.length, args.probability, args.rounds, args.seed
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as handle:
        write_header(
            handle, party=PARTY_SERVER, seed=args.seed,
            count=len(database), nproj=len(projections), args=args,
        )
        write_projections(handle, projections)
        for row in range(len(database)):
            handle.write(
                struct.pack(
                    "<IIQ", row, int(identities[row]), int(tokens[row])
                )
            )
            handle.write(packed[row].tobytes())

    return {
        "party": "server",
        "records": int(len(database)),
        "ell_bits": ENCODING.length,
        "rounds": int(args.rounds),
        "slots": int(args.slots),
        "seed": int(args.seed),
        "bytes": int(args.out.stat().st_size),
    }


def export_client(args) -> dict:
    queries = _subset(
        load_queries(
            args.data, split="test",
            queries_file=getattr(args, "queries_file", None),
        ),
        args.limit_queries,
    )
    bits = encode_names(queries["canonical"].tolist(), ENCODING)
    packed = pack_bits(bits).astype(np.uint8)
    projections = sample_projections(
        ENCODING.length, args.probability, args.rounds, args.seed
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as handle:
        write_header(
            handle, party=PARTY_CLIENT, seed=args.seed,
            count=len(queries), nproj=len(projections), args=args,
        )
        write_projections(handle, projections)
        for row in range(len(queries)):
            handle.write(struct.pack("<I", row))
            handle.write(packed[row].tobytes())

    return {
        "party": "client",
        "queries": int(len(queries)),
        "ell_bits": ENCODING.length,
        "rounds": int(args.rounds),
        "slots": int(args.slots),
        "seed": int(args.seed),
        "bytes": int(args.out.stat().st_size),
    }


# ---------------------------------------------------------------------------
# Plaintext reference (mirror of the secure protocol)
# ---------------------------------------------------------------------------


def reference_candidates(
    db_bits: np.ndarray,
    query_bits: np.ndarray,
    projections: list[np.ndarray],
    slots: int,
) -> list[set[int]]:
    """Candidate rows per query after S-slot expansion.

    For every round each database bucket contributes only its first ``slots``
    members in ascending row order - exactly what the server puts into PSI.
    """
    candidates: list[set[int]] = [set() for _ in range(len(query_bits))]
    for indices in projections:
        db_keys = _keys_for(db_bits, indices)
        query_keys = _keys_for(query_bits, indices)
        order = np.argsort(db_keys, kind="stable")
        sorted_keys = db_keys[order]
        for query_index, key in enumerate(query_keys):
            left = int(np.searchsorted(sorted_keys, key, side="left"))
            right = int(np.searchsorted(sorted_keys, key, side="right"))
            if right <= left:
                continue
            members = order[left : min(right, left + slots)]
            candidates[query_index].update(int(member) for member in members)
    return candidates


def reference_predictions(args) -> dict:
    database = _subset(load_database(args.data), args.limit_records)
    queries = _subset(
        load_queries(
            args.data, split="test",
            queries_file=getattr(args, "queries_file", None),
        ),
        args.limit_queries,
    )
    db_bits = encode_names(database["canonical"].tolist(), ENCODING)
    query_bits = encode_names(queries["canonical"].tolist(), ENCODING)
    projections = sample_projections(
        ENCODING.length, args.probability, args.rounds, args.seed
    )

    candidates = reference_candidates(db_bits, query_bits, projections, args.slots)
    packed_db = pack_bits(db_bits)
    packed_query = pack_bits(query_bits)
    identities = database["identity"].to_numpy()
    labels = queries["label_bool"].to_numpy()
    expected = queries["expected_identity"].to_numpy()

    top_k = int(getattr(args, "top_k", DEFAULT_TOP_K) or DEFAULT_TOP_K)
    predictions = []
    for query_index, members in enumerate(candidates):
        ranked: list[int] = []
        best_distance = -1
        if members:
            rows = np.array(sorted(members), dtype=np.int64)
            distances = hamming_packed(
                packed_query[query_index : query_index + 1], packed_db[rows]
            )[0]
            # Tie-break on (distance, row index), matching the circuit.
            order = np.lexsort((rows, distances))
            best_distance = int(distances[order[0]])
            for local in order[:top_k]:
                if int(distances[local]) > args.distance:
                    break
                ranked.append(int(identities[rows[local]]))
        predictions.append(
            {
                "query_index": query_index,
                "identity": ranked[0] if ranked else -1,
                "identities": ranked,
                "distance": best_distance,
                "candidates": len(members),
            }
        )

    predicted_sets = [set(row["identities"]) for row in predictions]
    correct = np.array(
        [expected[i] in predicted_sets[i] for i in range(len(predictions))],
        dtype=bool,
    )
    returned = np.array([bool(row["identities"]) for row in predictions], dtype=bool)
    tp = int(np.sum(labels & correct))
    fp = int(np.sum(~labels & returned))
    fn = int(np.sum(labels & ~correct))
    tn = int(np.sum(~labels & ~returned))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    payload = {
        "seed": int(args.seed),
        "rounds": int(args.rounds),
        "slots": int(args.slots),
        "distance_threshold": int(args.distance),
        "top_k": top_k,
        "metrics": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "accuracy": (tp + tn) / len(labels),
        },
        "candidate_pairs": int(sum(len(items) for items in candidates)),
        "predictions": predictions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--party", choices=["server", "client", "reference"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--probability", type=float, default=DEFAULT_PROBABILITY)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    parser.add_argument("--distance", type=int, default=DEFAULT_DISTANCE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--queries-file", type=Path, default=None,
                        help="alternative query CSV (default: data/queries.csv)")
    # Smoke-test subsets: the same limits must be used for the reference.
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--limit-queries", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.party == "server":
        result = export_server(args)
    elif args.party == "client":
        result = export_client(args)
    else:
        result = reference_predictions(args)
    summary = {
        key: value
        for key, value in result.items()
        if key != "predictions"
    }
    print(json.dumps(summary, indent=2))
    if args.party == "reference":
        print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
