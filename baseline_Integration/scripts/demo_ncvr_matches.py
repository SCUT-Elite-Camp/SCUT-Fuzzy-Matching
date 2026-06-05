"""Small NCVR demo that prints query-level match and cluster details.

This script is intentionally presentation-oriented: it runs a tiny real-data
slice fast, then shows which queries matched, which cluster Party A selected,
and which B-side record was hit first.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.params import SIMILARITY_THRESHOLD
from ckks.keys import serialize_public_context
from evaluation.dataset_loader import load_dataset
from party_a.local_prep import (
    create_ckks_context,
    prepare_encrypted_query_with_context,
)
from party_a.online_querier import (
    check_encrypted_scores_debug,
    choose_cluster_and_build_request,
)
from party_b.offline_prep import prepare_party_b_offline
from party_b.online_responder import compare_to_centroids, column_wise_matching


def _parse_query_indices(value: str) -> list[int]:
    indices = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        indices.append(int(part))
    if not indices:
        raise argparse.ArgumentTypeError("query indices cannot be empty")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small NCVR 10K protocol demo and show matched queries plus "
            "selected cluster ids."
        )
    )
    parser.add_argument(
        "--data-path",
        default=str(PROJECT_ROOT / "data"),
        help="Path to data/ or data/ncvr_10k.",
    )
    parser.add_argument(
        "--db-limit",
        type=int,
        default=100,
        help="Use the first N B-side records. Default 100 keeps the demo fast.",
    )
    parser.add_argument(
        "--query-indices",
        type=_parse_query_indices,
        default=_parse_query_indices("2,7,26,49,100"),
        help=(
            "Comma-separated query row indices from ncvr_10k_queries.csv. "
            "Default uses positives from different clusters/columns and one negative."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Fixed cluster count for the demo. Default 10.",
    )
    parser.add_argument("--tau", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "artifacts" / "demo" / "ncvr_matches"),
        help="Directory for JSON/CSV demo outputs.",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Scan all cluster columns instead of stopping at the first hit.",
    )
    return parser.parse_args()


def _first_hit_name(
    names_b: list[str],
    cluster_assignments: np.ndarray,
    selected_cluster: int,
    first_positive_column: int | None,
) -> tuple[int | None, str | None, int]:
    members = np.where(cluster_assignments == selected_cluster)[0]
    cluster_size = int(len(members))
    if first_positive_column is None or first_positive_column >= cluster_size:
        return None, None, cluster_size
    database_index = int(members[first_positive_column])
    return database_index, names_b[database_index], cluster_size


def _format_bool(value: bool) -> str:
    return "Y" if value else "N"


def _print_table(
    rows: list[dict],
    *,
    offline_sec: float,
    context_sec: float,
    online_sec: float,
    total_sec: float,
) -> None:
    columns = [
        ("query_index", "idx"),
        ("query_name", "query"),
        ("expected_label", "label"),
        ("predicted_match", "catch"),
        ("selected_cluster", "cluster"),
        ("first_positive_column", "hit_col"),
        ("matched_name", "matched B-side name"),
        ("checked_columns", "checked"),
        ("total_sec", "sec"),
    ]
    rendered = []
    for row in rows:
        rendered.append(
            {
                "query_index": str(row["query_index"]),
                "query_name": row["query_name"],
                "expected_label": _format_bool(row["expected_label"]),
                "predicted_match": _format_bool(row["predicted_match"]),
                "selected_cluster": str(row["selected_cluster"]),
                "first_positive_column": (
                    "" if row["first_positive_column"] is None
                    else str(row["first_positive_column"])
                ),
                "matched_name": row["matched_name"] or "",
                "checked_columns": str(row["checked_columns"]),
                "total_sec": f"{row['total_sec']:.2f}",
            }
        )

    widths = {
        key: max(len(title), *(len(item[key]) for item in rendered))
        for key, title in columns
    }
    header = " | ".join(title.ljust(widths[key]) for key, title in columns)
    divider = "-+-".join("-" * widths[key] for key, _ in columns)
    print(header)
    print(divider)
    for item in rendered:
        print(" | ".join(item[key].ljust(widths[key]) for key, _ in columns))
    print()
    print(f"Offline preprocessing: {offline_sec:.2f}s")
    print(f"HE context init:       {context_sec:.2f}s")
    print(f"Online query total:    {online_sec:.2f}s")
    print(f"Demo total time:       {total_sec:.2f}s")
    print(
        "Note: selected_cluster is shown only by this local demo/debug script; "
        "the production second-round request still sends an encrypted selector."
    )


def main() -> int:
    args = parse_args()
    if args.db_limit <= 0:
        raise ValueError("--db-limit must be positive for this demo")
    if args.k <= 0:
        raise ValueError("--k must be positive")

    names_a, names_b_all, labels = load_dataset("ncvr_10k", args.data_path)
    names_b = names_b_all[: args.db_limit]
    for index in args.query_indices:
        if index < 0 or index >= len(names_a):
            raise ValueError(
                f"query index {index} is out of range 0..{len(names_a) - 1}"
            )

    started = time.perf_counter()
    offline_start = time.perf_counter()
    artifacts = prepare_party_b_offline(names_b, k_mode=args.k)
    offline_sec = time.perf_counter() - offline_start

    context_start = time.perf_counter()
    secret_context = create_ckks_context()
    secret_context.generate_relin_keys()
    public_context_bytes = serialize_public_context(secret_context)
    context_sec = time.perf_counter() - context_start

    rows = []
    for query_index in args.query_indices:
        query_name = names_a[query_index]
        round1_start = time.perf_counter()
        first_round_request, party_a_state, public_context_bytes = (
            prepare_encrypted_query_with_context(
                query_name,
                artifacts.scaler_mean,
                artifacts.scaler_scale,
                secret_context,
                public_context_bytes,
            )
        )
        encrypted_sim_scores = compare_to_centroids(
            first_round_request,
            artifacts.centroids,
        )
        second_round_request, cluster_debug = choose_cluster_and_build_request(
            encrypted_sim_scores,
            party_a_state,
            k=artifacts.centroids.shape[0],
        )
        round1_sec = time.perf_counter() - round1_start

        round2_start = time.perf_counter()
        encrypted_scores = column_wise_matching(
            artifacts.cluster_matrix,
            second_round_request,
            first_round_request.public_context_bytes,
            tau=args.tau,
        )
        match_result, match_debug = check_encrypted_scores_debug(
            encrypted_scores,
            party_a_state.secret_context,
            early_stop=not args.no_early_stop,
        )
        round2_sec = time.perf_counter() - round2_start

        db_index, matched_name, cluster_size = _first_hit_name(
            names_b,
            artifacts.cluster_assignments,
            cluster_debug.selected_cluster,
            match_debug.first_positive_column,
        )
        rows.append(
            {
                "query_index": query_index,
                "query_name": query_name,
                "expected_label": bool(labels[query_index]),
                "predicted_match": bool(match_result.catch),
                "selected_cluster": int(cluster_debug.selected_cluster),
                "selected_cluster_size": cluster_size,
                "first_positive_column": match_debug.first_positive_column,
                "matched_database_index": db_index,
                "matched_name": matched_name,
                "checked_columns": int(match_debug.checked_columns),
                "round1_sec": round1_sec,
                "round2_sec": round2_sec,
                "total_sec": round1_sec + round2_sec,
            }
        )

    total_sec = time.perf_counter() - started
    online_sec = sum(row["total_sec"] for row in rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "dataset": "ncvr_10k",
            "data_path": args.data_path,
            "db_limit": args.db_limit,
            "query_indices": args.query_indices,
            "k": args.k,
            "tau": args.tau,
            "early_stop": not args.no_early_stop,
        },
        "timing_sec": {
            "offline_preprocessing": offline_sec,
            "he_context_init": context_sec,
            "online_queries": online_sec,
            "total": total_sec,
        },
        "rows": rows,
    }
    json_path = output_dir / "demo_ncvr_matches.json"
    csv_path = output_dir / "demo_ncvr_matches.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    _print_table(
        rows,
        offline_sec=offline_sec,
        context_sec=context_sec,
        online_sec=online_sec,
        total_sec=total_sec,
    )
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
