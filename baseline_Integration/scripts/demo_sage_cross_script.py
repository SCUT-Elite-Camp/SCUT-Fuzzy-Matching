"""Presentation-oriented terminal demo for SAGE cross-script name matching."""

# Project imports follow the direct-execution path bootstrap below.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
from wcwidth import wcswidth

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline import PreparedDataset, build_prepared_dataset
from ckks.keys import serialize_public_context
from config.params import SIMILARITY_THRESHOLD
from party_a.local_prep import create_ckks_context, prepare_tiled_query_batch
from party_a.online_querier import (
    check_tiled_score_batch_debug,
    choose_clusters_and_build_tiled_request,
)
from party_b.offline_prep import prepare_party_b_offline
from party_b.online_responder import (
    compare_tiled_batch_to_centroids,
    tiled_batch_matching,
)
from protocol.transport import (
    serialize_tiled_first_round_request,
    serialize_tiled_second_round_request,
)

DEFAULT_QUERY_IDS = (
    "hu-accent-fold",
    "ar-romanized",
    "zh-romanized",
    "my-width-normalization",
    "zh-latin-mixed",
    "negative-latin",
    "negative-arabic",
    "negative-han",
    "negative-myanmar",
)


def _parse_query_ids(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return DEFAULT_QUERY_IDS
    query_ids = tuple(part.strip() for part in value.split(",") if part.strip())
    if not query_ids:
        raise argparse.ArgumentTypeError("query ids cannot be empty")
    return query_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show SAGE Unicode cleaning, Latin transliteration variants, and "
            "a tiny real encrypted batch query."
        )
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "examples" / "sage_pipeline.json"),
    )
    parser.add_argument(
        "--query-ids",
        type=_parse_query_ids,
        default=DEFAULT_QUERY_IDS,
        help="Comma-separated validation query IDs, or 'all'. Default: all 9.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "artifacts" / "demo" / "sage_cross_script"),
    )
    parser.add_argument("--no-color", action="store_true")
    return parser.parse_args()


def _select_queries(prepared: PreparedDataset, query_ids: tuple[str, ...]):
    query_by_id = {query.query_id: query for query in prepared.queries}
    missing = [query_id for query_id in query_ids if query_id not in query_by_id]
    if missing:
        raise ValueError(f"Unknown query IDs: {missing}")
    return [query_by_id[query_id] for query_id in query_ids]


def _build_demo_database(prepared: PreparedDataset, queries):
    database_by_id = {record.record_id: record for record in prepared.database}
    record_ids = sorted(
        {
            record_id
            for query in queries
            for record_id in query.expected_record_ids
        }
    )
    entries: list[dict] = []
    seen_values: set[str] = set()
    for record_id in record_ids:
        record = database_by_id[record_id]
        candidates = [("canonical", record.canonical_name), *record.variants.items()]
        for variant_name, value in candidates:
            if not value or value in seen_values:
                continue
            seen_values.add(value)
            entries.append(
                {
                    "search_name": value,
                    "variant": variant_name,
                    "record_id": record.record_id,
                    "source_name": record.raw_name,
                    "source_script": record.script,
                    "country": record.country,
                }
            )
    return entries


def _resolve_hit(
    artifacts,
    selected_clusters: np.ndarray,
    first_positive_columns: np.ndarray,
    entries: list[dict],
    query_index: int,
) -> dict | None:
    cluster = int(selected_clusters[query_index])
    column = int(first_positive_columns[query_index])
    if column < 0:
        return None
    members = np.where(artifacts.cluster_assignments == cluster)[0]
    if column >= len(members):
        return None
    return entries[int(members[column])]


def run_demo(config: str | Path, query_ids: tuple[str, ...]) -> dict:
    preparation_start = time.perf_counter()
    prepared = build_prepared_dataset(config)
    preparation_sec = time.perf_counter() - preparation_start
    queries = _select_queries(prepared, query_ids)
    entries = _build_demo_database(prepared, queries)

    database_names = [entry["search_name"] for entry in entries]
    query_names = [query.canonical_name for query in queries]
    timings: dict[str, float] = {"dataset_preparation": preparation_sec}

    started = time.perf_counter()
    step_start = time.perf_counter()
    artifacts = prepare_party_b_offline(
        database_names,
        k_mode="sqrt",
        random_state=42,
    )
    timings["offline_index_and_clustering"] = time.perf_counter() - step_start

    step_start = time.perf_counter()
    secret_context = create_ckks_context()
    secret_context.generate_relin_keys()
    public_context_bytes = serialize_public_context(secret_context)
    first_request, party_a_state = prepare_tiled_query_batch(
        query_names,
        artifacts.scaler_mean,
        artifacts.scaler_scale,
        context=secret_context,
        public_context_bytes=public_context_bytes,
    )
    wire_first_request = serialize_tiled_first_round_request(first_request)
    timings["query_encode_and_encrypt"] = time.perf_counter() - step_start

    step_start = time.perf_counter()
    encrypted_centroid_scores = compare_tiled_batch_to_centroids(
        wire_first_request,
        artifacts.centroids,
        serialize_output=True,
    )
    timings["encrypted_centroid_comparison"] = time.perf_counter() - step_start

    step_start = time.perf_counter()
    second_request, cluster_debug = choose_clusters_and_build_tiled_request(
        encrypted_centroid_scores,
        party_a_state,
        k=artifacts.centroids.shape[0],
    )
    wire_second_request = serialize_tiled_second_round_request(second_request)
    timings["cluster_selection"] = time.perf_counter() - step_start

    step_start = time.perf_counter()
    encrypted_match_tiles = list(
        tiled_batch_matching(
            artifacts.cluster_matrix,
            wire_second_request,
            public_context_bytes,
            tau=SIMILARITY_THRESHOLD,
            serialize_output=True,
        )
    )
    timings["masked_column_matching"] = time.perf_counter() - step_start

    step_start = time.perf_counter()
    match_result, match_debug = check_tiled_score_batch_debug(
        encrypted_match_tiles,
        secret_context,
        layout=party_a_state.layout,
        logical_width=artifacts.cluster_matrix.shape[1],
        early_stop=False,
    )
    timings["decrypt_and_judge"] = time.perf_counter() - step_start
    timings["real_he_batch"] = time.perf_counter() - started
    timings["total"] = preparation_sec + timings["real_he_batch"]

    sent_bytes = (
        len(public_context_bytes)
        + sum(len(value) for value in wire_first_request.encrypted_query_200)
        + sum(len(value) for value in wire_second_request.encrypted_query_50)
        + sum(len(value) for value in wire_second_request.encrypted_selectors)
    )
    received_bytes = sum(len(value) for value in encrypted_centroid_scores) + sum(
        len(value) for value in encrypted_match_tiles
    )

    rows = []
    per_query_ms = (
        timings["real_he_batch"] * 1000 / len(queries) if queries else 0.0
    )
    for index, query in enumerate(queries):
        predicted = bool(match_result.catches[index])
        hit = _resolve_hit(
            artifacts,
            cluster_debug.selected_clusters,
            match_debug.first_positive_columns,
            entries,
            index,
        )
        rows.append(
            {
                "query_id": query.query_id,
                "raw_query": query.raw_name,
                "canonical_query": query.canonical_name,
                "query_script": query.script,
                "expected": query.label,
                "predicted": predicted,
                "passed": predicted == query.label,
                "selected_cluster": int(
                    cluster_debug.selected_clusters[index]
                ),
                "checked_columns": int(match_debug.logical_columns_checked),
                "first_positive_column": int(
                    match_debug.first_positive_columns[index]
                ),
                "matched_search_name": hit["search_name"] if hit else None,
                "matched_variant": hit["variant"] if hit else None,
                "matched_source_name": hit["source_name"] if hit else None,
                "time_ms": per_query_ms,
            }
        )
    return {
        "dataset": prepared.manifest.dataset_name,
        "full_database_records": len(prepared.database),
        "full_matching_entries": len(prepared.matching_database_names),
        "source_rows": prepared.manifest.source_stats.get("source_rows"),
        "script_counts": dict(prepared.manifest.database_script_counts),
        "demo_database_entries": entries,
        "protocol": {
            "signature_dimensions": [200, 50],
            "clusters": int(artifacts.centroids.shape[0]),
            "cluster_sizes": np.bincount(
                artifacts.cluster_assignments,
                minlength=artifacts.centroids.shape[0],
            ).tolist(),
            "max_cluster_size": int(artifacts.max_size),
            "batch_size": len(queries),
            "slot_tile_width": int(party_a_state.layout.tile_width),
            "active_slots": int(party_a_state.layout.active_slots),
            "round1_ciphertexts": len(wire_first_request.encrypted_query_200),
            "centroid_score_ciphertexts": len(encrypted_centroid_scores),
            "round2_query_ciphertexts": len(wire_second_request.encrypted_query_50),
            "selector_ciphertexts": len(wire_second_request.encrypted_selectors),
            "match_score_ciphertexts": len(encrypted_match_tiles),
            "threshold": SIMILARITY_THRESHOLD,
            "sent_mb": sent_bytes / 1024**2,
            "received_mb": received_bytes / 1024**2,
        },
        "timing_sec": timings,
        "rows": rows,
        "all_passed": all(row["passed"] for row in rows),
    }


# =============================================================================
# Terminal presentation helpers
# =============================================================================

class Colors:
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"

    @classmethod
    def disable(cls) -> None:
        for name in ("BLUE", "CYAN", "GREEN", "YELLOW", "RED", "BOLD", "DIM"):
            setattr(cls, name, "")
        cls.END = ""


def _c(value: object, color: str) -> str:
    return f"{color}{value}{Colors.END}"


def _banner(title: str, width: int = 96) -> None:
    print()
    print(_c("=" * width, Colors.CYAN))
    print(_c(f"  {title}", Colors.BOLD + Colors.CYAN))
    print(_c("=" * width, Colors.CYAN))


def _phase(number: int, name: str, party: str, width: int = 96) -> None:
    print()
    print("-" * width)
    color = Colors.GREEN if party == "B" else Colors.BLUE
    print(
        f"  {_c(f'[Party {party}]', Colors.BOLD + color)} "
        f"{_c(f'Step {number}: {name}', Colors.BOLD + Colors.YELLOW)}"
    )
    print("-" * width)


def _info(text: str) -> None:
    print(f"  {_c('i', Colors.CYAN)} {text}")


def _ok(text: str) -> None:
    print(f"  {_c('✓', Colors.GREEN)} {text}")


def _timing(text: str, seconds: float) -> None:
    print(f"  {_c('⏱', Colors.YELLOW)} {text}: {_c(f'{seconds:.3f}s', Colors.GREEN)}")


def _kv(key: str, value: object) -> None:
    print(f"  {_c(key, Colors.DIM)}: {value}")


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_ANSI_PREFIX_RE = re.compile(r"^(?:\x1b\[[0-9;]*m)+")
_ANSI_SUFFIX_RE = re.compile(r"(?:\x1b\[[0-9;]*m)+$")
_LRI = "\u2066"
_PDI = "\u2069"
_BIDI_CLASSES = {"R", "AL", "RLE", "RLO", "RLI"}


def _terminal_width(value: object) -> int:
    text = _ANSI_RE.sub("", str(value))
    return max(0, wcswidth(text))


def _clip_plain(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if _terminal_width(text) <= width:
        return text

    ellipsis = "…"
    budget = max(0, width - _terminal_width(ellipsis))
    clipped: list[str] = []
    for char in text:
        candidate = "".join(clipped) + char
        if _terminal_width(candidate) == 0 and not clipped:
            continue
        if _terminal_width(candidate) > budget:
            break
        clipped.append(char)
    return "".join(clipped) + ellipsis


def _preserve_cell_color(raw: str, clipped_plain: str) -> str:
    prefix_match = _ANSI_PREFIX_RE.match(raw)
    suffix_match = _ANSI_SUFFIX_RE.search(raw)
    plain = _ANSI_RE.sub("", raw)
    if (
        prefix_match is not None
        and suffix_match is not None
        and prefix_match.group(0) + plain + suffix_match.group(0) == raw
    ):
        return prefix_match.group(0) + clipped_plain + suffix_match.group(0)
    return clipped_plain


def _isolate_bidi(text: str) -> str:
    plain = _ANSI_RE.sub("", text)
    if any(unicodedata.bidirectional(char) in _BIDI_CLASSES for char in plain):
        return _LRI + text + _PDI
    return text


def _clip(value: object, width: int) -> str:
    raw = str(value)
    plain = _ANSI_RE.sub("", raw)
    clipped_plain = _clip_plain(plain, width)
    if clipped_plain == plain:
        return raw
    return _preserve_cell_color(raw, clipped_plain)


def _display_query(row: dict) -> str:
    """Show the normalized value sent to the protocol, not the fixture ID."""
    return row["canonical_query"]


def _pad(value: object, width: int) -> str:
    text = _isolate_bidi(_clip(value, width))
    return text + " " * max(0, width - _terminal_width(text))


def _table(headers: list[str], rows: list[list[object]], widths: list[int]) -> None:
    print(
        "    "
        + " ".join(
            _c(_pad(header, width), Colors.DIM)
            for header, width in zip(headers, widths)
        ).rstrip()
    )
    print("    " + "-" * (sum(widths) + len(widths) - 1))
    for row in rows:
        print(
            "    "
            + " ".join(
                _pad(value, width)
                for value, width in zip(row, widths)
            ).rstrip()
        )


def _print_demo(result: dict) -> None:
    timing = result["timing_sec"]
    protocol = result["protocol"]
    rows = result["rows"]

    _banner("SAGE MULTILINGUAL CROSS-SCRIPT FUZZY MATCHING")
    _info(
        "The source SAGE database is cleaned once, then native and Latin "
        "variants are indexed for the same encrypted protocol."
    )
    _kv("Source rows", f"{result['source_rows']:,}")
    _kv("Prepared records", f"{result['full_database_records']:,}")
    _kv("Searchable names", f"{result['full_matching_entries']:,}")
    _kv("Demo queries", len(rows))

    _phase(1, "Clean, Transliterate & Index (Offline)", "B")
    _info("Unicode NFKC -> casefold -> punctuation cleanup -> searchable variants")
    grouped: dict[str, list[dict]] = {}
    for entry in result["demo_database_entries"]:
        grouped.setdefault(entry["record_id"], []).append(entry)
    for entries in grouped.values():
        first = entries[0]
        print(
            f"\n    {_c(first['source_name'], Colors.BOLD)} "
            f"{_c(f'({first["source_script"]}, {first["country"]})', Colors.DIM)}"
        )
        for entry in entries:
            print(
                f"      {_pad(_c(entry['variant'], Colors.CYAN), 28)} "
                f"{entry['search_name']}"
            )
    print()
    _kv("Demo database entries", len(result["demo_database_entries"]))
    _kv("Signature dimensions", "EL=200 cluster / EL=50 match")
    _kv("Clusters", protocol["clusters"])
    _kv("Cluster matrix", f"({protocol['clusters']}, {protocol['max_cluster_size']}, 50)")
    _timing("Offline indexing", timing["offline_index_and_clustering"])

    _phase(2, "Query Prepare & Encrypt", "A")
    _info("All selected queries are normalized and packed into one tiled HE batch")
    _table(
        ["Raw Query", "Canonical Query", "Script"],
        [
            [
                row["raw_query"],
                row["canonical_query"],
                row["query_script"],
            ]
            for row in rows
        ],
        [42, 42, 10],
    )
    print()
    _kv("Slot layout", f"{protocol['batch_size']} queries × {protocol['slot_tile_width']} candidates")
    _kv("Round-1 ciphertexts", protocol["round1_ciphertexts"])
    _timing("Encode + encrypt", timing["query_encode_and_encrypt"])
    _ok("Serialized Q200 request is bytes-only before reaching Party B")

    _phase(3, "Compare to Encrypted Centroids", "B")
    _kv("Centroid score ciphertexts", protocol["centroid_score_ciphertexts"])
    _timing("Encrypted centroid comparison", timing["encrypted_centroid_comparison"])
    _info("Party B sees encrypted query vectors and returns encrypted scores")

    _phase(4, "Decrypt Scores & Select Clusters", "A")
    _table(
        ["Query", "Cluster", "Prediction basis"],
        [
            [_display_query(row), row["selected_cluster"], "argmax decrypted score"]
            for row in rows
        ],
        [28, 10, 28],
    )
    print()
    _kv("Selector ciphertexts", protocol["selector_ciphertexts"])
    _timing("Cluster selection", timing["cluster_selection"])
    _info("Party A re-encrypts one-hot selectors; B cannot learn the selected cluster")

    _phase(5, "Selector×Matrix | Random Mask | Encrypted Similarity", "B")
    _kv("Cluster matrix", f"({protocol['clusters']}, {protocol['max_cluster_size']}, 50)")
    _kv("Q50 ciphertexts", protocol["round2_query_ciphertexts"])
    _kv("Match score ciphertexts", protocol["match_score_ciphertexts"])
    _table(
        ["Query", "Cluster", "Columns", "Masks"],
        [
            [_display_query(row), row["selected_cluster"], row["checked_columns"], "applied"]
            for row in rows
        ],
        [28, 10, 10, 12],
    )
    print()
    _timing("Masked column matching", timing["masked_column_matching"])
    _ok("Encrypted scores ready for all queries")
    _info("Independent positive masks prevent B from learning raw similarities")

    _phase(6, "Check Sim (Decrypt & Judge)", "A")
    _kv("Threshold τ", protocol["threshold"])
    _table(
        ["Query", "Match", "Checked", "Hit Col", "Hit Variant"],
        [
            [
                _display_query(row),
                _c("Y", Colors.GREEN) if row["predicted"] else _c("N", Colors.RED),
                row["checked_columns"],
                row["first_positive_column"] if row["first_positive_column"] >= 0 else "-",
                row["matched_variant"] or "-",
            ]
            for row in rows
        ],
        [28, 8, 10, 10, 30],
    )
    print()
    _timing("Match checking", timing["decrypt_and_judge"])

    print()
    print(_c("=" * 96, Colors.CYAN))
    print(_c("  FINAL RESULTS SUMMARY", Colors.BOLD + Colors.YELLOW))
    print(_c("=" * 96, Colors.CYAN))
    _table(
        ["Query", "Label", "Pred", "Cluster", "Correct", "Time(ms)", "Hit Source"],
        [
            [
                _display_query(row),
                "Y" if row["expected"] else "N",
                _c("Y", Colors.GREEN) if row["predicted"] else _c("N", Colors.RED),
                row["selected_cluster"],
                _c("✓", Colors.GREEN) if row["passed"] else _c("✗", Colors.RED),
                f"{row['time_ms']:.1f}",
                row["matched_source_name"] or "-",
            ]
            for row in rows
        ],
        [28, 6, 6, 8, 8, 10, 24],
    )
    correct = sum(row["passed"] for row in rows)
    accuracy = correct / len(rows) * 100 if rows else 0.0
    print()
    _ok(f"Accuracy: {correct}/{len(rows)} = {accuracy:.1f}%")
    _timing("Total query time", timing["real_he_batch"])
    _timing("Offline prep time", timing["offline_index_and_clustering"])
    _timing("Dataset clean time", timing["dataset_preparation"])
    _kv("A→B payload", f"{protocol['sent_mb']:.1f} MiB")
    _kv("B→A payload", f"{protocol['received_mb']:.1f} MiB")
    print(
        _c(
            "  Demo-only details: cluster IDs and hit variants are decrypted locally; "
            "wire payloads stay serialized and encrypted.",
            Colors.DIM,
        )
    )


def _save_outputs(
    result: dict,
    output_dir: str | Path,
) -> tuple[Path | None, Path | None]:
    root = Path(output_dir)
    json_path = root / "demo_sage_cross_script.json"
    csv_path = root / "demo_sage_cross_script.csv"
    try:
        root.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
            writer.writeheader()
            writer.writerows(result["rows"])
    except PermissionError as exc:
        print(f"Warning: could not save demo artifacts to {root}: {exc}")
        return None, None
    return json_path, csv_path


def main() -> int:
    args = parse_args()
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()
    result = run_demo(args.config, args.query_ids)
    _print_demo(result)
    json_path, csv_path = _save_outputs(result, args.output_dir)
    if json_path is not None and csv_path is not None:
        print(f"Saved JSON: {json_path}")
        print(f"Saved CSV:  {csv_path}")
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
