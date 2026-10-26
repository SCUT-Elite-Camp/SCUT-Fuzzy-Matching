"""Compare cluster strategies with plaintext audits and real CKKS evaluation."""

# Project imports follow the direct-execution path bootstrap below.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ckks.tiling import make_slot_tile_layout, pack_candidate_tile
from clustering.kmeans_cosine import run_cosine_kmeans
from clustering.projection import project_normalized_signatures, rademacher_matrix
from config.params import COEFF_MOD_BIT_SIZES, POLY_MODULUS_DEGREE, SCALE
from data_pipeline import build_prepared_dataset, ncvr_10k_pipeline_config
from evaluation.benchmark import benchmark
from evaluation.metrics import compute_confusion_counts, compute_metrics
from minhash.encoder import batch_encode
from party_b.offline_prep import build_cluster_matrix, fit_scaler
from preprocessing.normalizer import l2_normalize


@dataclass(frozen=True)
class Strategy:
    name: str
    k: int
    projection_dim: int | None = None
    projection_seed: int = 42
    probes: int = 1
    skip_zero_blocks: bool = False


STRATEGIES = (
    Strategy("baseline_k50", k=50),
    Strategy("baseline_k50_sparse", k=50, skip_zero_blocks=True),
    Strategy("rp128_k50", k=50, projection_dim=128),
    Strategy("rp128_k50_sparse", k=50, projection_dim=128, skip_zero_blocks=True),
    Strategy("k100", k=100),
    Strategy("rp128_k100", k=100, projection_dim=128),
    Strategy("rp128_k100_p2", k=100, projection_dim=128, probes=2),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=str(PROJECT_ROOT / "data"))
    parser.add_argument(
        "--output-dir",
        default=str(
            PROJECT_ROOT
            / "artifacts"
            / "evaluation"
            / "cluster_strategy_comparison"
        ),
    )
    parser.add_argument("--query-limit", type=int, default=-1)
    parser.add_argument("--db-limit", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--fuzzy-ratio", type=float, default=0.3)
    parser.add_argument("--fuzzy-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=20260929)
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--strategies",
        default=",".join(strategy.name for strategy in STRATEGIES),
        help="Comma-separated subset of the registered strategies.",
    )
    parser.add_argument(
        "--skip-he",
        action="store_true",
        help="Run only the explicitly labelled plaintext proxy audit.",
    )
    parser.add_argument(
        "--reuse-raw",
        action="store_true",
        help="Rebuild summaries from existing raw HE JSON without rerunning HE.",
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    requested = set(args.strategies.split(","))
    unknown = requested - {strategy.name for strategy in STRATEGIES}
    if unknown:
        parser.error(f"Unknown strategies: {sorted(unknown)}")
    for strategy in STRATEGIES:
        if strategy.name in requested:
            make_slot_tile_layout(args.batch_size * min(strategy.probes, strategy.k))
    if args.skip_he and args.reuse_raw:
        parser.error("--skip-he and --reuse-raw are mutually exclusive")
    return args


def split_queries(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    development = []
    for label in (False, True):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        development.extend(indices[: len(indices) // 2].tolist())
    development = np.asarray(sorted(development), dtype=int)
    held_out = np.asarray(
        [index for index in range(len(labels)) if index not in set(development)],
        dtype=int,
    )
    return development, held_out


def wilson_interval(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.96
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def quality(
    predictions: np.ndarray,
    labels: np.ndarray,
    reference: np.ndarray,
    indices: np.ndarray,
) -> dict:
    predicted = np.asarray(predictions, dtype=bool)[indices]
    truth = np.asarray(labels, dtype=bool)[indices]
    reference_subset = np.asarray(reference, dtype=bool)[indices]
    reference_count = int(reference_subset.sum())
    confusion = compute_confusion_counts(predicted.tolist(), truth.tolist())
    reference_successes = int(np.sum(predicted[reference_subset]))
    return {
        "metrics": compute_metrics(predicted.tolist(), truth.tolist()),
        "confusion": confusion,
        "precision_wilson_95": wilson_interval(
            confusion["tp"], confusion["tp"] + confusion["fp"]
        ),
        "recall_wilson_95": wilson_interval(
            confusion["tp"], confusion["tp"] + confusion["fn"]
        ),
        "reference_positive_recall": (
            reference_successes / reference_count if reference_count else None
        ),
        "reference_positive_recall_wilson_95": wilson_interval(
            reference_successes, reference_count
        ),
        "reference_positive_count": reference_count,
    }


def paired_changes(predictions, baseline, labels, indices, query_ids) -> dict:
    predicted = np.asarray(predictions, dtype=bool)
    original = np.asarray(baseline, dtype=bool)
    truth = np.asarray(labels, dtype=bool)
    masks = {
        "recovered_true_positives": truth & ~original & predicted,
        "new_false_negatives": truth & original & ~predicted,
        "removed_false_positives": ~truth & original & ~predicted,
        "new_false_positives": ~truth & ~original & predicted,
    }
    return {
        key: [query_ids[index] for index in indices if mask[index]]
        for key, mask in masks.items()
    }


def audit_strategy(
    strategy: Strategy,
    normalized_b200: np.ndarray,
    normalized_q200: np.ndarray,
    normalized_b50: np.ndarray,
    normalized_q50: np.ndarray,
    labels: np.ndarray,
    reference: np.ndarray,
    development: np.ndarray,
    held_out: np.ndarray,
    tau: float,
    batch_size: int,
) -> tuple[dict, np.ndarray]:
    if strategy.projection_dim is None:
        _, cluster_b, mean, scale = fit_scaler(normalized_b200)
        cluster_q = (normalized_q200 - mean) / scale
        feature_dim = normalized_b200.shape[1]
    else:
        cluster_b = project_normalized_signatures(
            normalized_b200, strategy.projection_dim, strategy.projection_seed
        )
        cluster_q = project_normalized_signatures(
            normalized_q200, strategy.projection_dim, strategy.projection_seed
        )
        feature_dim = strategy.projection_dim

    centroids, assignments = run_cosine_kmeans(
        cluster_b, k=strategy.k, random_state=42
    )
    scores = cluster_q @ centroids.T
    order = np.argsort(scores, axis=1)[:, ::-1]
    candidate = np.zeros(len(labels), dtype=bool)
    for query_index, query_scores in enumerate(scores):
        selected = order[query_index, : strategy.probes]
        selected_rows = np.isin(assignments, selected)
        candidate[query_index] = np.any(
            (normalized_q50[query_index] @ normalized_b50[selected_rows].T) >= tau
        )

    sizes = np.bincount(assignments, minlength=len(centroids))
    cluster_matrix, max_size = build_cluster_matrix(
        normalized_b50, assignments, len(centroids)
    )
    slot_capacity = POLY_MODULUS_DEGREE // 2
    round1_width = slot_capacity // batch_size
    round2_layout = make_slot_tile_layout(batch_size * strategy.probes)
    round2_width = round2_layout.tile_width
    round1_tiles = (len(centroids) + round1_width - 1) // round1_width
    round2_tiles = (max_size + round2_width - 1) // round2_width
    total_selector_blocks = len(centroids) * 50 * round2_tiles
    nonzero_selector_blocks = 0
    zero_feature_anchors = 0
    for tile_index in range(round2_tiles):
        start = tile_index * round2_width
        valid_width = min(round2_width, max_size - start)
        tile = pack_candidate_tile(cluster_matrix, round2_layout, start, valid_width)
        nonzero_selector_blocks += int(np.count_nonzero(np.any(tile, axis=1)))
        zero_feature_anchors += int(np.count_nonzero(~np.any(tile, axis=(0, 1))))
    result = {
        "config": {
            "k": strategy.k,
            "projection_dim": strategy.projection_dim,
            "projection_seed": strategy.projection_seed,
            "probes": strategy.probes,
            "skip_zero_blocks": strategy.skip_zero_blocks,
        },
        "cluster_feature_dim": feature_dim,
        "clusters": len(centroids),
        "max_cluster_size": max_size,
        "mean_cluster_size": float(sizes.mean()),
        "cluster_matrix_padded_mb": (
            len(centroids) * max_size * normalized_b50.shape[1] * 8 / 1024**2
        ),
        "round1_query_ciphertexts": feature_dim,
        "round2_request_ciphertexts": 50 + len(centroids),
        "round1_tiles": round1_tiles,
        "round2_tiles": round2_tiles,
        "selector_ctpt_operations_unskipped": total_selector_blocks,
        "selector_ctpt_operations": (
            nonzero_selector_blocks + zero_feature_anchors
            if strategy.skip_zero_blocks
            else total_selector_blocks
        ),
        "skipped_zero_blocks": (
            total_selector_blocks - nonzero_selector_blocks - zero_feature_anchors
            if strategy.skip_zero_blocks
            else 0
        ),
        "ctct_operations": 50 * round2_tiles,
        "all": quality(candidate, labels, reference, np.arange(len(labels))),
        "development": quality(candidate, labels, reference, development),
        "held_out": quality(candidate, labels, reference, held_out),
    }
    return result, candidate


def audit_direct_projection(
    normalized_b200: np.ndarray,
    normalized_q200: np.ndarray,
    normalized_b50: np.ndarray,
    normalized_q50: np.ndarray,
    labels: np.ndarray,
    reference: np.ndarray,
    held_out: np.ndarray,
    tau: float,
) -> dict:
    _, standardized_b, mean, scale = fit_scaler(normalized_b200)
    standardized_q = (normalized_q200 - mean) / scale
    centroids, assignments = run_cosine_kmeans(
        standardized_b, k=50, random_state=42
    )
    original_top1 = np.argmax(standardized_q @ centroids.T, axis=1)
    results = {}
    for dimension in (32, 64, 96, 128):
        projection = rademacher_matrix(200, dimension, 42)
        scores = (standardized_q @ projection) @ (centroids @ projection).T
        selected = np.argmax(scores, axis=1)
        candidate = np.zeros(len(labels), dtype=bool)
        for query_index, cluster_index in enumerate(selected):
            selected_rows = assignments == cluster_index
            candidate[query_index] = np.any(
                (normalized_q50[query_index] @ normalized_b50[selected_rows].T) >= tau
            )
        results[f"dim_{dimension}"] = {
            "projection_seed": 42,
            "top1_agreement_with_original": float(
                np.mean(selected == original_top1)
            ),
            "all": quality(candidate, labels, reference, np.arange(len(labels))),
            "held_out": quality(candidate, labels, reference, held_out),
            "secure_measured": None,
            "decision": "Rejected at the plaintext gate; no HE run warranted.",
        }
    return results


def benchmark_config(args: argparse.Namespace, strategy: Strategy) -> dict:
    return {
        "dataset": "ncvr_10k",
        "data_path": args.data_path,
        "el_cluster": 200,
        "el_match": 50,
        "k": strategy.k,
        "k_mode": strategy.k,
        "tau": args.tau,
        "fuzzy_ratio": args.fuzzy_ratio,
        "fuzzy_seed": args.fuzzy_seed,
        "query_limit": args.query_limit,
        "db_limit": args.db_limit,
        "he_batch_size": args.batch_size,
        "projection_dim": strategy.projection_dim,
        "projection_seed": strategy.projection_seed,
        "probes": strategy.probes,
        "skip_zero_blocks": strategy.skip_zero_blocks,
        "use_mock": False,
        "early_stop": False,
        "reuse_context": True,
    }


def stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_reused_run(run: dict, expected_config: dict, manifest: dict) -> None:
    """Do not relabel an old run with today's requested evaluation settings."""
    config = run["config"]
    for key, value in expected_config.items():
        if key == "data_path":
            continue  # Data manifests are checked below; paths can be relocated.
        actual = config.get(key, False if key == "skip_zero_blocks" else None)
        if actual != value:
            raise ValueError(f"Raw HE config mismatch for {key}: {actual!r} != {value!r}")
    if run["dataset_manifest"] != manifest:
        raise ValueError("Raw HE dataset manifest does not match the current dataset")
    predictions = run["predictions"]
    if len(predictions) != manifest["output_query_rows"] or any(
        type(value) is not bool for value in predictions
    ):
        raise ValueError("Raw HE predictions must contain one boolean per query")


def main() -> int:
    args = parse_args()
    requested = set(args.strategies.split(","))
    strategies = [strategy for strategy in STRATEGIES if strategy.name in requested]
    if not strategies:
        raise ValueError(f"No registered strategies selected: {sorted(requested)}")
    slot_capacity = POLY_MODULUS_DEGREE // 2
    for strategy in strategies:
        effective_probes = min(strategy.probes, strategy.k)
        if args.batch_size * effective_probes > slot_capacity:
            raise ValueError(
                f"{strategy.name}: batch_size × effective probes exceeds "
                f"{slot_capacity}"
            )

    prepared = build_prepared_dataset(
        ncvr_10k_pipeline_config(
            args.data_path,
            database_limit=args.db_limit,
            query_limit=args.query_limit,
            fuzzy_ratio=args.fuzzy_ratio,
            fuzzy_seed=args.fuzzy_seed,
        )
    )
    names_a, names_b, labels_list = prepared.as_legacy_tuple()
    labels = np.asarray(labels_list, dtype=bool)
    development, held_out = split_queries(labels, args.split_seed)
    overlap = {
        prepared.queries[index].base_canonical_name for index in development
    } & {prepared.queries[index].base_canonical_name for index in held_out}
    if overlap:
        raise ValueError("Development and held-out queries share base names")

    raw_b200 = batch_encode(names_b, 200)
    raw_q200 = batch_encode(names_a, 200)
    normalized_b200 = l2_normalize(raw_b200)
    normalized_q200 = l2_normalize(raw_q200)
    normalized_b50 = l2_normalize(raw_b200[:, :50])
    normalized_q50 = l2_normalize(raw_q200[:, :50])
    reference = np.asarray(
        [np.any(row >= args.tau) for row in normalized_q50 @ normalized_b50.T],
        dtype=bool,
    )
    direct_projection_audit = audit_direct_projection(
        normalized_b200,
        normalized_q200,
        normalized_b50,
        normalized_q50,
        labels,
        reference,
        held_out,
        args.tau,
    )

    audits = {}
    audit_predictions = {}
    for strategy in strategies:
        print(f"[plaintext audit] {strategy.name}", flush=True)
        audit, candidate = audit_strategy(
            strategy,
            normalized_b200,
            normalized_q200,
            normalized_b50,
            normalized_q50,
            labels,
            reference,
            development,
            held_out,
            args.tau,
            args.batch_size,
        )
        audits[strategy.name] = audit
        audit_predictions[strategy.name] = candidate

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = {strategy.name: [] for strategy in strategies}
    if args.reuse_raw:
        for strategy in strategies:
            for repeat in range(args.repeats):
                raw_path = (
                    output_dir / "raw" / f"{strategy.name}_run_{repeat + 1}.json"
                )
                if not raw_path.exists():
                    raise FileNotFoundError(f"Missing raw HE result: {raw_path}")
                run = json.loads(raw_path.read_text(encoding="utf-8"))
                validate_reused_run(
                    run, benchmark_config(args, strategy), prepared.manifest.to_dict()
                )
                runs[strategy.name].append(run)
    elif not args.skip_he:
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        for strategy in strategies:
            for warmup_index in range(args.warmup):
                print(f"[HE warmup {warmup_index + 1}] {strategy.name}", flush=True)
                benchmark(benchmark_config(args, strategy))
        for repeat in range(args.repeats):
            ordered = strategies if repeat % 2 == 0 else list(reversed(strategies))
            for strategy in ordered:
                print(f"[HE run {repeat + 1}] {strategy.name}", flush=True)
                result = benchmark(benchmark_config(args, strategy))
                runs[strategy.name].append(result)
                (raw_dir / f"{strategy.name}_run_{repeat + 1}.json").write_text(
                    json.dumps(result, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

    summaries = {}
    for strategy in strategies:
        strategy_runs = runs[strategy.name]
        summary = {"plaintext_audit": audits[strategy.name]}
        if strategy_runs:
            stable = all(
                run["predictions"] == strategy_runs[0]["predictions"]
                for run in strategy_runs[1:]
            )
            summary["secure_measured"] = {
                "runs": [
                    {
                        "raw_file": f"raw/{strategy.name}_run_{repeat + 1}.json",
                        "raw_sha256": file_sha256(
                            output_dir / "raw" / f"{strategy.name}_run_{repeat + 1}.json"
                        ),
                        "all": quality(
                            run["predictions"], labels, reference, np.arange(len(labels))
                        ),
                        "held_out": quality(
                            run["predictions"], labels, reference, held_out
                        ),
                        "plaintext_proxy_comparison": {
                            "prediction_agreement": float(
                                np.mean(
                                    np.asarray(run["predictions"], dtype=bool)
                                    == audit_predictions[strategy.name]
                                )
                            ),
                            "different_predictions": int(
                                np.sum(
                                    np.asarray(run["predictions"], dtype=bool)
                                    != audit_predictions[strategy.name]
                                )
                            ),
                        },
                        "protocol": run["protocol"],
                    }
                    for repeat, run in enumerate(strategy_runs)
                ],
                "prediction_stable_across_repeats": stable,
                "timing_sec": {
                    key: stats([run["timing_sec"][key] for run in strategy_runs])
                    for key in ("offline", "avg_round1", "avg_round2", "online_total")
                },
                "communication_mb": {
                    key: stats(
                        [run["communication_mb"][key] for run in strategy_runs]
                    )
                    for key in (
                        "sent",
                        "recv",
                        "total",
                        "public_context_setup",
                        "total_without_public_context",
                    )
                },
            }
        summaries[strategy.name] = summary

    baseline_name = "baseline_k50"
    comparisons = {}
    if baseline_name in summaries and runs[baseline_name]:
        baseline_secure = summaries[baseline_name]["secure_measured"]
        baseline_online = baseline_secure["timing_sec"]["online_total"]["median"]
        baseline_comm = baseline_secure["communication_mb"]["total"]["median"]
        baseline_f1 = float(
            np.median(
                [run["held_out"]["metrics"]["f1"] for run in baseline_secure["runs"]]
            )
        )
        for strategy in strategies:
            if not runs[strategy.name]:
                continue
            secure = summaries[strategy.name]["secure_measured"]
            f1 = float(
                np.median(
                    [run["held_out"]["metrics"]["f1"] for run in secure["runs"]]
                )
            )
            comparisons[strategy.name] = {
                "online_ratio_vs_baseline": (
                    secure["timing_sec"]["online_total"]["median"] / baseline_online
                ),
                "communication_ratio_vs_baseline": (
                    secure["communication_mb"]["total"]["median"] / baseline_comm
                ),
                "held_out_f1_delta_vs_baseline": f1 - baseline_f1,
                "paired_prediction_changes": [
                    {
                        split: paired_changes(
                            run["predictions"],
                            runs[baseline_name][repeat]["predictions"],
                            labels,
                            indices,
                            [query.query_id for query in prepared.queries],
                        )
                        for split, indices in (
                            ("all", np.arange(len(labels))),
                            ("development", development),
                            ("held_out", held_out),
                        )
                    }
                    for repeat, run in enumerate(runs[strategy.name])
                ],
            }

    per_query_path = output_dir / "per_query.csv"
    with per_query_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["query_id", "split", "label", "full_database_reference"]
        for strategy in strategies:
            fieldnames.append(f"{strategy.name}_audit")
            for repeat in range(len(runs[strategy.name])):
                fieldnames.append(f"{strategy.name}_he_run_{repeat + 1}")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, query in enumerate(prepared.queries):
            split = "development" if index in set(development) else "held_out"
            row = {
                "query_id": query.query_id,
                "split": split,
                "label": bool(labels[index]),
                "full_database_reference": bool(reference[index]),
            }
            for strategy in strategies:
                row[f"{strategy.name}_audit"] = bool(
                    audit_predictions[strategy.name][index]
                )
                for repeat, run in enumerate(runs[strategy.name], start=1):
                    row[f"{strategy.name}_he_run_{repeat}"] = run["predictions"][index]
            writer.writerow(row)

    data_root = Path(args.data_path)
    if not (data_root / "ncvr_10k_database.csv").exists():
        data_root = data_root / "ncvr_10k"
    summary = {
        "evidence_semantics": {
            "plaintext_audit": "Plaintext proxy: cluster selection plus exact 50-d cosine threshold; no CKKS, masks, or wire boundary.",
            "secure_measured": "Real TenSEAL CKKS, bytes boundaries, no mock, no early stop.",
            "timing": "Single-machine wall time, not a WAN estimate.",
            "split": "Development is for parameter selection; this run does not tune on held_out.",
            "prior_full_dataset_exploration": "Post-hoc exploration, not independent confirmation.",
            "direct_projection_ablation": "Low-dimensional approximation of the original centroid scores; rejected after plaintext audit and never presented as a secure result.",
            "memory": "Padded cluster matrix is theoretical storage, not process RSS.",
            "reference_positive_recall": "Query-level agreement on full-database threshold-positive queries, including business-label negatives; not recall of particular reference record IDs.",
            "raw_reuse": "Config, manifest and prediction length are checked. Current data hashes cannot retrospectively prove historical raw data or source identity.",
        },
        "environment": {
            "scope": "Environment at summary assembly; historical raw runs did not capture per-run dependency or source hashes.",
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": version("numpy"),
            "scikit_learn": version("scikit-learn"),
            "tenseal": version("tenseal"),
            "git_head": git_value(["rev-parse", "HEAD"]),
            "git_worktree_dirty": bool(git_value(["status", "--porcelain"])),
        },
        "data": {
            "manifest": prepared.manifest.to_dict(),
            "database_sha256": file_sha256(data_root / "ncvr_10k_database.csv"),
            "queries_sha256": file_sha256(data_root / "ncvr_10k_queries.csv"),
            "development_queries": len(development),
            "held_out_queries": len(held_out),
        },
        "settings": {
            "args": vars(args),
            "ckks": {
                "poly_modulus_degree": POLY_MODULUS_DEGREE,
                "scale": SCALE,
                "coeff_mod_bit_sizes": COEFF_MOD_BIT_SIZES,
            },
        },
        "full_database_reference": {
            "all": quality(reference, labels, reference, np.arange(len(labels))),
            "development": quality(reference, labels, reference, development),
            "held_out": quality(reference, labels, reference, held_out),
        },
        "strategies": summaries,
        "direct_projection_ablation": direct_projection_audit,
        "comparisons_vs_baseline": comparisons,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nPlaintext proxy and real HE summaries (held-out)")
    for strategy in strategies:
        audit = audits[strategy.name]["held_out"]
        line = (
            f"{strategy.name} audit: recall={audit['metrics']['recall']:.4f}, "
            f"F1={audit['metrics']['f1']:.4f}, "
            f"max_cluster={audits[strategy.name]['max_cluster_size']}"
        )
        if runs[strategy.name]:
            secure = summaries[strategy.name]["secure_measured"]
            secure_recall = np.median(
                [run["held_out"]["metrics"]["recall"] for run in secure["runs"]]
            )
            secure_f1 = np.median(
                [run["held_out"]["metrics"]["f1"] for run in secure["runs"]]
            )
            line += (
                f"\n  secure: recall={secure_recall:.4f}, F1={secure_f1:.4f}"
                f", online={secure['timing_sec']['online_total']['median']:.3f}s"
                f", communication={secure['communication_mb']['total']['median']:.3f}MB"
            )
        print(line)
    print(f"Saved: {summary_path}")
    print(f"Saved: {per_query_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
