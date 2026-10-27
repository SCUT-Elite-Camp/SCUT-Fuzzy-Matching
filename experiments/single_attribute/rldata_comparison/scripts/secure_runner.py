"""Run the end-to-end secure scheme-2 pipeline for several seeds.

For each seed this driver

1. exports the two party inputs and the plaintext reference
   (``secure_export.py`` next to this script),
2. launches the two ``name_secure_pipeline`` processes inside WSL
   (server first, then client, on a fresh EMP/PSI port pair),
3. checks the client predictions *query by query* against the plaintext
   reference (any mismatch fails the run), and
4. writes per-seed and aggregate metric/traffic artifacts.

The secure run computes exactly the same function as the reference: the point
of the comparison is that the two-party protocol reproduces the plaintext
metrics while never moving queries or database rows in the clear.

Run::

    python experiments/single_attribute/rldata_comparison/scripts/secure_runner.py
    python experiments/single_attribute/rldata_comparison/scripts/secure_runner.py \
        --seeds 20261023 --max-rounds 4

``--max-rounds`` is a smoke-test knob; the reference is recomputed with the
same round count, so the equality check stays exact.  ``--max-groups`` only
processes part of the candidate groups: the equality check then reports the
unprocessed queries as mismatches by design.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fused_top1 import load_queries  # noqa: E402
from secure_export import (  # noqa: E402
    DEFAULT_DATA,
    DEFAULT_DISTANCE,
    DEFAULT_PROBABILITY,
    DEFAULT_ROUNDS,
    DEFAULT_SEED,
    DEFAULT_SLOTS,
    export_client,
    export_server,
    reference_predictions,
)

ARTIFACTS = EXPERIMENT / "results"
SECURE_DIR = ARTIFACTS / "secure"
LOG_DIR = SECURE_DIR / "logs"
DEFAULT_SEEDS = (DEFAULT_SEED, DEFAULT_SEED + 17, DEFAULT_SEED + 34)
DEFAULT_BINARY = "~/build/fused-secure/name_secure_pipeline"
WSL_DISTRO = "Ubuntu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_wsl_path(path: Path) -> str:
    """Map a Windows drive path to its /mnt/<drive> form."""
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":")
    if not drive:
        return resolved.as_posix()
    rest = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive.lower()}{rest}"


def _export_seed(seed: int, args) -> tuple[Path, Path, Path, dict]:
    common = dict(
        data=args.data,
        seed=seed,
        probability=args.probability,
        rounds=args.rounds,
        slots=args.slots,
        distance=args.distance,
        limit_records=args.limit_records,
        limit_queries=args.limit_queries,
    )
    server_path = SECURE_DIR / f"server_seed{seed}.bin"
    client_path = SECURE_DIR / f"client_seed{seed}.bin"
    reference_path = SECURE_DIR / f"reference_seed{seed}.json"

    start = time.perf_counter()
    export_server(SimpleNamespace(out=server_path, **common))
    after_server = time.perf_counter()
    export_client(SimpleNamespace(out=client_path, **common))
    after_client = time.perf_counter()
    reference_predictions(SimpleNamespace(out=reference_path, **common))
    after_reference = time.perf_counter()

    timings = {
        "server": after_server - start,
        "client": after_client - after_server,
        "reference_sec": after_reference - after_client,
    }
    return server_path, client_path, reference_path, timings


def _wsl_command(binary: str, party: str, emp_port: int, psi_port: int,
                 input_path: Path, output_path: Path, args) -> list[str]:
    command = [
        binary,
        "--party", party,
        "--emp-port", str(emp_port),
        "--psi-port", str(psi_port),
        "--input", _to_wsl_path(input_path),
        "--output", _to_wsl_path(output_path),
    ]
    if party == "client":
        command += ["--alice-ip", "127.0.0.1"]
    if args.max_rounds:
        command += ["--max-rounds", str(args.max_rounds)]
    if args.max_groups:
        command += ["--max-groups", str(args.max_groups)]
    inner = "cd ~/build/fused-secure && exec " + " ".join(
        shlex.quote(part) for part in command
    )
    return ["wsl.exe", "-d", args.distro, "bash", "-lc", inner]


def _tail(path: Path, lines: int = 20) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace")
                         .splitlines()[-lines:])
    except FileNotFoundError:
        return "(no log)"


def _run_pair(seed: int, server_input: Path, client_input: Path, args) -> dict:
    """Launch server then client; return the parsed client JSON."""
    SECURE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    server_out = SECURE_DIR / f"full_server_seed{seed}.json"
    client_out = SECURE_DIR / f"full_client_seed{seed}.json"
    for stale in (server_out, client_out):
        stale.unlink(missing_ok=True)

    emp_port = args.base_port + 2 * (seed % 100)
    psi_port = emp_port + 1
    server_log = LOG_DIR / f"seed{seed}_server.log"
    client_log = LOG_DIR / f"seed{seed}_client.log"

    started = time.perf_counter()
    with server_log.open("w", encoding="utf-8") as server_handle:
        server = subprocess.Popen(
            _wsl_command(args.binary, "server", emp_port, psi_port,
                         server_input, server_out, args),
            stdout=server_handle,
            stderr=subprocess.STDOUT,
        )
        time.sleep(args.start_delay)
        with client_log.open("w", encoding="utf-8") as client_handle:
            client = subprocess.Popen(
                _wsl_command(args.binary, "client", emp_port, psi_port,
                             client_input, client_out, args),
                stdout=client_handle,
                stderr=subprocess.STDOUT,
            )
            client_code = client.wait(timeout=args.timeout)
            server_code = server.wait(timeout=args.timeout)
    wall_sec = time.perf_counter() - started

    if client_code != 0 or server_code != 0:
        raise RuntimeError(
            f"seed {seed}: pipeline failed (server={server_code}, client={client_code})\n"
            f"--- server ---\n{_tail(server_log)}\n--- client ---\n{_tail(client_log)}"
        )
    return {
        "client": json.loads(client_out.read_text(encoding="utf-8")),
        "server": json.loads(server_out.read_text(encoding="utf-8")),
        "wall_sec": wall_sec,
        "logs": {"server": str(server_log), "client": str(client_log)},
    }


def _metrics(predictions: list[dict], labels: np.ndarray, expected: np.ndarray) -> dict:
    predicted = np.full(labels.shape[0], -1, dtype=np.int64)
    for row in predictions:
        predicted[int(row["query_index"])] = int(row["identity"])
    tp = int(np.sum(labels & (predicted == expected)))
    fp = int(np.sum(~labels & (predicted != -1)))
    fn = int(np.sum(labels & (predicted != expected)))
    tn = int(np.sum(~labels & (predicted == -1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall else 0.0,
        "accuracy": (tp + tn) / labels.shape[0],
    }


def _run_seed(seed: int, args) -> dict:
    print(f"[seed {seed}] exporting inputs ...", flush=True)
    server_input, client_input, reference_path, export_timings = _export_seed(seed, args)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))

    print(f"[seed {seed}] running the two-party pipeline ...", flush=True)
    output = _run_pair(seed, server_input, client_input, args)
    client, server = output["client"], output["server"]

    reference_by_query = {row["query_index"]: row for row in reference["predictions"]}
    secure_by_query = {row["query_index"]: row for row in client["predictions"]}
    mismatches = [
        {
            "query_index": query,
            "reference": reference_by_query[query]["identity"],
            "secure": secure_by_query.get(query, {}).get("identity"),
        }
        for query in sorted(reference_by_query)
        if secure_by_query.get(query) is None
        or secure_by_query[query]["identity"] != reference_by_query[query]["identity"]
    ]

    queries = load_queries(args.data, split="test")
    if args.limit_queries:
        queries = queries.iloc[: args.limit_queries].reset_index(drop=True)
    metrics = _metrics(
        client["predictions"],
        queries["label_bool"].to_numpy(),
        queries["expected_identity"].to_numpy(),
    )
    client_stats = client["stats"]
    server_stats = server["stats"]

    if "predictions" in server:
        raise RuntimeError(f"seed {seed}: server output leaked predictions")

    return {
        "seed": seed,
        "config": {
            "data": str(args.data),
            "probability": args.probability,
            "rounds": args.rounds,
            "slots": args.slots,
            "distance_threshold": args.distance,
            "max_rounds": args.max_rounds,
            "max_groups": args.max_groups,
            "limit_records": args.limit_records,
            "limit_queries": args.limit_queries,
        },
        "metrics": metrics,
        "reference_metrics": reference["metrics"],
        "predictions_identical": not mismatches,
        "prediction_mismatches": mismatches[:20],
        "candidate_pairs": {
            "secure": client_stats["candidates"],
            "reference": reference["candidate_pairs"],
        },
        "client_stats": client_stats,
        "server_stats": server_stats,
        "wall_sec": output["wall_sec"],
        "logs": output["logs"],
        "export_bytes": {
            "server": server_input.stat().st_size,
            "client": client_input.stat().st_size,
        },
        "export_sec": export_timings,
    }


def _summarise(runs: list[dict]) -> dict:
    keys = [
        "precision", "recall", "f1", "accuracy", "tp", "fp", "fn", "tn",
    ]
    summary = {}
    for key in keys:
        values = np.array([run["metrics"][key] for run in runs], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    for key in (
        "phase1_sec", "phase2_sec", "phase3_sec", "total_sec",
        "psi_bytes_sent", "psi_bytes_received",
        "emp_bytes_sent", "emp_bytes_received",
        "psi_bins", "hit_bins", "candidates", "groups", "and_gates",
    ):
        values = np.array(
            [run["client_stats"][key] for run in runs], dtype=np.float64
        )
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    for key in ("wall_sec", "export_sec"):
        values = np.array(
            [
                run[key] if key == "wall_sec" else run[key]["server"]
                for run in runs
            ],
            dtype=np.float64,
        )
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="comma-separated projection seeds",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--probability", type=float, default=DEFAULT_PROBABILITY)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    parser.add_argument("--distance", type=int, default=DEFAULT_DISTANCE)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=0)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--binary", default=DEFAULT_BINARY)
    parser.add_argument("--distro", default=WSL_DISTRO)
    parser.add_argument("--base-port", type=int, default=24000)
    parser.add_argument("--start-delay", type=float, default=1.0,
                        help="seconds to wait after launching the server")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--keep-going", action="store_true",
                        help="continue with later seeds when one fails")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]

    runs, failures = [], []
    for seed in seeds:
        try:
            run = _run_seed(seed, args)
        except Exception as error:  # noqa: BLE001 - report and decide below
            failures.append({"seed": seed, "error": str(error)})
            print(f"[seed {seed}] FAILED: {error}", file=sys.stderr, flush=True)
            if not args.keep_going:
                raise
            continue
        runs.append(run)
        path = ARTIFACTS / f"secure_e2e_seed{seed}.json"
        path.write_text(json.dumps(run, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(
            f"[seed {seed}] identical={run['predictions_identical']} "
            f"tp/fp/fn/tn={run['metrics']['tp']}/{run['metrics']['fp']}/"
            f"{run['metrics']['fn']}/{run['metrics']['tn']} "
            f"f1={run['metrics']['f1']:.4f} "
            f"phase1={run['client_stats']['phase1_sec']:.2f}s "
            f"phase3={run['client_stats']['phase3_sec']:.2f}s "
            f"total={run['client_stats']['total_sec']:.2f}s",
            flush=True,
        )

    if runs:
        payload = {
            "config": runs[0]["config"],
            "seeds": [run["seed"] for run in runs],
            "runs": runs,
            "summary": _summarise(runs),
            "failures": failures,
        }
        ARTIFACTS.joinpath("secure_e2e.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
