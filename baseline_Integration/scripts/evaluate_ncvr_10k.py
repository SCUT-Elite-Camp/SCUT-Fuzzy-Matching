"""Run NCVR 10K evaluation and write metrics plus plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmark import benchmark
from evaluation.reporting import save_evaluation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real NCVR 10K benchmark and save metrics plots."
    )
    parser.add_argument(
        "--data-path",
        default=str(PROJECT_ROOT / "data"),
        help="Path to data/ or data/ncvr_10k.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "artifacts" / "evaluation" / "ncvr_10k"),
        help="Directory for JSON, CSV, and PNG outputs.",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=-1,
        help="Number of queries to run; -1 uses all NCVR 10K queries.",
    )
    parser.add_argument(
        "--db-limit",
        type=int,
        default=-1,
        help="Number of B-side records to use; -1 uses all 10000 records.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=50,
        help="Fixed cluster count; default 50 follows the paper's 10K NCVR setting. Use 0 to fall back to --k-mode.",
    )
    parser.add_argument(
        "--k-mode",
        default="sqrt",
        help='Cluster count mode only when --k is 0, for example "sqrt".',
    )
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stop during A-side final judgment.",
    )
    parser.add_argument(
        "--no-reuse-context",
        action="store_true",
        help="Regenerate a CKKS context for every query.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = {
        "dataset": "ncvr_10k",
        "data_path": args.data_path,
        "el_cluster": 200,
        "el_match": 50,
        "k": args.k,
        "k_mode": args.k if args.k > 0 else args.k_mode,
        "tau": args.tau,
        "query_limit": args.query_limit,
        "db_limit": args.db_limit,
        "use_mock": False,
        "early_stop": not args.no_early_stop,
        "reuse_context": not args.no_reuse_context,
    }
    result = benchmark(config)
    paths = save_evaluation_report(
        result,
        args.output_dir,
        prefix="ncvr_10k",
    )

    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    print(json.dumps(result.get("confusion", {}), indent=2, ensure_ascii=False))
    print("Saved outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
