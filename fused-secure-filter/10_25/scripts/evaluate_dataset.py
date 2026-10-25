"""Run the unified fuzzy-matching benchmark from one JSON configuration."""

# Project imports follow the direct-execution path bootstrap below.
# ruff: noqa: E402

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
        description="Run fuzzy matching for a configured database/query pair."
    )
    parser.add_argument("--config", required=True, help="Evaluation JSON file.")
    parser.add_argument(
        "--output-dir",
        help="Override the output directory in the JSON file.",
    )
    return parser.parse_args()


def load_evaluation_config(path: str | Path) -> tuple[dict, Path, str, str]:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        root = json.load(handle)
    if not isinstance(root, dict):
        raise ValueError("Evaluation JSON root must be an object")
    pipeline = root.get("dataset_pipeline")
    if not isinstance(pipeline, (dict, str)):
        raise ValueError("dataset_pipeline must be an object or JSON path")
    matching = root.get("matching", {})
    if not isinstance(matching, dict):
        raise ValueError("matching must be an object")

    benchmark_config = {
        "el_cluster": 200,
        "el_match": 50,
        "k": 50,
        "tau": 0.9,
        "he_batch_size": 0,
        "use_mock": False,
        "early_stop": True,
        "reuse_context": True,
        **matching,
        "dataset_pipeline": pipeline,
        "dataset_base_dir": str(config_path.parent),
    }
    output_dir = str(root.get("output_dir", "artifacts/evaluation/configured"))
    prefix = str(root.get("output_prefix", "fuzzy_matching"))
    return benchmark_config, config_path.parent, output_dir, prefix


def main() -> int:
    args = parse_args()
    config, base_dir, configured_output, prefix = load_evaluation_config(args.config)
    output_dir = Path(args.output_dir or configured_output)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir

    result = benchmark(config)
    paths = save_evaluation_report(result, output_dir, prefix=prefix)
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    print("Saved outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
