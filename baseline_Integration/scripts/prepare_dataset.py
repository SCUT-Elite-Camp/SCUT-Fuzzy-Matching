"""Prepare any configured database/query pair into the canonical CSV schema."""

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

from data_pipeline import build_prepared_dataset, write_prepared_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize a configured database/query pair."
    )
    parser.add_argument("--config", required=True, help="Pipeline JSON file.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for database.csv, queries.csv and manifest.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = build_prepared_dataset(args.config)
    paths = write_prepared_dataset(dataset, args.output_dir)
    print(json.dumps(dataset.manifest.to_dict(), indent=2, ensure_ascii=False))
    print("Saved outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
