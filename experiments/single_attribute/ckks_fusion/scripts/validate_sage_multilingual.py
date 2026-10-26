"""Run a small real-HE validation over SAGE multilingual query examples."""

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

from data_pipeline import PreparedDataset, build_prepared_dataset
from protocol.orchestrator import run_batch_query_protocol


def validate_prepared_dataset(prepared: PreparedDataset) -> dict:
    database_by_id = {record.record_id: record for record in prepared.database}
    positive_ids = sorted(
        {
            record_id
            for query in prepared.queries
            for record_id in query.expected_record_ids
        }
    )
    validation_database: list[str] = []
    seen_names: set[str] = set()
    for record_id in positive_ids:
        record = database_by_id[record_id]
        for value in (record.canonical_name, *record.variants.values()):
            if value and value not in seen_names:
                seen_names.add(value)
                validation_database.append(value)
    query_names = [query.canonical_name for query in prepared.queries]
    labels = [query.label for query in prepared.queries]

    run = run_batch_query_protocol(
        validation_database,
        query_names,
        random_state=42,
        early_stop=False,
        serialize_communication=True,
    )
    predictions = [bool(value) for value in run.batch_match_result.catches]
    cases = [
        {
            "query_id": query.query_id,
            "raw_name": query.raw_name,
            "canonical_name": query.canonical_name,
            "script": query.script,
            "label": query.label,
            "predicted": prediction,
            "passed": prediction == query.label,
        }
        for query, prediction in zip(prepared.queries, predictions)
    ]
    return {
        "dataset_name": prepared.manifest.dataset_name,
        "database_records": len(prepared.database),
        "validation_database_records": len(validation_database),
        "validation_queries": len(prepared.queries),
        "passed_queries": sum(case["passed"] for case in cases),
        "all_passed": all(case["passed"] for case in cases),
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SAGE Unicode cleaning and real encrypted querying."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "examples" / "sage_pipeline.json"),
    )
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepared = build_prepared_dataset(args.config)
    result = validate_prepared_dataset(prepared)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
