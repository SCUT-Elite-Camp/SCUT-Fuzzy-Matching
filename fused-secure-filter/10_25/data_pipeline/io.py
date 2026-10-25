"""Persist prepared datasets with an auditable manifest."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .contracts import PreparedDataset


def write_prepared_dataset(
    dataset: PreparedDataset,
    output_dir: str | Path,
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    database_path = root / "database.csv"
    queries_path = root / "queries.csv"
    manifest_path = root / "manifest.json"

    with database_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "record_id",
            "raw_name",
            "canonical_name",
            "dataset",
            "script",
            "language_hint",
            "country",
            "variants_json",
            "warnings_json",
            "metadata_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in dataset.database:
            writer.writerow(
                {
                    "record_id": record.record_id,
                    "raw_name": record.raw_name,
                    "canonical_name": record.canonical_name,
                    "dataset": record.dataset,
                    "script": record.script or "",
                    "language_hint": record.language_hint or "",
                    "country": record.country or "",
                    "variants_json": _json(record.variants),
                    "warnings_json": _json(record.warnings),
                    "metadata_json": _json(record.metadata),
                }
            )

    with queries_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "query_id",
            "raw_name",
            "base_canonical_name",
            "canonical_name",
            "dataset",
            "label",
            "expected_record_ids_json",
            "perturbation",
            "script",
            "language_hint",
            "country",
            "variants_json",
            "warnings_json",
            "metadata_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for query in dataset.queries:
            writer.writerow(
                {
                    "query_id": query.query_id,
                    "raw_name": query.raw_name,
                    "base_canonical_name": query.base_canonical_name,
                    "canonical_name": query.canonical_name,
                    "dataset": query.dataset,
                    "label": query.label,
                    "expected_record_ids_json": _json(
                        sorted(query.expected_record_ids)
                    ),
                    "perturbation": query.perturbation or "",
                    "script": query.script or "",
                    "language_hint": query.language_hint or "",
                    "country": query.country or "",
                    "variants_json": _json(query.variants),
                    "warnings_json": _json(query.warnings),
                    "metadata_json": _json(query.metadata),
                }
            )

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            dataset.manifest.to_dict(),
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    return {
        "database": database_path,
        "queries": queries_path,
        "manifest": manifest_path,
    }


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
