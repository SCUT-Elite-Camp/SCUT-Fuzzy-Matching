"""Tests for the configurable, dataset-independent preparation pipeline."""

from __future__ import annotations

import csv
import json

from data_pipeline import build_prepared_dataset, write_prepared_dataset
from evaluation.benchmark import benchmark
from scripts.evaluate_dataset import load_evaluation_config


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _csv_pair_config(tmp_path, *, fuzzy_ratio=0.0, normalizer="english_v1"):
    database_path = tmp_path / "people.csv"
    queries_path = tmp_path / "lookups.csv"
    _write_csv(
        database_path,
        ["person_id", "full_name", "country", "source"],
        [
            {"person_id": "p1", "full_name": "  JOHN   SMITH ", "country": "US", "source": "a"},
            {"person_id": "p2", "full_name": "MARY-JONES", "country": "US", "source": "b"},
            {"person_id": "p2", "full_name": "Duplicate ID", "country": "US", "source": "c"},
        ],
    )
    _write_csv(
        queries_path,
        ["lookup_id", "input_name", "is_match", "person_id"],
        [
            {"lookup_id": "q1", "input_name": "John Smith", "is_match": "true", "person_id": "p1"},
            {"lookup_id": "q2", "input_name": "Mary Jones", "is_match": "1", "person_id": "p2"},
            {"lookup_id": "q3", "input_name": "Nobody", "is_match": "false", "person_id": ""},
        ],
    )
    return {
        "adapter": {
            "type": "csv_pair",
            "name": "custom_people",
            "database": {
                "path": database_path.name,
                "id_column": "person_id",
                "name_column": "full_name",
                "country_column": "country",
                "metadata_columns": ["source"],
            },
            "queries": {
                "path": queries_path.name,
                "id_column": "lookup_id",
                "name_column": "input_name",
                "label_column": "is_match",
                "match_id_column": "person_id",
            },
        },
        "build": {
            "normalizer": normalizer,
            "database_deduplication": "record_id",
            "query_deduplication": "query_id",
            "fuzzy_ratio": fuzzy_ratio,
            "fuzzy_seed": 7,
        },
    }


def test_csv_pair_maps_independent_schemas_to_canonical_records(tmp_path):
    prepared = build_prepared_dataset(
        _csv_pair_config(tmp_path),
        base_dir=tmp_path,
    )

    assert prepared.database_names == ["john smith", "maryjones"]
    assert prepared.query_names == ["john smith", "mary jones", "nobody"]
    assert prepared.labels == [True, True, False]
    assert prepared.queries[0].expected_record_ids == frozenset({"p1"})
    assert prepared.database[0].metadata["source"] == "a"
    assert prepared.manifest.duplicate_database_rows == 1
    assert prepared.manifest.adapter_name == "csv_pair"


def test_evaluation_perturbation_changes_only_positive_queries(tmp_path):
    prepared = build_prepared_dataset(
        _csv_pair_config(tmp_path, fuzzy_ratio=0.5),
        base_dir=tmp_path,
    )

    changed = [query for query in prepared.queries if query.perturbation]
    assert len(changed) == 1
    assert changed[0].label is True
    assert changed[0].canonical_name != changed[0].base_canonical_name
    assert prepared.queries[2].canonical_name == "nobody"
    assert prepared.manifest.perturbed_query_rows == 1


def test_unicode_normalizer_preserves_native_and_transliterated_names(tmp_path):
    database_path = tmp_path / "db.csv"
    queries_path = tmp_path / "queries.csv"
    _write_csv(database_path, ["name"], [{"name": " 张  伟 "}])
    _write_csv(queries_path, ["name", "label"], [{"name": "张 伟", "label": "true"}])
    config = {
        "adapter": {
            "type": "csv_pair",
            "database": {"path": "db.csv", "name_column": "name"},
            "queries": {"path": "queries.csv", "name_column": "name", "label_column": "label"},
        },
        "build": {"normalizer": "unicode_v1"},
    }

    prepared = build_prepared_dataset(config, base_dir=tmp_path)

    assert prepared.database_names == ["张 伟"]
    assert prepared.database[0].script == "Han"
    assert prepared.database[0].variants == {
        "native": "张 伟",
        "latin_transliterated": "zhang wei",
        "latin_compact": "zhangwei",
    }


def test_prepared_dataset_can_be_exported_with_manifest(tmp_path):
    prepared = build_prepared_dataset(_csv_pair_config(tmp_path), base_dir=tmp_path)
    paths = write_prepared_dataset(prepared, tmp_path / "prepared")

    assert all(path.exists() for path in paths.values())
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["output_database_rows"] == 2
    assert manifest["output_query_rows"] == 3


def test_benchmark_accepts_pipeline_config_without_dataset_specific_code(tmp_path):
    result = benchmark(
        {
            "dataset_pipeline": _csv_pair_config(tmp_path),
            "dataset_base_dir": str(tmp_path),
            "el_cluster": 200,
            "el_match": 50,
            "k": 2,
            "tau": 0.9,
            "use_mock": True,
        }
    )

    assert len(result["predictions"]) == 3
    assert result["dataset_manifest"]["dataset_name"] == "custom_people"
    assert result["dataset_manifest"]["adapter_name"] == "csv_pair"


def test_evaluation_config_resolves_pipeline_relative_to_config_file(tmp_path):
    pipeline = _csv_pair_config(tmp_path)
    (tmp_path / "pipeline.json").write_text(json.dumps(pipeline), encoding="utf-8")
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(
        json.dumps({"dataset_pipeline": "pipeline.json", "matching": {"use_mock": True}}),
        encoding="utf-8",
    )

    config, base_dir, output_dir, prefix = load_evaluation_config(evaluation_path)
    prepared = build_prepared_dataset(
        config["dataset_pipeline"],
        base_dir=config["dataset_base_dir"],
    )

    assert prepared.manifest.output_database_rows == 2
    assert base_dir == tmp_path
    assert output_dir == "artifacts/evaluation/configured"
    assert prefix == "fuzzy_matching"
