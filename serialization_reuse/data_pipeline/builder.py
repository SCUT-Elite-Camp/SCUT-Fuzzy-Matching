"""Build validated, normalized datasets for database and query matching."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .contracts import (
    NameRecord,
    PipelineManifest,
    PreparedDataset,
    PreparedName,
    PreparedQuery,
    QueryRecord,
    RawDataset,
)
from .normalizers import NameNormalizer, create_normalizer
from .perturbations import perturb_positive_queries


@dataclass(frozen=True, slots=True)
class DatasetBuildConfig:
    normalizer: str = "english_v1"
    normalizer_options: Mapping[str, Any] = field(default_factory=dict)
    reject_empty: bool = True
    database_deduplication: str = "record_id"
    query_deduplication: str = "query_id"
    database_limit: int | None = None
    query_limit: int | None = None
    fuzzy_ratio: float = 0.0
    fuzzy_seed: int = 42
    perturbation: str = "single_substitution"
    perturbation_options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
    ) -> "DatasetBuildConfig":
        value = dict(value or {})
        allowed = {
            "normalizer",
            "normalizer_options",
            "reject_empty",
            "database_deduplication",
            "query_deduplication",
            "database_limit",
            "query_limit",
            "fuzzy_ratio",
            "fuzzy_seed",
            "perturbation",
            "perturbation_options",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown build options: {sorted(unknown)}")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalizer": self.normalizer,
            "normalizer_options": dict(self.normalizer_options),
            "reject_empty": self.reject_empty,
            "database_deduplication": self.database_deduplication,
            "query_deduplication": self.query_deduplication,
            "database_limit": self.database_limit,
            "query_limit": self.query_limit,
            "fuzzy_ratio": self.fuzzy_ratio,
            "fuzzy_seed": self.fuzzy_seed,
            "perturbation": self.perturbation,
            "perturbation_options": dict(self.perturbation_options),
        }


class DatasetBuilder:
    def __init__(self, config: DatasetBuildConfig):
        self.config = config
        self.normalizer = create_normalizer(
            config.normalizer,
            config.normalizer_options,
        )
        _validate_deduplication(config.database_deduplication, "database")
        _validate_deduplication(config.query_deduplication, "query")
        _validate_limit(config.database_limit, "database_limit")
        _validate_limit(config.query_limit, "query_limit")

    def build(self, raw: RawDataset) -> PreparedDataset:
        database_rows = _apply_limit(raw.database, self.config.database_limit)
        query_rows = _apply_limit(raw.queries, self.config.query_limit)

        database, rejected_database = self._normalize_database(database_rows)
        queries, rejected_queries = self._normalize_queries(query_rows)
        database, duplicate_database = _deduplicate_database(
            database,
            self.config.database_deduplication,
        )
        queries, duplicate_queries = _deduplicate_queries(
            queries,
            self.config.query_deduplication,
        )
        queries, perturbed_count = perturb_positive_queries(
            queries,
            ratio=self.config.fuzzy_ratio,
            seed=self.config.fuzzy_seed,
            perturbation_name=self.config.perturbation,
            perturbation_options=self.config.perturbation_options,
            forbidden={record.canonical_name for record in database},
        )

        database_ids = {record.record_id for record in database}
        orphan_positive_queries = sum(
            bool(query.expected_record_ids)
            and not bool(query.expected_record_ids & database_ids)
            for query in queries
            if query.label
        )
        manifest = PipelineManifest(
            dataset_name=raw.dataset_name,
            adapter_name=raw.adapter_name,
            normalizer_name=self.normalizer.name,
            perturbation_name=(
                self.config.perturbation if self.config.fuzzy_ratio > 0 else None
            ),
            input_database_rows=len(database_rows),
            input_query_rows=len(query_rows),
            output_database_rows=len(database),
            output_query_rows=len(queries),
            rejected_database_rows=rejected_database,
            rejected_query_rows=rejected_queries,
            duplicate_database_rows=duplicate_database,
            duplicate_query_rows=duplicate_queries,
            perturbed_query_rows=perturbed_count,
            orphan_positive_queries=orphan_positive_queries,
            unique_database_canonical_names=len(
                {record.canonical_name for record in database}
            ),
            matching_database_entries=len(
                {
                    value
                    for record in database
                    for value in (record.canonical_name, *record.variants.values())
                    if value
                }
            ),
            database_script_counts=dict(
                sorted(Counter(record.script or "Unknown" for record in database).items())
            ),
            query_script_counts=dict(
                sorted(Counter(record.script or "Unknown" for record in queries).items())
            ),
            config=self.config.to_dict(),
            source_stats=raw.source_stats,
        )
        return PreparedDataset(
            database=tuple(database),
            queries=tuple(queries),
            manifest=manifest,
        )

    def _normalize_database(
        self,
        records: Sequence[NameRecord],
    ) -> tuple[list[PreparedName], int]:
        prepared: list[PreparedName] = []
        rejected = 0
        for record in records:
            normalized = self.normalizer.normalize(record)
            if not normalized.canonical_name and self.config.reject_empty:
                rejected += 1
                continue
            prepared.append(
                PreparedName(
                    record_id=record.record_id,
                    raw_name=record.raw_name,
                    canonical_name=normalized.canonical_name,
                    dataset=record.dataset,
                    script=normalized.script,
                    variants=normalized.variants,
                    language_hint=record.language_hint,
                    country=record.country,
                    warnings=normalized.warnings,
                    metadata=record.metadata,
                )
            )
        return prepared, rejected

    def _normalize_queries(
        self,
        records: Sequence[QueryRecord],
    ) -> tuple[list[PreparedQuery], int]:
        prepared: list[PreparedQuery] = []
        rejected = 0
        for record in records:
            normalized = self.normalizer.normalize(record)
            if not normalized.canonical_name and self.config.reject_empty:
                rejected += 1
                continue
            prepared.append(
                PreparedQuery(
                    query_id=record.query_id,
                    raw_name=record.raw_name,
                    base_canonical_name=normalized.canonical_name,
                    canonical_name=normalized.canonical_name,
                    dataset=record.dataset,
                    label=record.label,
                    expected_record_ids=record.expected_record_ids,
                    script=normalized.script,
                    variants=normalized.variants,
                    language_hint=record.language_hint,
                    country=record.country,
                    warnings=normalized.warnings,
                    metadata=record.metadata,
                )
            )
        return prepared, rejected


def prepare_name_lists(
    names_a: Sequence[str],
    names_b: Sequence[str],
    labels: Sequence[bool],
    *,
    normalizer: str = "english_v1",
    fuzzy_ratio: float = 0.0,
    fuzzy_seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Compatibility helper for existing list-based callers."""
    if len(names_a) != len(labels):
        raise ValueError("Query names and labels must have the same length")
    raw = RawDataset(
        dataset_name="legacy_lists",
        adapter_name="legacy_lists",
        database=tuple(
            NameRecord(
                record_id=f"db-{index + 1}",
                raw_name=name,
                dataset="legacy_lists",
            )
            for index, name in enumerate(names_b)
        ),
        queries=tuple(
            QueryRecord(
                query_id=f"query-{index + 1}",
                raw_name=name,
                dataset="legacy_lists",
                label=bool(label),
            )
            for index, (name, label) in enumerate(zip(names_a, labels))
        ),
    )
    prepared = DatasetBuilder(
        DatasetBuildConfig(
            normalizer=normalizer,
            database_deduplication="none",
            query_deduplication="none",
            fuzzy_ratio=fuzzy_ratio,
            fuzzy_seed=fuzzy_seed,
        )
    ).build(raw)
    return prepared.query_names, prepared.database_names


def _apply_limit(records: Sequence[Any], limit: int | None) -> Sequence[Any]:
    if limit is None or limit == -1:
        return records
    return records[:limit]


def _validate_limit(value: int | None, name: str) -> None:
    if value is None or value == -1:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be -1, None, or a non-negative integer")


def _validate_deduplication(value: str, target: str) -> None:
    identifier = "record_id" if target == "database" else "query_id"
    allowed = {"none", identifier, "canonical_name"}
    if value not in allowed:
        raise ValueError(
            f"{target}_deduplication must be one of {sorted(allowed)}, got {value}"
        )


def _deduplicate_database(
    records: Sequence[PreparedName],
    policy: str,
) -> tuple[list[PreparedName], int]:
    if policy == "none":
        return list(records), 0
    key_name = "record_id" if policy == "record_id" else "canonical_name"
    return _deduplicate(records, key_name)


def _deduplicate_queries(
    records: Sequence[PreparedQuery],
    policy: str,
) -> tuple[list[PreparedQuery], int]:
    if policy == "none":
        return list(records), 0
    key_name = "query_id" if policy == "query_id" else "canonical_name"
    return _deduplicate(records, key_name)


def _deduplicate(records: Sequence[Any], key_name: str) -> tuple[list[Any], int]:
    seen: set[Any] = set()
    result: list[Any] = []
    duplicate_count = 0
    for record in records:
        key = getattr(record, key_name)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        result.append(record)
    return result, duplicate_count
