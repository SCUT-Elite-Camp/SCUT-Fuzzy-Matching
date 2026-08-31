"""Canonical records shared by dataset adapters and matching pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class NameRecord:
    record_id: str
    raw_name: str
    dataset: str
    language_hint: str | None = None
    country: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QueryRecord:
    query_id: str
    raw_name: str
    dataset: str
    label: bool
    expected_record_ids: frozenset[str] = field(default_factory=frozenset)
    language_hint: str | None = None
    country: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RawDataset:
    dataset_name: str
    adapter_name: str
    database: tuple[NameRecord, ...]
    queries: tuple[QueryRecord, ...]
    source_stats: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NormalizedValue:
    canonical_name: str
    script: str | None
    variants: Mapping[str, str]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedName:
    record_id: str
    raw_name: str
    canonical_name: str
    dataset: str
    script: str | None
    variants: Mapping[str, str]
    language_hint: str | None = None
    country: str | None = None
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreparedQuery:
    query_id: str
    raw_name: str
    base_canonical_name: str
    canonical_name: str
    dataset: str
    label: bool
    expected_record_ids: frozenset[str]
    script: str | None
    variants: Mapping[str, str]
    perturbation: str | None = None
    language_hint: str | None = None
    country: str | None = None
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PipelineManifest:
    dataset_name: str
    adapter_name: str
    normalizer_name: str
    perturbation_name: str | None
    input_database_rows: int
    input_query_rows: int
    output_database_rows: int
    output_query_rows: int
    rejected_database_rows: int
    rejected_query_rows: int
    duplicate_database_rows: int
    duplicate_query_rows: int
    perturbed_query_rows: int
    orphan_positive_queries: int
    unique_database_canonical_names: int
    matching_database_entries: int
    database_script_counts: Mapping[str, int]
    query_script_counts: Mapping[str, int]
    config: Mapping[str, Any]
    source_stats: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "adapter_name": self.adapter_name,
            "normalizer_name": self.normalizer_name,
            "perturbation_name": self.perturbation_name,
            "input_database_rows": self.input_database_rows,
            "input_query_rows": self.input_query_rows,
            "output_database_rows": self.output_database_rows,
            "output_query_rows": self.output_query_rows,
            "rejected_database_rows": self.rejected_database_rows,
            "rejected_query_rows": self.rejected_query_rows,
            "duplicate_database_rows": self.duplicate_database_rows,
            "duplicate_query_rows": self.duplicate_query_rows,
            "perturbed_query_rows": self.perturbed_query_rows,
            "orphan_positive_queries": self.orphan_positive_queries,
            "unique_database_canonical_names": self.unique_database_canonical_names,
            "matching_database_entries": self.matching_database_entries,
            "database_script_counts": dict(self.database_script_counts),
            "query_script_counts": dict(self.query_script_counts),
            "config": dict(self.config),
            "source_stats": dict(self.source_stats),
        }


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    database: tuple[PreparedName, ...]
    queries: tuple[PreparedQuery, ...]
    manifest: PipelineManifest

    @property
    def database_names(self) -> list[str]:
        return [record.canonical_name for record in self.database]

    @property
    def matching_database_names(self) -> list[str]:
        """Return canonical names plus unique searchable variants."""
        names: list[str] = []
        seen: set[str] = set()
        for record in self.database:
            for value in (record.canonical_name, *record.variants.values()):
                if value and value not in seen:
                    seen.add(value)
                    names.append(value)
        return names

    @property
    def query_names(self) -> list[str]:
        return [record.canonical_name for record in self.queries]

    @property
    def labels(self) -> list[bool]:
        return [record.label for record in self.queries]

    def as_legacy_tuple(self) -> tuple[list[str], list[str], list[bool]]:
        """Return the existing Party A / Party B / labels interface."""
        return self.query_names, self.matching_database_names, self.labels
