"""Dataset adapters that map source-specific schemas to canonical records."""

from __future__ import annotations

import csv
import hashlib
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from .contracts import NameRecord, QueryRecord, RawDataset


class DatasetAdapter(Protocol):
    name: str

    def load(self) -> RawDataset: ...


AdapterFactory = Callable[[Mapping[str, Any], Path], DatasetAdapter]
_ADAPTERS: dict[str, AdapterFactory] = {}


def register_adapter(
    name: str,
    factory: AdapterFactory,
    *,
    replace: bool = False,
) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("Adapter name cannot be empty")
    if key in _ADAPTERS and not replace:
        raise ValueError(f"Adapter already registered: {key}")
    _ADAPTERS[key] = factory


def create_adapter(
    config: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> DatasetAdapter:
    adapter_type = str(config.get("type", "")).strip().lower()
    if not adapter_type:
        raise ValueError("adapter.type is required")
    try:
        factory = _ADAPTERS[adapter_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown adapter: {adapter_type}. Available: {sorted(_ADAPTERS)}"
        ) from exc
    return factory(config, Path(base_dir or ".").resolve())


class Ncvr10kAdapter:
    name = "ncvr_10k"

    def __init__(self, config: Mapping[str, Any], base_dir: Path):
        self.dataset_name = str(config.get("name", "ncvr_10k"))
        self.root = _resolve_path(config.get("path", "data"), base_dir)
        self.encoding = str(config.get("encoding", "utf-8"))

    def load(self) -> RawDataset:
        database_path, queries_path = _resolve_ncvr_paths(self.root)
        database_rows = _read_csv(database_path, self.encoding)
        query_rows = _read_csv(queries_path, self.encoding)
        _require_columns(database_rows, database_path, {"ncid", "full_name"})
        _require_columns(
            query_rows,
            queries_path,
            {"query_ncid", "query_name", "label"},
        )

        database = tuple(
            NameRecord(
                record_id=row["ncid"].strip(),
                raw_name=row["full_name"].strip(),
                dataset=self.dataset_name,
                metadata={"source_row": index + 2},
            )
            for index, row in enumerate(database_rows)
            if row["full_name"].strip()
        )
        queries: list[QueryRecord] = []
        for index, row in enumerate(query_rows):
            raw_name = row["query_name"].strip()
            if not raw_name:
                continue
            label = _parse_bool(row["label"])
            match_id = row["query_ncid"].strip()
            queries.append(
                QueryRecord(
                    query_id=f"query-{index + 1}",
                    raw_name=raw_name,
                    dataset=self.dataset_name,
                    label=label,
                    expected_record_ids=(
                        frozenset({match_id}) if label and match_id else frozenset()
                    ),
                    metadata={
                        "source_row": index + 2,
                        "source_query_ncid": match_id,
                    },
                )
            )
        return RawDataset(
            dataset_name=self.dataset_name,
            adapter_name=self.name,
            database=database,
            queries=tuple(queries),
        )


class CsvPairAdapter:
    name = "csv_pair"

    def __init__(self, config: Mapping[str, Any], base_dir: Path):
        self.dataset_name = str(config.get("name", "csv_pair"))
        self.database_config = _require_mapping(config, "database")
        self.query_config = _require_mapping(config, "queries")
        self.base_dir = base_dir

    def load(self) -> RawDataset:
        database = self._load_database()
        queries = self._load_queries()
        return RawDataset(
            dataset_name=self.dataset_name,
            adapter_name=self.name,
            database=tuple(database),
            queries=tuple(queries),
        )

    def _load_database(self) -> list[NameRecord]:
        config = self.database_config
        path = _resolve_path(_required(config, "path"), self.base_dir)
        rows = _read_csv(path, str(config.get("encoding", "utf-8")))
        name_column = str(_required(config, "name_column"))
        id_column = _optional_column(config, "id_column")
        country_column = _optional_column(config, "country_column")
        language_column = _optional_column(config, "language_column")
        _require_columns(rows, path, _configured_columns(config, "database"))

        result: list[NameRecord] = []
        for index, row in enumerate(rows):
            raw_name = row[name_column].strip()
            if not raw_name:
                continue
            record_id = row[id_column].strip() if id_column else f"db-{index + 1}"
            if not record_id:
                raise ValueError(f"Empty database ID at {path}:{index + 2}")
            result.append(
                NameRecord(
                    record_id=record_id,
                    raw_name=raw_name,
                    dataset=self.dataset_name,
                    country=_column_value(row, country_column),
                    language_hint=_column_value(row, language_column),
                    metadata=_metadata(row, config, index),
                )
            )
        return result

    def _load_queries(self) -> list[QueryRecord]:
        config = self.query_config
        path = _resolve_path(_required(config, "path"), self.base_dir)
        rows = _read_csv(path, str(config.get("encoding", "utf-8")))
        name_column = str(_required(config, "name_column"))
        id_column = _optional_column(config, "id_column")
        label_column = _optional_column(config, "label_column")
        match_id_column = _optional_column(config, "match_id_column")
        expected_ids_column = _optional_column(config, "expected_ids_column")
        country_column = _optional_column(config, "country_column")
        language_column = _optional_column(config, "language_column")
        _require_columns(rows, path, _configured_columns(config, "queries"))
        separator = str(config.get("expected_ids_separator", "|"))

        result: list[QueryRecord] = []
        for index, row in enumerate(rows):
            raw_name = row[name_column].strip()
            if not raw_name:
                continue
            query_id = row[id_column].strip() if id_column else f"query-{index + 1}"
            if not query_id:
                raise ValueError(f"Empty query ID at {path}:{index + 2}")
            expected_ids = _expected_ids(
                row,
                match_id_column=match_id_column,
                expected_ids_column=expected_ids_column,
                separator=separator,
            )
            label = _parse_bool(row[label_column]) if label_column else bool(expected_ids)
            if not label:
                expected_ids = frozenset()
            result.append(
                QueryRecord(
                    query_id=query_id,
                    raw_name=raw_name,
                    dataset=self.dataset_name,
                    label=label,
                    expected_record_ids=expected_ids,
                    country=_column_value(row, country_column),
                    language_hint=_column_value(row, language_column),
                    metadata=_metadata(row, config, index),
                )
            )
        return result


class SageNamesAdapter:
    """Load the extracted SAGE candidate-name corpus and validation queries."""

    name = "sage_names"

    def __init__(self, config: Mapping[str, Any], base_dir: Path):
        self.dataset_name = str(config.get("name", "sage"))
        self.path = _resolve_path(config.get("path", "data/sage/sage_names_all.csv"), base_dir)
        raw_queries_path = config.get("queries_path")
        self.queries_path = (
            _resolve_path(raw_queries_path, base_dir) if raw_queries_path else None
        )
        self.encoding = str(config.get("encoding", "utf-8"))

    def load(self) -> RawDataset:
        source_rows = _read_csv(self.path, self.encoding)
        _require_columns(source_rows, self.path, {"candidate_name", "country"})

        database: list[NameRecord] = []
        seen_pairs: set[tuple[str, str]] = set()
        countries: Counter[str] = Counter()
        blank_rows = 0
        for index, row in enumerate(source_rows):
            raw_name = _clean_source_display_text(row["candidate_name"])
            country = _clean_source_display_text(row["country"]) or "Unknown"
            if not raw_name:
                blank_rows += 1
                continue
            pair = (
                _normalize_source_text(raw_name).casefold(),
                _normalize_source_text(country).casefold(),
            )
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            countries[country] += 1
            database.append(
                NameRecord(
                    record_id=sage_record_id(raw_name, country),
                    raw_name=raw_name,
                    dataset=self.dataset_name,
                    language_hint=_SAGE_COUNTRY_LANGUAGES.get(country),
                    country=country,
                    metadata={"source_row": index + 2},
                )
            )

        queries = self._load_queries()
        return RawDataset(
            dataset_name=self.dataset_name,
            adapter_name=self.name,
            database=tuple(database),
            queries=tuple(queries),
            source_stats={
                "source_rows": len(source_rows),
                "blank_name_rows": blank_rows,
                "unique_name_country_pairs": len(database),
                "source_duplicate_rows": len(source_rows) - blank_rows - len(database),
                "countries": len(countries),
                "country_counts": dict(sorted(countries.items())),
            },
        )

    def _load_queries(self) -> list[QueryRecord]:
        if self.queries_path is None:
            return []
        rows = _read_csv(self.queries_path, self.encoding)
        _require_columns(
            rows,
            self.queries_path,
            {"query_id", "query_name", "label", "expected_record_id"},
        )
        result: list[QueryRecord] = []
        for index, row in enumerate(rows):
            raw_name = _clean_source_display_text(row["query_name"])
            if not raw_name:
                continue
            label = _parse_bool(row["label"])
            expected_id = row["expected_record_id"].strip()
            country = _optional_row_value(row, "country")
            language = _optional_row_value(row, "language")
            result.append(
                QueryRecord(
                    query_id=row["query_id"].strip() or f"query-{index + 1}",
                    raw_name=raw_name,
                    dataset=self.dataset_name,
                    label=label,
                    expected_record_ids=(
                        frozenset({expected_id})
                        if label and expected_id
                        else frozenset()
                    ),
                    language_hint=language,
                    country=country,
                    metadata={"source_row": index + 2, "validation_query": True},
                )
            )
        return result


def _resolve_ncvr_paths(root: Path) -> tuple[Path, Path]:
    for candidate in (root, root / "ncvr_10k"):
        database = candidate / "ncvr_10k_database.csv"
        queries = candidate / "ncvr_10k_queries.csv"
        if database.exists() and queries.exists():
            return database, queries
    raise FileNotFoundError(f"Could not find NCVR 10K files below {root}")


def _read_csv(path: Path, encoding: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding=encoding) as handle:
        return list(csv.DictReader(handle))


def _require_columns(
    rows: list[dict[str, str]],
    path: Path,
    expected: set[str],
) -> None:
    if not rows:
        raise ValueError(f"CSV file has no data rows: {path}")
    missing = expected - set(rows[0])
    if missing:
        raise ValueError(f"CSV file {path} is missing columns: {sorted(missing)}")


def _parse_bool(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean label: {value!r}")


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _required(config: Mapping[str, Any], key: str) -> Any:
    value = config.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing required configuration: {key}")
    return value


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _optional_column(config: Mapping[str, Any], key: str) -> str | None:
    value = config.get(key)
    return str(value) if value is not None and str(value).strip() else None


def _configured_columns(config: Mapping[str, Any], section: str) -> set[str]:
    keys = ["name_column", "id_column", "country_column", "language_column"]
    if section == "queries":
        keys.extend(["label_column", "match_id_column", "expected_ids_column"])
    return {str(config[key]) for key in keys if config.get(key)}


def _column_value(row: Mapping[str, str], column: str | None) -> str | None:
    if not column:
        return None
    value = row[column].strip()
    return value or None


def _metadata(
    row: Mapping[str, str],
    config: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"source_row": index + 2}
    keep_columns = config.get("metadata_columns", [])
    if not isinstance(keep_columns, list):
        raise ValueError("metadata_columns must be a list")
    for column in keep_columns:
        column = str(column)
        if column not in row:
            raise ValueError(f"Unknown metadata column: {column}")
        metadata[column] = row[column]
    return metadata


def _expected_ids(
    row: Mapping[str, str],
    *,
    match_id_column: str | None,
    expected_ids_column: str | None,
    separator: str,
) -> frozenset[str]:
    if expected_ids_column:
        return frozenset(
            value.strip()
            for value in row[expected_ids_column].split(separator)
            if value.strip()
        )
    if match_id_column and row[match_id_column].strip():
        return frozenset({row[match_id_column].strip()})
    return frozenset()


def sage_record_id(name: str, country: str) -> str:
    """Return a stable ID for one SAGE name/country pair."""
    normalized_name = _normalize_source_text(name).casefold()
    normalized_country = _normalize_source_text(country).casefold()
    key = f"{normalized_name}\0{normalized_country}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"sage-{digest}"


def _normalize_source_text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def _clean_source_display_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _optional_row_value(row: Mapping[str, str], column: str) -> str | None:
    value = row.get(column, "").strip()
    return value or None


_SAGE_COUNTRY_LANGUAGES = {
    "Afghanistan": "fa/ps",
    "Barbados": "en",
    "Belize": "en/es",
    "Botswana": "en/tn",
    "Dominica": "en",
    "Gambia": "en",
    "Ghana": "en",
    "Grenada": "en",
    "Hungary": "hu",
    "India": "multi",
    "Iraq": "ar",
    "Italy": "it",
    "Ivory Coast": "fr",
    "Lesotho": "st/en",
    "Liberia": "en",
    "Malaysia": "ms",
    "Myanmar": "my",
    "Nicaragua": "es",
    "Pakistan": "ur/en",
    "Saint Lucia": "en",
    "Samoa": "sm/en",
    "Seychelles": "crs/en/fr",
    "Taiwan": "zh",
}


register_adapter(
    "ncvr_10k",
    lambda config, base_dir: Ncvr10kAdapter(config, base_dir),
)
register_adapter(
    "csv_pair",
    lambda config, base_dir: CsvPairAdapter(config, base_dir),
)
register_adapter(
    "sage_names",
    lambda config, base_dir: SageNamesAdapter(config, base_dir),
)
