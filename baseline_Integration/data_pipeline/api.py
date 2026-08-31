"""Public API for loading and preparing configured datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .adapters import create_adapter
from .builder import DatasetBuildConfig, DatasetBuilder
from .contracts import PreparedDataset


def build_prepared_dataset(
    config: Mapping[str, Any] | str | Path,
    *,
    base_dir: str | Path | None = None,
) -> PreparedDataset:
    """Build a prepared dataset from a mapping or JSON configuration file."""
    config_mapping, resolved_base_dir = load_pipeline_config(
        config,
        base_dir=base_dir,
    )
    adapter_config = config_mapping.get("adapter")
    if not isinstance(adapter_config, Mapping):
        raise ValueError("Pipeline configuration requires an adapter object")
    build_config = config_mapping.get("build", {})
    if not isinstance(build_config, Mapping):
        raise ValueError("Pipeline build configuration must be an object")

    raw = create_adapter(adapter_config, base_dir=resolved_base_dir).load()
    return DatasetBuilder(DatasetBuildConfig.from_mapping(build_config)).build(raw)


def load_pipeline_config(
    config: Mapping[str, Any] | str | Path,
    *,
    base_dir: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    if isinstance(config, Mapping):
        return deepcopy(dict(config)), Path(base_dir or ".").resolve()

    path = Path(config)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    path = path.resolve()
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Pipeline JSON root must be an object")
    return value, path.parent


def ncvr_10k_pipeline_config(
    path: str | Path,
    *,
    database_limit: int | None = None,
    query_limit: int | None = None,
    fuzzy_ratio: float = 0.0,
    fuzzy_seed: int = 42,
) -> dict[str, Any]:
    return {
        "adapter": {
            "type": "ncvr_10k",
            "name": "ncvr_10k",
            "path": str(Path(path).resolve()),
        },
        "build": {
            "normalizer": "english_v1",
            "database_deduplication": "record_id",
            "query_deduplication": "query_id",
            "database_limit": database_limit,
            "query_limit": query_limit,
            "fuzzy_ratio": fuzzy_ratio,
            "fuzzy_seed": fuzzy_seed,
            "perturbation": "single_substitution",
        },
    }
