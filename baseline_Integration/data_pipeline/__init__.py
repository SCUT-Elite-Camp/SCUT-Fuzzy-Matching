"""Unified dataset preparation API for fuzzy name matching."""

from .api import build_prepared_dataset, ncvr_10k_pipeline_config
from .adapters import register_adapter
from .contracts import (
    NameRecord,
    PipelineManifest,
    PreparedDataset,
    PreparedName,
    PreparedQuery,
    QueryRecord,
    RawDataset,
)
from .io import write_prepared_dataset
from .normalizers import register_normalizer
from .perturbations import register_perturbation

__all__ = [
    "NameRecord",
    "PipelineManifest",
    "PreparedDataset",
    "PreparedName",
    "PreparedQuery",
    "QueryRecord",
    "RawDataset",
    "build_prepared_dataset",
    "ncvr_10k_pipeline_config",
    "register_adapter",
    "register_normalizer",
    "register_perturbation",
    "write_prepared_dataset",
]
