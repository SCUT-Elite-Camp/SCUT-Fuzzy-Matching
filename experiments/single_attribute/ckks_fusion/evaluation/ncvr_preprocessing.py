"""Backward-compatible NCVR preprocessing wrapper.

New code should use :mod:`data_pipeline` directly.  This module keeps the
original function available for scripts and tests that still import it.
"""

from __future__ import annotations

from collections.abc import Sequence

from data_pipeline.builder import prepare_name_lists


def prepare_ncvr_names(
    names_a: Sequence[str],
    names_b: Sequence[str],
    labels: Sequence[bool],
    *,
    fuzzy_ratio: float = 0.0,
    fuzzy_seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Normalize A/B names and fuzz an exact fraction of positive queries."""
    return prepare_name_lists(
        names_a,
        names_b,
        labels,
        normalizer="english_v1",
        fuzzy_ratio=fuzzy_ratio,
        fuzzy_seed=fuzzy_seed,
    )
