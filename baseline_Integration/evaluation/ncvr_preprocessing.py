"""English-only normalization and controlled typo generation for NCVR evaluation."""

from __future__ import annotations

import random
import string
from collections.abc import Sequence

from preprocessing.text_cleaner import clean_name


def prepare_ncvr_names(
    names_a: Sequence[str],
    names_b: Sequence[str],
    labels: Sequence[bool],
    *,
    fuzzy_ratio: float = 0.0,
    fuzzy_seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Normalize A/B names and fuzz an exact fraction of positive queries."""
    if len(names_a) != len(labels):
        raise ValueError("NCVR query names and labels must have the same length")
    _validate_fuzzy_ratio(fuzzy_ratio)

    normalized_a = [clean_name(name) for name in names_a]
    normalized_b = [clean_name(name) for name in names_b]
    if fuzzy_ratio == 0:
        return normalized_a, normalized_b

    eligible = [
        index
        for index, (name, label) in enumerate(zip(normalized_a, labels))
        if label and any(char in string.ascii_lowercase for char in name)
    ]
    fuzzy_count = int(len(eligible) * fuzzy_ratio + 0.5)
    rng = random.Random(fuzzy_seed)
    selected = rng.sample(eligible, fuzzy_count)
    database_names = set(normalized_b)

    for index in selected:
        normalized_a[index] = _replace_one_letter(
            normalized_a[index], rng, forbidden=database_names
        )
    return normalized_a, normalized_b


def _validate_fuzzy_ratio(ratio: float) -> None:
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise ValueError(
            "fuzzy_ratio must be a number between 0 and 1, "
            f"got {ratio!r}"
        )
    if not 0 <= ratio <= 1:
        raise ValueError(f"fuzzy_ratio must be between 0 and 1, got {ratio}")


def _replace_one_letter(name: str, rng: random.Random, *, forbidden: set[str]) -> str:
    """Create one letter substitution that is not an exact database name."""
    positions = [
        index for index, char in enumerate(name) if char in string.ascii_lowercase
    ]
    rng.shuffle(positions)
    for position in positions:
        replacements = [
            char for char in string.ascii_lowercase if char != name[position]
        ]
        rng.shuffle(replacements)
        for replacement in replacements:
            candidate = name[:position] + replacement + name[position + 1 :]
            if candidate not in forbidden:
                return candidate
    return name
