"""Evaluation-only query perturbations, separate from production normalization."""

from __future__ import annotations

import random
import string
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Protocol

from .contracts import PreparedQuery


class QueryPerturbation(Protocol):
    name: str

    def can_apply(self, value: str) -> bool: ...

    def apply(
        self,
        value: str,
        rng: random.Random,
        *,
        forbidden: set[str],
    ) -> str: ...


PerturbationFactory = Callable[[Mapping[str, Any]], QueryPerturbation]
_PERTURBATIONS: dict[str, PerturbationFactory] = {}


def register_perturbation(
    name: str,
    factory: PerturbationFactory,
    *,
    replace_existing: bool = False,
) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("Perturbation name cannot be empty")
    if key in _PERTURBATIONS and not replace_existing:
        raise ValueError(f"Perturbation already registered: {key}")
    _PERTURBATIONS[key] = factory


def create_perturbation(
    name: str,
    options: Mapping[str, Any] | None = None,
) -> QueryPerturbation:
    key = name.strip().lower()
    try:
        factory = _PERTURBATIONS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown perturbation: {name}. Available: {sorted(_PERTURBATIONS)}"
        ) from exc
    return factory(dict(options or {}))


class SingleSubstitutionPerturbation:
    name = "single_substitution"

    def __init__(self, options: Mapping[str, Any] | None = None):
        options = dict(options or {})
        if options:
            raise ValueError(
                f"single_substitution has no options, got {sorted(options)}"
            )

    def apply(
        self,
        value: str,
        rng: random.Random,
        *,
        forbidden: set[str],
    ) -> str:
        positions = [
            index for index, char in enumerate(value) if char in string.ascii_lowercase
        ]
        rng.shuffle(positions)
        for position in positions:
            replacements = [
                char for char in string.ascii_lowercase if char != value[position]
            ]
            rng.shuffle(replacements)
            for replacement in replacements:
                candidate = value[:position] + replacement + value[position + 1 :]
                if candidate not in forbidden:
                    return candidate
        return value

    def can_apply(self, value: str) -> bool:
        return any(char in string.ascii_lowercase for char in value)


def perturb_positive_queries(
    queries: Sequence[PreparedQuery],
    *,
    ratio: float,
    seed: int,
    perturbation_name: str,
    perturbation_options: Mapping[str, Any] | None,
    forbidden: set[str],
) -> tuple[list[PreparedQuery], int]:
    _validate_ratio(ratio)
    result = list(queries)
    if ratio == 0:
        return result, 0

    perturbation = create_perturbation(
        perturbation_name,
        perturbation_options,
    )
    eligible = [
        index
        for index, query in enumerate(result)
        if query.label and perturbation.can_apply(query.canonical_name)
    ]
    target_count = int(len(eligible) * ratio + 0.5)
    rng = random.Random(seed)
    selected = rng.sample(eligible, target_count)
    changed = 0
    for index in selected:
        query = result[index]
        value = perturbation.apply(query.canonical_name, rng, forbidden=forbidden)
        if value == query.canonical_name:
            continue
        variants = dict(query.variants)
        variants["native"] = value
        result[index] = replace(
            query,
            canonical_name=value,
            variants=variants,
            perturbation=perturbation.name,
        )
        changed += 1
    return result, changed


def _validate_ratio(ratio: float) -> None:
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise ValueError(
            "fuzzy_ratio must be a number between 0 and 1, "
            f"got {ratio!r}"
        )
    if not 0 <= ratio <= 1:
        raise ValueError(f"fuzzy_ratio must be between 0 and 1, got {ratio}")


register_perturbation(
    "single_substitution",
    lambda options: SingleSubstitutionPerturbation(options),
)
