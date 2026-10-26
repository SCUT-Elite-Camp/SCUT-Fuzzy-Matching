"""Tests for English-only NCVR evaluation preprocessing."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.dataset_loader import load_dataset
from evaluation.ncvr_preprocessing import prepare_ncvr_names


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")


def test_ncvr_names_are_normalized_without_mutating_raw_loader_output():
    raw_a, raw_b, labels = load_dataset("ncvr_10k", DATA_PATH)
    names_a, names_b = prepare_ncvr_names(raw_a, raw_b, labels)

    assert raw_a[0] == "RUTH EVELYN AABEL"
    assert raw_b[0] == "RUTH EVELYN AABEL"
    assert names_a[0] == "ruth evelyn aabel"
    assert names_b[0] == "ruth evelyn aabel"


def test_fuzzification_changes_only_requested_positive_queries():
    raw_a, raw_b, labels = load_dataset("ncvr_10k", DATA_PATH)
    exact_a, exact_b = prepare_ncvr_names(raw_a, raw_b, labels)
    fuzzy_a, fuzzy_b = prepare_ncvr_names(
        raw_a,
        raw_b,
        labels,
        fuzzy_ratio=0.3,
        fuzzy_seed=7,
    )

    changed = [
        index
        for index, (exact, fuzzy) in enumerate(zip(exact_a, fuzzy_a))
        if exact != fuzzy
    ]
    assert len(changed) == 30
    assert all(labels[index] for index in changed)
    assert fuzzy_a[100:] == exact_a[100:]
    assert fuzzy_b == exact_b
    assert all(fuzzy_a[index] not in set(fuzzy_b) for index in changed)


def test_fuzzification_is_reproducible():
    raw_a, raw_b, labels = load_dataset("ncvr_10k", DATA_PATH)
    first, _ = prepare_ncvr_names(
        raw_a, raw_b, labels, fuzzy_ratio=0.25, fuzzy_seed=123
    )
    second, _ = prepare_ncvr_names(
        raw_a, raw_b, labels, fuzzy_ratio=0.25, fuzzy_seed=123
    )

    assert first == second


@pytest.mark.parametrize("ratio", [-0.1, 1.1, True, "0.3"])
def test_rejects_invalid_fuzzy_ratio(ratio):
    with pytest.raises(ValueError, match="fuzzy_ratio"):
        prepare_ncvr_names(["JOHN SMITH"], ["JOHN SMITH"], [True], fuzzy_ratio=ratio)


def test_rejects_misaligned_labels():
    with pytest.raises(ValueError, match="same length"):
        prepare_ncvr_names(["JOHN SMITH"], ["JOHN SMITH"], [])
