"""Tests for centralized cluster-count selection."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.params import choose_k


def test_choose_k_sqrt_mode_uses_baseline_rule():
    assert choose_k(10, "sqrt") == 3
    assert choose_k(1, "sqrt") == 1


def test_choose_k_accepts_fixed_integer_and_clamps_to_dataset_size():
    assert choose_k(100, 50) == 50
    assert choose_k(10, 50) == 10


def test_choose_k_accepts_fixed_string_modes():
    assert choose_k(100, "50") == 50
    assert choose_k(100, "fixed:50") == 50


def test_choose_k_rejects_unknown_or_unmapped_modes():
    with pytest.raises(NotImplementedError):
        choose_k(100, "paper")
    with pytest.raises(ValueError):
        choose_k(100, "fixed:abc")
    with pytest.raises(ValueError):
        choose_k(100, "unknown")
