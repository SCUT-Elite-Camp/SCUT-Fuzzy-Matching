"""Tests for the curated two-file NCVR 10K dataset loader."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.dataset_loader import load_dataset


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_ncvr_10k_loader_reads_database_and_queries_directory():
    names_a, names_b, labels = load_dataset(
        "ncvr_10k",
        os.path.join(PROJECT_ROOT, "data", "ncvr_10k"),
    )

    assert len(names_b) == 10000
    assert len(names_a) == 200
    assert len(labels) == 200
    assert names_b[0] == "RUTH EVELYN AABEL"
    assert names_a[0] == "RUTH EVELYN AABEL"
    assert labels[:100] == [True] * 100
    assert labels[100:] == [False] * 100


def test_ncvr_10k_loader_accepts_parent_data_directory():
    names_a, names_b, labels = load_dataset(
        "ncvr_10k",
        os.path.join(PROJECT_ROOT, "data"),
    )

    assert names_a
    assert names_b
    assert labels
