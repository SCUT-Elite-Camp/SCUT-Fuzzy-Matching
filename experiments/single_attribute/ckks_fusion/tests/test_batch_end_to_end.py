"""End-to-end integration tests for batch HE protocol in protocol/orchestrator.py."""

import numpy as np
import pytest

from protocol.orchestrator import (
    run_batch_query_protocol,
    run_single_query_protocol,
)

_NAMES_B = [
    "JOHN SMITH",
    "MARY JONES",
    "ALICE BROWN",
    "BOB WILSON",
    "CHARLIE DAVIS",
    "DAVID MILLER",
    "EMILY TAYLOR",
    "FRANK MOORE",
    "GRACE JACKSON",
    "HENRY WHITE",
]


def test_batch_m1_matches_single_query_serial():
    """Test m=1 batch run results in identical cluster choice and match catch as single-query serial."""
    query = "JOHN SMITH"

    serial_run = run_single_query_protocol(_NAMES_B, query, random_state=42)
    batch_run = run_batch_query_protocol(_NAMES_B, [query], random_state=42)

    assert batch_run.batch_match_result.catches.shape == (1,)
    assert batch_run.batch_match_result.catches[0] == serial_run.match_result.catch
    assert (
        batch_run.batch_cluster_debug.selected_clusters[0]
        == serial_run.cluster_debug.selected_cluster
    )
    assert (
        batch_run.batch_match_debug.first_positive_columns[0]
        == serial_run.match_debug.first_positive_column
    )


def test_batch_m5_mixed_queries_against_serial():
    """Test m=5 containing exact, fuzzy, and non-match queries against serial runs."""
    query_names = [
        "JOHN SMITH",            # Exact match
        "JOHN SMYTH",            # Fuzzy match
        "MARY JONES",            # Exact match
        "ZZZZZZ NONEXISTENT",    # Non-match
        "ALICE BROWN",           # Exact match
    ]

    batch_run = run_batch_query_protocol(_NAMES_B, query_names, random_state=42)

    assert batch_run.batch_match_result.catches.shape == (5,)
    assert batch_run.batch_cluster_debug.selected_clusters.shape == (5,)

    # Run each query through single-query protocol for comparison
    for q_idx, q_name in enumerate(query_names):
        s_run = run_single_query_protocol(_NAMES_B, q_name, random_state=42)
        assert (
            batch_run.batch_match_result.catches[q_idx]
            == s_run.match_result.catch
        ), f"Mismatch catch for query '{q_name}'"
        assert (
            batch_run.batch_cluster_debug.selected_clusters[q_idx]
            == s_run.cluster_debug.selected_cluster
        ), f"Mismatch cluster choice for query '{q_name}'"


def test_batch_reproducibility():
    """Test repeated execution with identical inputs gives consistent cluster and catch results."""
    query_names = ["JOHN SMITH", "MARY JONES", "UNKNOWN NAME"]

    run1 = run_batch_query_protocol(_NAMES_B, query_names, random_state=123)
    run2 = run_batch_query_protocol(_NAMES_B, query_names, random_state=123)

    np.testing.assert_array_equal(
        run1.batch_match_result.catches, run2.batch_match_result.catches
    )
    np.testing.assert_array_equal(
        run1.batch_cluster_debug.selected_clusters,
        run2.batch_cluster_debug.selected_clusters,
    )


def test_batch_native_vs_serialized_communication():
    """Test native object ciphertext path and full bytes serialized path produce identical results."""
    query_names = ["JOHN SMITH", "CHARLIE DAVIS", "NONEXISTENT QUERY"]

    native_run = run_batch_query_protocol(
        _NAMES_B, query_names, random_state=42, serialize_communication=False
    )
    bytes_run = run_batch_query_protocol(
        _NAMES_B, query_names, random_state=42, serialize_communication=True
    )

    np.testing.assert_array_equal(
        native_run.batch_match_result.catches,
        bytes_run.batch_match_result.catches,
    )
    np.testing.assert_array_equal(
        native_run.batch_cluster_debug.selected_clusters,
        bytes_run.batch_cluster_debug.selected_clusters,
    )
    np.testing.assert_array_equal(
        native_run.batch_match_debug.first_positive_columns,
        bytes_run.batch_match_debug.first_positive_columns,
    )


def test_batch_early_stop_behavior():
    """Test early_stop=False processes max_size columns, and early_stop=True doesn't drop unhit queries."""
    # All matching queries
    all_hits = ["JOHN SMITH", "MARY JONES"]

    run_early = run_batch_query_protocol(
        _NAMES_B, all_hits, random_state=42, early_stop=True
    )
    run_full = run_batch_query_protocol(
        _NAMES_B, all_hits, random_state=42, early_stop=False
    )

    assert run_full.batch_match_debug.checked_columns == run_full.artifacts.max_size
    assert run_early.batch_match_debug.checked_columns <= run_full.artifacts.max_size
    np.testing.assert_array_equal(
        run_early.batch_match_result.catches, run_full.batch_match_result.catches
    )

    # Mixed hits including 1 non-match: early_stop=True MUST NOT stop early because 1 query hasn't caught
    mixed = ["JOHN SMITH", "NONEXISTENT QUERY"]
    run_mixed_early = run_batch_query_protocol(
        _NAMES_B, mixed, random_state=42, early_stop=True
    )
    assert run_mixed_early.batch_match_debug.checked_columns == run_mixed_early.artifacts.max_size
