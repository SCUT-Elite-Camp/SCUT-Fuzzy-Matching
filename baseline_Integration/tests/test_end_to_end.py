"""Real in-process end-to-end tests for the documented baseline flow."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.orchestrator import run_single_query_protocol


def test_protocol_returns_catch_for_known_similar_name():
    result = run_single_query_protocol(
        ["john smith", "mary jones"],
        "john smith",
        k_mode="sqrt",
        random_state=1,
        early_stop=False,
    )

    assert result.match_result.catch is True
    assert result.artifacts.centroids.shape == (1, 200)
    assert result.artifacts.cluster_matrix.shape == (1, 2, 50)
    assert result.match_debug.checked_columns == 2


def test_protocol_returns_no_catch_for_known_different_name():
    result = run_single_query_protocol(
        ["john smith", "mary jones"],
        "zzzz qqqq",
        k_mode="sqrt",
        random_state=1,
        early_stop=False,
    )

    assert result.match_result.catch is False
    assert result.artifacts.centroids.shape == (1, 200)
    assert result.artifacts.cluster_matrix.shape == (1, 2, 50)
    assert result.match_debug.checked_columns == 2


if __name__ == "__main__":
    test_protocol_returns_catch_for_known_similar_name()
    test_protocol_returns_no_catch_for_known_different_name()
    print("real end-to-end tests passed")
