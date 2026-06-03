"""Script-friendly integration test for the real protocol flow."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.orchestrator import run_single_query_protocol


def run_integration_test() -> None:
    names_b = ["john smith", "mary jones"]

    positive = run_single_query_protocol(
        names_b,
        "john smith",
        k_mode="sqrt",
        random_state=1,
        early_stop=False,
    )
    negative = run_single_query_protocol(
        names_b,
        "zzzz qqqq",
        k_mode="sqrt",
        random_state=1,
        early_stop=False,
    )

    assert positive.match_result.catch is True
    assert negative.match_result.catch is False
    assert positive.artifacts.centroids.shape == (1, 200)
    assert positive.artifacts.cluster_matrix.shape == (1, 2, 50)
    assert positive.match_debug.checked_columns == 2
    assert negative.match_debug.checked_columns == 2


if __name__ == "__main__":
    run_integration_test()
    print("real integration test passed")
