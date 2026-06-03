"""Security-boundary checks against real protocol interfaces."""

import inspect
import os
import sys

import pytest
import tenseal as ts

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from party_a.local_prep import prepare_encrypted_query
from party_a.online_querier import check_encrypted_scores, choose_cluster_and_build_request
from party_b.offline_prep import prepare_party_b_offline
from party_b.online_responder import _sample_positive_mask, column_wise_matching
from protocol.types import FirstRoundRequest, MatchResult, SecondRoundRequest


def _build_round1():
    artifacts = prepare_party_b_offline(["john smith", "mary jones"], random_state=1)
    first_round_request, party_a_state = prepare_encrypted_query(
        "john smith",
        artifacts.scaler_mean,
        artifacts.scaler_scale,
    )
    return artifacts, first_round_request, party_a_state


def test_round1_request_does_not_include_short_ciphertext():
    _, first_round_request, _ = _build_round1()

    assert isinstance(first_round_request, FirstRoundRequest)
    assert hasattr(first_round_request, "encrypted_query_200")
    assert not hasattr(first_round_request, "encrypted_query_50")


def test_public_context_cannot_decrypt_query_ciphertext():
    _, first_round_request, _ = _build_round1()
    public_context = ts.Context.load(first_round_request.public_context_bytes)
    public_ciphertext = ts.ckks_vector_from(
        public_context,
        first_round_request.encrypted_query_200.serialize(),
    )

    with pytest.raises(ValueError, match="secret_key"):
        public_ciphertext.decrypt()


def test_round2_request_does_not_include_plaintext_cluster():
    _, _, party_a_state = _build_round1()
    second_round_request, debug = choose_cluster_and_build_request(
        [ts.ckks_vector(party_a_state.secret_context, [1.0])],
        party_a_state,
        k=1,
    )

    assert isinstance(second_round_request, SecondRoundRequest)
    assert hasattr(second_round_request, "encrypted_selector")
    assert not hasattr(second_round_request, "selected_cluster")
    assert debug.selected_cluster == 0


def test_b_side_functions_do_not_accept_secret_context():
    signature = inspect.signature(column_wise_matching)

    assert "secret_context" not in signature.parameters


def test_a_side_final_judger_does_not_accept_cluster_matrix():
    signature = inspect.signature(check_encrypted_scores)

    assert "cluster_matrix" not in signature.parameters


def test_match_result_only_exposes_catch():
    result = MatchResult(catch=True)

    assert result.catch is True
    assert not hasattr(result, "checked_columns")
    assert not hasattr(result, "first_positive_column")


def test_random_mask_is_positive_bounded_and_varies():
    values = [_sample_positive_mask() for _ in range(10)]

    assert all(1.0 <= value <= 10.0 for value in values)
    assert len(set(values)) > 1


if __name__ == "__main__":
    test_round1_request_does_not_include_short_ciphertext()
    test_public_context_cannot_decrypt_query_ciphertext()
    test_round2_request_does_not_include_plaintext_cluster()
    test_b_side_functions_do_not_accept_secret_context()
    test_a_side_final_judger_does_not_accept_cluster_matrix()
    test_match_result_only_exposes_catch()
    test_random_mask_is_positive_bounded_and_varies()
    print("real security boundary tests passed")
