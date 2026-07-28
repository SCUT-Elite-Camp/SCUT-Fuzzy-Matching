"""Security-boundary checks against real protocol interfaces."""

import inspect
import os
import sys

import numpy as np
import pytest
import tenseal as ts


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ckks.batching import load_feature_batch
from party_a.local_prep import prepare_encrypted_query, prepare_encrypted_query_batch
from party_a.online_querier import (
    check_encrypted_score_batch,
    check_encrypted_scores,
    choose_cluster_and_build_request,
    choose_clusters_and_build_batch_request,
)
from party_b.offline_prep import prepare_party_b_offline
from party_b.online_responder import (
    _sample_positive_mask,
    column_wise_batch_matching,
    column_wise_matching,
    compare_batch_to_centroids,
)
from protocol.orchestrator import run_batch_query_protocol
from evaluation.benchmark import benchmark
from party_a.online_querier import check_tiled_score_batch_debug
from party_b.online_responder import (
    compare_tiled_batch_to_centroids,
    tiled_batch_matching,
)
from protocol.types import (
    BatchFirstRoundRequest,
    BatchMatchResult,
    BatchSecondRoundRequest,
    FirstRoundRequest,
    MatchResult,
    SecondRoundRequest,
)
from protocol.transport import (
    serialize_batch_first_round_request,
    serialize_batch_second_round_request,
)


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


def test_batch_round1_request_does_not_include_short_ciphertext():
    artifacts = prepare_party_b_offline(["john smith", "mary jones"], random_state=1)
    batch_req1, _ = prepare_encrypted_query_batch(
        ["john smith"],
        artifacts.scaler_mean,
        artifacts.scaler_scale,
    )

    assert isinstance(batch_req1, BatchFirstRoundRequest)
    assert hasattr(batch_req1, "encrypted_query_200")
    assert not hasattr(batch_req1, "encrypted_query_50")


def test_public_context_cannot_decrypt_query_ciphertext():
    _, first_round_request, _ = _build_round1()
    public_context = ts.Context.load(first_round_request.public_context_bytes)
    public_ciphertext = ts.ckks_vector_from(
        public_context,
        first_round_request.encrypted_query_200.serialize(),
    )

    with pytest.raises(ValueError, match="secret_key"):
        public_ciphertext.decrypt()


def test_batch_wire_requests_rebind_ciphertexts_to_public_context():
    artifacts = prepare_party_b_offline(["john smith", "mary jones"], random_state=1)
    request, state = prepare_encrypted_query_batch(
        ["john smith"], artifacts.scaler_mean, artifacts.scaler_scale
    )
    wire_request = serialize_batch_first_round_request(request)

    assert all(isinstance(item, bytes) for item in wire_request.encrypted_query_200)
    public_context = ts.Context.load(wire_request.public_context_bytes)
    loaded_query = load_feature_batch(
        wire_request.encrypted_query_200,
        public_context,
        feature_count=200,
        batch_size=1,
    )
    assert public_context.is_private() is False
    assert loaded_query[0].context().is_private() is False
    with pytest.raises(ValueError, match="secret_key"):
        loaded_query[0].decrypt()

    second_request, _ = choose_clusters_and_build_batch_request(
        [ts.ckks_vector(state.secret_context, [1.0])], state, k=1
    )
    wire_second_request = serialize_batch_second_round_request(second_request)
    assert all(isinstance(item, bytes) for item in wire_second_request.encrypted_query_50)
    assert all(isinstance(item, bytes) for item in wire_second_request.encrypted_selectors)


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


def test_batch_round2_request_does_not_include_plaintext_cluster():
    artifacts = prepare_party_b_offline(["john smith", "mary jones"], random_state=1)
    _, batch_state = prepare_encrypted_query_batch(
        ["john smith"],
        artifacts.scaler_mean,
        artifacts.scaler_scale,
    )
    sim_ct = [ts.ckks_vector(batch_state.secret_context, [1.0])]
    batch_req2, debug = choose_clusters_and_build_batch_request(
        sim_ct, batch_state, k=1
    )

    assert isinstance(batch_req2, BatchSecondRoundRequest)
    assert hasattr(batch_req2, "encrypted_selectors")
    assert not hasattr(batch_req2, "selected_clusters")
    np.testing.assert_array_equal(debug.selected_clusters, [0])


def test_b_side_functions_do_not_accept_secret_context():
    sig1 = inspect.signature(column_wise_matching)
    assert "secret_context" not in sig1.parameters

    sig2 = inspect.signature(column_wise_batch_matching)
    assert "secret_context" not in sig2.parameters

    sig3 = inspect.signature(compare_batch_to_centroids)
    assert "secret_context" not in sig3.parameters

    assert "secret_context" not in inspect.signature(
        compare_tiled_batch_to_centroids
    ).parameters
    assert "secret_context" not in inspect.signature(tiled_batch_matching).parameters
    assert ".decrypt(" not in inspect.getsource(compare_tiled_batch_to_centroids)
    assert ".decrypt(" not in inspect.getsource(tiled_batch_matching)

    orchestrator_signature = inspect.signature(run_batch_query_protocol)
    assert orchestrator_signature.parameters["serialize_communication"].default is True


def test_production_batch_entrypoints_do_not_call_legacy_column_kernel():
    forbidden = (
        "prepare_encrypted_query_batch",
        "compare_batch_to_centroids",
        "column_wise_batch_matching",
        "check_encrypted_score_batch_debug",
    )
    orchestrator_source = inspect.getsource(run_batch_query_protocol)
    benchmark_source = inspect.getsource(benchmark)
    for name in forbidden:
        assert name not in orchestrator_source
        assert name not in benchmark_source

    tiled_judger_signature = inspect.signature(check_tiled_score_batch_debug)
    assert "cluster_matrix" not in tiled_judger_signature.parameters


def test_a_side_final_judger_does_not_accept_cluster_matrix():
    sig1 = inspect.signature(check_encrypted_scores)
    assert "cluster_matrix" not in sig1.parameters

    sig2 = inspect.signature(check_encrypted_score_batch)
    assert "cluster_matrix" not in sig2.parameters


def test_match_result_only_exposes_catch():
    result = MatchResult(catch=True)

    assert result.catch is True
    assert not hasattr(result, "checked_columns")
    assert not hasattr(result, "first_positive_column")

    batch_result = BatchMatchResult(catches=np.array([True, False]))
    np.testing.assert_array_equal(batch_result.catches, [True, False])
    assert not hasattr(batch_result, "checked_columns")
    assert not hasattr(batch_result, "first_positive_columns")


def test_random_mask_is_positive_bounded_and_varies():
    values = [_sample_positive_mask() for _ in range(10)]

    assert all(1.0 <= value <= 10.0 for value in values)
    assert len(set(values)) > 1


if __name__ == "__main__":
    test_round1_request_does_not_include_short_ciphertext()
    test_batch_round1_request_does_not_include_short_ciphertext()
    test_public_context_cannot_decrypt_query_ciphertext()
    test_round2_request_does_not_include_plaintext_cluster()
    test_batch_round2_request_does_not_include_plaintext_cluster()
    test_b_side_functions_do_not_accept_secret_context()
    test_a_side_final_judger_does_not_accept_cluster_matrix()
    test_match_result_only_exposes_catch()
    test_random_mask_is_positive_bounded_and_varies()
    print("real security boundary tests passed")
