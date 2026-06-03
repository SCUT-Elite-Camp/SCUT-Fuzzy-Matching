"""Interface shape checks against real modules."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minhash.encoder import batch_encode
from party_a.local_prep import encode_query_vectors, prepare_encrypted_query
from party_a.online_querier import build_selector, choose_cluster_and_build_request
from party_b.offline_prep import prepare_party_b_offline
from party_b.online_responder import compare_to_centroids, column_wise_matching
from protocol.types import FirstRoundRequest, MatchResult, SecondRoundRequest


def test_member1_offline_artifact_shapes():
    artifacts = prepare_party_b_offline(
        ["john smith", "mary jones", "alice brown"],
        random_state=1,
    )

    assert artifacts.centroids.shape == (1, 200)
    assert artifacts.cluster_matrix.shape == (1, 3, 50)
    assert artifacts.scaler_mean.shape == (200,)
    assert artifacts.scaler_scale.shape == (200,)
    assert artifacts.cluster_assignments.shape == (3,)
    assert artifacts.max_size == 3


def test_minhash_el50_is_prefix_of_el200():
    signatures_200 = batch_encode(["john smith", "mary jones"], 200)
    signatures_50 = batch_encode(["john smith", "mary jones"], 50)

    assert signatures_200.shape == (2, 200)
    assert signatures_50.shape == (2, 50)
    np.testing.assert_array_equal(signatures_50, signatures_200[:, :50])


def test_member2_query_shapes_and_first_round_boundary():
    artifacts = prepare_party_b_offline(["john smith", "mary jones"], random_state=1)

    query_200_std, query_50_norm = encode_query_vectors(
        "john smith",
        artifacts.scaler_mean,
        artifacts.scaler_scale,
    )
    first_round_request, party_a_state = prepare_encrypted_query(
        "john smith",
        artifacts.scaler_mean,
        artifacts.scaler_scale,
    )

    assert query_200_std.shape == (200,)
    assert query_50_norm.shape == (50,)
    assert isinstance(first_round_request, FirstRoundRequest)
    assert not hasattr(first_round_request, "encrypted_query_50")
    assert party_a_state.encrypted_query_50.size() == 50


def test_member3_selector_and_second_round_shapes():
    selected_cluster, selector = build_selector(np.array([0.1, 0.9]), k=2)

    assert selected_cluster == 1
    assert selector.shape == (2,)
    assert selector.sum() == 1.0


def test_member1_to_member4_online_shapes():
    artifacts = prepare_party_b_offline(["john smith", "mary jones"], random_state=1)
    first_round_request, party_a_state = prepare_encrypted_query(
        "john smith",
        artifacts.scaler_mean,
        artifacts.scaler_scale,
    )
    encrypted_sim_scores = compare_to_centroids(
        first_round_request,
        artifacts.centroids,
    )
    second_round_request, debug = choose_cluster_and_build_request(
        encrypted_sim_scores,
        party_a_state,
        k=artifacts.centroids.shape[0],
    )
    encrypted_scores = list(
        column_wise_matching(
            artifacts.cluster_matrix,
            second_round_request,
            first_round_request.public_context_bytes,
        )
    )

    assert len(encrypted_sim_scores) == artifacts.centroids.shape[0]
    assert isinstance(second_round_request, SecondRoundRequest)
    assert not hasattr(second_round_request, "selected_cluster")
    assert debug.selected_cluster == 0
    assert len(encrypted_scores) == artifacts.max_size
    assert isinstance(MatchResult(catch=True).catch, bool)


if __name__ == "__main__":
    test_member1_offline_artifact_shapes()
    test_minhash_el50_is_prefix_of_el200()
    test_member2_query_shapes_and_first_round_boundary()
    test_member3_selector_and_second_round_shapes()
    test_member1_to_member4_online_shapes()
    print("real interface shape tests passed")
