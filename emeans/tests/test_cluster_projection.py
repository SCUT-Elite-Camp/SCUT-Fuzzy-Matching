"""Random-projection clustering and independent tiled-probe integration tests."""

import numpy as np
import pytest
import tenseal as ts

from ckks.tiling import (
    decrypt_tiled_score_tile,
    iter_tile_slices,
    make_slot_tile_layout,
)
from clustering.projection import project_normalized_signatures
from party_a.local_prep import encode_query_batch, prepare_tiled_query_batch
from party_a.online_querier import (
    check_tiled_score_batch_debug,
    choose_clusters_and_build_tiled_request,
)
from party_b.offline_prep import (
    load_offline_artifacts,
    prepare_party_b_offline,
    save_offline_artifacts,
)
from party_b.online_responder import (
    compare_tiled_batch_to_centroids,
    tiled_batch_matching,
)
from protocol.orchestrator import run_single_query_protocol
from protocol.transport import (
    serialize_tiled_first_round_request,
    serialize_tiled_second_round_request,
)

_NAMES = [
    "JOHN SMITH",
    "MARY JONES",
    "ALICE BROWN",
    "BOB WILSON",
    "CHARLIE DAVIS",
]


@pytest.fixture(scope="module")
def projected_artifacts():
    return prepare_party_b_offline(_NAMES, k_mode=2, projection_dim=4)


def test_projection_is_deterministic_and_unit_normalized():
    matrix = np.arange(1, 41, dtype=np.float64).reshape(4, 10)
    first = project_normalized_signatures(matrix, 5, 42)
    second = project_normalized_signatures(matrix, 5, 42)
    different = project_normalized_signatures(matrix, 5, 43)

    np.testing.assert_allclose(first, second)
    assert not np.allclose(first, different)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0)
    with pytest.raises(ValueError, match="output_dim"):
        project_normalized_signatures(matrix, 0, 42)
    with pytest.raises(ValueError, match="output_dim"):
        project_normalized_signatures(matrix, 11, 42)


def test_offline_projection_metadata_round_trip(projected_artifacts, tmp_path):
    artifacts = projected_artifacts
    assert artifacts.centroids.shape == (2, 4)
    assert artifacts.cluster_matrix.shape[2] == 50
    assert artifacts.scaler_mean.size == 0
    assert artifacts.scaler_scale.size == 0

    save_offline_artifacts(artifacts, tmp_path)
    loaded = load_offline_artifacts(tmp_path)
    assert loaded.projection_dim == 4
    assert loaded.projection_seed == 42
    np.testing.assert_allclose(loaded.centroids, artifacts.centroids)
    np.testing.assert_allclose(loaded.cluster_matrix, artifacts.cluster_matrix)


def test_projected_first_round_matches_plaintext(projected_artifacts):
    artifacts = projected_artifacts
    req1, state = prepare_tiled_query_batch(
        _NAMES[:2],
        artifacts.scaler_mean,
        artifacts.scaler_scale,
        projection_dim=artifacts.projection_dim,
        projection_seed=artifacts.projection_seed,
    )
    wire_req1 = serialize_tiled_first_round_request(req1)
    assert len(wire_req1.encrypted_query_200) == 4

    encrypted = compare_tiled_batch_to_centroids(
        wire_req1, artifacts.centroids, serialize_output=True
    )
    actual = []
    for item, (_, _, valid_width) in zip(
        encrypted,
        iter_tile_slices(artifacts.centroids.shape[0], state.layout),
        strict=True,
    ):
        actual.append(
            decrypt_tiled_score_tile(
                item, state.secret_context, state.layout, valid_width
            )
        )
    actual_scores = np.concatenate(actual, axis=1)
    query_cluster, _ = encode_query_batch(
        _NAMES[:2],
        artifacts.scaler_mean,
        artifacts.scaler_scale,
        projection_dim=4,
        projection_seed=42,
    )
    expected_scores = query_cluster @ artifacts.centroids.T
    np.testing.assert_allclose(actual_scores, expected_scores, rtol=0.0, atol=1e-4)


def _run_probe_case(cluster_matrix, scores, probes):
    artifacts = prepare_party_b_offline(_NAMES, k_mode=2)
    req1, state = prepare_tiled_query_batch(
        [_NAMES[0]], artifacts.scaler_mean, artifacts.scaler_scale
    )
    padded_scores = scores + [0.0] * (state.layout.active_slots - len(scores))
    score_tile = ts.ckks_vector(state.secret_context, padded_scores)
    req2, debug = choose_clusters_and_build_tiled_request(
        [score_tile.serialize()], state, k=len(scores), probes=probes
    )
    wire_req2 = serialize_tiled_second_round_request(req2)
    encrypted = tiled_batch_matching(
        cluster_matrix,
        wire_req2,
        req1.public_context_bytes,
        tau=0.9,
        serialize_output=True,
    )
    result, _ = check_tiled_score_batch_debug(
        encrypted,
        state.secret_context,
        layout=req2.layout,
        logical_width=cluster_matrix.shape[1],
        early_stop=False,
        probe_count=debug.probe_count,
        original_batch_size=1,
    )
    return bool(result.catches[0]), req2, debug


def test_second_probe_recovers_match_without_changing_final_threshold():
    artifacts = prepare_party_b_offline(_NAMES, k_mode=2)
    _, state = prepare_tiled_query_batch(
        [_NAMES[0]], artifacts.scaler_mean, artifacts.scaler_scale
    )
    query_50 = state.query_50_norm[0]
    cluster_matrix = np.zeros((2, 1, 50), dtype=np.float64)
    cluster_matrix[1, 0] = query_50

    top1, _, _ = _run_probe_case(cluster_matrix, [0.9, 0.1], probes=1)
    top2, req2, debug = _run_probe_case(cluster_matrix, [0.9, 0.1], probes=2)

    assert top1 is False
    assert top2 is True
    assert debug.probed_clusters.tolist() == [0, 1]
    assert req2.layout.batch_size == 2
    assert not hasattr(req2, "query_50_norm")
    assert not hasattr(req2, "probed_clusters")


def test_independent_probes_never_sum_cross_cluster_similarities():
    artifacts = prepare_party_b_offline(_NAMES, k_mode=2)
    _, state = prepare_tiled_query_batch(
        [_NAMES[0]], artifacts.scaler_mean, artifacts.scaler_scale
    )
    query_50 = state.query_50_norm[0]
    cluster_matrix = np.zeros((2, 1, 50), dtype=np.float64)
    cluster_matrix[0, 0] = 0.6 * query_50
    cluster_matrix[1, 0] = 0.6 * query_50

    top2, _, _ = _run_probe_case(cluster_matrix, [0.9, 0.8], probes=2)
    assert top2 is False


def test_probe_early_stop_tracks_original_query_not_every_probe():
    artifacts = prepare_party_b_offline(_NAMES, k_mode=2)
    _, state = prepare_tiled_query_batch(
        [_NAMES[0]], artifacts.scaler_mean, artifacts.scaler_scale
    )
    layout = make_slot_tile_layout(2)
    first_tile = [1.0] * layout.tile_width + [0.0] * layout.tile_width
    encrypted = [
        ts.ckks_vector(state.secret_context, first_tile),
        ts.ckks_vector(state.secret_context, [0.0] * layout.active_slots),
    ]

    early_result, early_debug = check_tiled_score_batch_debug(
        encrypted,
        state.secret_context,
        layout=layout,
        logical_width=layout.tile_width + 1,
        early_stop=True,
        probe_count=2,
        original_batch_size=1,
    )
    _, full_debug = check_tiled_score_batch_debug(
        encrypted,
        state.secret_context,
        layout=layout,
        logical_width=layout.tile_width + 1,
        early_stop=False,
        probe_count=2,
        original_batch_size=1,
    )

    assert early_result.catches.tolist() == [True]
    assert early_debug.checked_tiles == 1
    assert full_debug.checked_tiles == 2


def test_probe_aggregation_preserves_query_major_order():
    artifacts = prepare_party_b_offline(_NAMES, k_mode=2)
    _, state = prepare_tiled_query_batch(
        _NAMES[:2], artifacts.scaler_mean, artifacts.scaler_scale
    )
    layout = make_slot_tile_layout(4)
    width = layout.tile_width
    values = np.concatenate(
        [
            np.full(width, -1.0),
            np.full(width, 1.0),
            np.full(width, -1.0),
            np.full(width, -1.0),
        ]
    )
    encrypted = [ts.ckks_vector(state.secret_context, values.tolist())]

    result, _ = check_tiled_score_batch_debug(
        encrypted,
        state.secret_context,
        layout=layout,
        logical_width=1,
        early_stop=False,
        probe_count=2,
        original_batch_size=2,
    )
    assert result.catches.tolist() == [True, False]


def test_single_query_projection_dimension_one_remains_a_vector():
    run = run_single_query_protocol(
        _NAMES,
        _NAMES[0],
        k_mode=2,
        projection_dim=1,
        early_stop=False,
    )
    assert run.artifacts.centroids.shape == (2, 1)
    assert isinstance(run.match_result.catch, bool)
