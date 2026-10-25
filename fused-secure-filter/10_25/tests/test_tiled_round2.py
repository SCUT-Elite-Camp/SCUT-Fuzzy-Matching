"""Tests for V2 tiled second round: query-column matching."""

from unittest.mock import patch

import numpy as np
import pytest

from ckks.context import create_ckks_context
from ckks.tiling import iter_tile_slices, make_slot_tile_layout, tile_count
from party_a.local_prep import prepare_tiled_query_batch
from party_a.online_querier import (
    check_tiled_score_batch_debug,
    choose_clusters_and_build_tiled_request,
)
from party_b.online_responder import (
    _sample_positive_mask_matrix,
    compare_tiled_batch_to_centroids,
    tiled_batch_matching,
)
from protocol.transport import serialize_tiled_second_round_request


@pytest.fixture(scope="module")
def mock_scaler():
    np.random.seed(42)
    mean = np.random.uniform(-1.0, 1.0, size=200)
    scale = np.random.uniform(0.5, 2.0, size=200)
    return mean, scale


@pytest.fixture(scope="module")
def ckks_ctx():
    return create_ckks_context()


def build_tiled_second_request(names, centroids, cluster_matrix, mock_scaler, ckks_ctx):
    mean, scale = mock_scaler
    req1, state = prepare_tiled_query_batch(names, mean, scale, context=ckks_ctx)
    score_tiles = compare_tiled_batch_to_centroids(req1, centroids)
    req2, debug = choose_clusters_and_build_tiled_request(
        score_tiles, state, k=centroids.shape[0]
    )
    return req2, debug, state, req1.public_context_bytes


def test_tiled_batch_matching_small_numerics(mock_scaler, ckks_ctx):
    mean, scale = mock_scaler
    names = ["Q0", "Q1", "Q2"]
    m = len(names)
    k = 2
    L = 7
    d = 50
    layout = make_slot_tile_layout(m)

    centroids = np.random.uniform(-1.0, 1.0, size=(k, 200))
    cluster_matrix = np.random.uniform(-1.0, 1.0, size=(k, L, d))

    req2, debug, _, public_context_bytes = build_tiled_second_request(
        names, centroids, cluster_matrix, mock_scaler, ckks_ctx
    )

    fixed_mask = np.full((m, layout.tile_width), 1.0, dtype=np.float64)
    with patch(
        "party_b.online_responder._sample_positive_mask_matrix",
        return_value=fixed_mask,
    ):
        score_tiles = list(
            tiled_batch_matching(cluster_matrix, req2, public_context_bytes, tau=0.5)
        )
    assert len(score_tiles) == tile_count(L, layout)

    # Decrypt and verify against plaintext reference.
    from ckks.tiling import decrypt_tiled_score_tile
    from party_a.local_prep import encode_query_batch

    _, q50_plain = encode_query_batch(names, mean, scale)
    selected = debug.selected_clusters

    arrays = []
    for tile_idx, (_, _, valid_width) in enumerate(iter_tile_slices(L, layout)):
        tile = decrypt_tiled_score_tile(
            score_tiles[tile_idx], ckks_ctx, layout, valid_width
        )
        arrays.append(tile)
    scores = np.concatenate(arrays, axis=1)

    expected = np.array(
        [
            [
                np.dot(q50_plain[q], cluster_matrix[selected[q], j]) - 0.5
                for j in range(L)
            ]
            for q in range(m)
        ]
    )
    np.testing.assert_allclose(scores, expected, rtol=0.0, atol=1e-4)


def test_tiled_batch_matching_m200_L483_outputs_25_tiles(mock_scaler, ckks_ctx):
    names = [f"NAME_{i}" for i in range(200)]
    k = 50
    L = 483
    d = 50

    centroids = np.random.uniform(-1.0, 1.0, size=(k, 200))
    cluster_matrix = np.random.uniform(-1.0, 1.0, size=(k, L, d))

    req2, _, _, public_context_bytes = build_tiled_second_request(
        names, centroids, cluster_matrix, mock_scaler, ckks_ctx
    )

    # Patch mask to fixed positive values for deterministic testing.
    fixed_mask = np.full((200, 20), 2.0, dtype=np.float64)
    with patch(
        "party_b.online_responder._sample_positive_mask_matrix",
        return_value=fixed_mask,
    ):
        score_tiles = list(
            tiled_batch_matching(
                cluster_matrix,
                req2,
                public_context_bytes,
                tau=0.9,
                serialize_output=True,
            )
        )

    assert len(score_tiles) == 25
    assert all(isinstance(c, bytes) for c in score_tiles)


def test_tiled_batch_matching_mask_is_positive_and_independent():
    R = _sample_positive_mask_matrix(2, 5)
    assert R.shape == (2, 5)
    assert np.all(R > 0)
    assert np.all(R >= 1.0) and np.all(R <= 10.0)
    # With high probability independent samples differ; check not all identical.
    assert not np.allclose(R, R[0, 0])


def test_tiled_batch_matching_serialized_bytes(mock_scaler, ckks_ctx):
    names = ["Q0", "Q1"]
    k = 2
    L = 3
    d = 50

    centroids = np.random.uniform(-1.0, 1.0, size=(k, 200))
    cluster_matrix = np.random.uniform(-1.0, 1.0, size=(k, L, d))

    req2, _, _, public_context_bytes = build_tiled_second_request(
        names, centroids, cluster_matrix, mock_scaler, ckks_ctx
    )
    wire_req = serialize_tiled_second_round_request(req2)

    score_tiles = list(
        tiled_batch_matching(
            cluster_matrix,
            wire_req,
            public_context_bytes,
            tau=0.5,
            serialize_output=True,
        )
    )
    assert all(isinstance(c, bytes) for c in score_tiles)


def test_tiled_batch_matching_hit_detection(mock_scaler, ckks_ctx):
    mean, scale = mock_scaler
    names = ["JOHN SMITH", "ALICE BROWN"]
    np.random.seed(456)
    m = len(names)
    k = 2
    L = 10
    d = 50
    layout = make_slot_tile_layout(m)

    centroids = np.random.uniform(-1.0, 1.0, size=(k, 200))
    cluster_matrix = np.random.uniform(-1.0, 1.0, size=(k, L, d))

    req2, debug, _, public_context_bytes = build_tiled_second_request(
        names, centroids, cluster_matrix, mock_scaler, ckks_ctx
    )

    fixed_mask = np.full((m, layout.tile_width), 1.0, dtype=np.float64)
    with patch(
        "party_b.online_responder._sample_positive_mask_matrix",
        return_value=fixed_mask,
    ):
        score_tiles = list(
            tiled_batch_matching(
                cluster_matrix,
                req2,
                public_context_bytes,
                tau=0.5,
                serialize_output=False,
            )
        )

    result, match_debug = check_tiled_score_batch_debug(
        score_tiles,
        ckks_ctx,
        layout=layout,
        logical_width=L,
        early_stop=True,
        eps=1e-4,
    )

    from party_a.local_prep import encode_query_batch

    _, q50_plain = encode_query_batch(names, mean, scale)
    selected = debug.selected_clusters

    # Verify decrypted scores align with plaintext reference; first-positive
    # column equality follows if scores are within tolerance.
    from ckks.tiling import decrypt_tiled_score_tile

    arrays = []
    for tile_idx, (_, _, valid_width) in enumerate(iter_tile_slices(L, layout)):
        tile = decrypt_tiled_score_tile(
            score_tiles[tile_idx], ckks_ctx, layout, valid_width
        )
        arrays.append(tile)
    scores = np.concatenate(arrays, axis=1)
    expected = np.array(
        [
            [
                np.dot(q50_plain[q], cluster_matrix[selected[q], j]) - 0.5
                for j in range(L)
            ]
            for q in range(m)
        ]
    )
    np.testing.assert_allclose(scores, expected, rtol=0.0, atol=1e-4)

    # Hit results must be consistent with the decrypted scores.
    for q in range(m):
        positive = expected[q] > 1e-4
        has_hit = positive.any()
        if has_hit:
            assert result.catches[q]
            assert match_debug.first_positive_columns[q] == positive.argmax()
        else:
            assert not result.catches[q]
            assert match_debug.first_positive_columns[q] == -1
