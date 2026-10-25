"""Tests for V2 tiled first round: centroid scoring and cluster selection."""

import numpy as np
import pytest

from ckks.context import create_ckks_context
from ckks.tiling import make_slot_tile_layout, tile_count
from party_a.local_prep import prepare_tiled_query_batch
from party_a.online_querier import choose_clusters_and_build_tiled_request
from party_b.online_responder import compare_tiled_batch_to_centroids
from protocol.transport import serialize_tiled_first_round_request
from protocol.types import TiledSecondRoundRequest


@pytest.fixture(scope="module")
def mock_scaler():
    np.random.seed(42)
    mean = np.random.uniform(-1.0, 1.0, size=200)
    scale = np.random.uniform(0.5, 2.0, size=200)
    return mean, scale


@pytest.fixture(scope="module")
def ckks_ctx():
    return create_ckks_context()


@pytest.mark.parametrize("k", [5, 20, 21, 50])
def test_compare_tiled_batch_to_centroids_numerics(mock_scaler, ckks_ctx, k):
    mean, scale = mock_scaler
    names = ["JOHN SMITH", "ALICE BROWN", "BOB WILSON"]
    m = len(names)
    layout = make_slot_tile_layout(m)

    req, _ = prepare_tiled_query_batch(names, mean, scale, context=ckks_ctx)
    centroids = np.random.uniform(-1.0, 1.0, size=(k, 200))

    from party_a.local_prep import encode_query_batch

    q200_plain, _ = encode_query_batch(names, mean, scale)

    score_tiles = compare_tiled_batch_to_centroids(req, centroids)
    assert len(score_tiles) == tile_count(k, layout)

    # Decrypt and concatenate
    from ckks.tiling import decrypt_tiled_score_tile, iter_tile_slices

    arrays = []
    for tile_idx, (_, _, valid_width) in enumerate(iter_tile_slices(k, layout)):
        tile = decrypt_tiled_score_tile(
            score_tiles[tile_idx], ckks_ctx, layout, valid_width
        )
        arrays.append(tile)
    scores = np.concatenate(arrays, axis=1)

    expected = q200_plain @ centroids.T
    assert scores.shape == expected.shape
    np.testing.assert_allclose(scores, expected, atol=1e-4)


def test_compare_tiled_batch_to_centroids_m200_k50_outputs_3_tiles(
    mock_scaler, ckks_ctx
):
    mean, scale = mock_scaler
    names = [f"NAME_{i}" for i in range(200)]
    k = 50

    req, _ = prepare_tiled_query_batch(names, mean, scale, context=ckks_ctx)
    centroids = np.random.uniform(-1.0, 1.0, size=(k, 200))

    score_tiles = compare_tiled_batch_to_centroids(req, centroids)
    assert len(score_tiles) == 3
    for c in score_tiles:
        assert c.size() == 4000


def test_compare_tiled_batch_to_centroids_bytes_path(mock_scaler, ckks_ctx):
    mean, scale = mock_scaler
    names = ["JOHN SMITH", "ALICE BROWN"]
    k = 7

    req, _ = prepare_tiled_query_batch(names, mean, scale, context=ckks_ctx)
    wire_req = serialize_tiled_first_round_request(req)
    centroids = np.random.uniform(-1.0, 1.0, size=(k, 200))

    score_tiles = compare_tiled_batch_to_centroids(
        wire_req, centroids, serialize_output=True
    )
    assert all(isinstance(c, bytes) for c in score_tiles)


def test_choose_clusters_and_build_tiled_request(mock_scaler, ckks_ctx):
    mean, scale = mock_scaler
    names = ["QUERY_0", "QUERY_1", "QUERY_2"]
    k = 5

    req, state = prepare_tiled_query_batch(names, mean, scale, context=ckks_ctx)
    centroids = np.random.uniform(-1.0, 1.0, size=(k, 200))
    score_tiles = compare_tiled_batch_to_centroids(req, centroids)

    req2, debug = choose_clusters_and_build_tiled_request(score_tiles, state, k=k)

    assert isinstance(req2, TiledSecondRoundRequest)
    assert req2.layout == state.layout
    assert len(req2.encrypted_query_50) == 50
    assert len(req2.encrypted_selectors) == k
    for s in req2.encrypted_selectors:
        assert s.size() == state.layout.active_slots

    # Verify selected clusters match plaintext argmax
    from party_a.local_prep import encode_query_batch

    q200_plain, _ = encode_query_batch(names, mean, scale)
    expected_clusters = np.argmax(q200_plain @ centroids.T, axis=1)
    np.testing.assert_array_equal(debug.selected_clusters, expected_clusters)
