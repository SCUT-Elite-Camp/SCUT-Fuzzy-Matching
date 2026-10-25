"""Tests for CKKS 2-D query-major slot tiling primitives (ckks/tiling.py)."""

import numpy as np
import pytest

from ckks.tiling import (
    SLOT_CAPACITY,
    expand_plain_tile,
    iter_tile_slices,
    make_slot_tile_layout,
    pack_candidate_tile,
    pack_centroid_tile,
    pack_selector_tile,
    repeat_query_rows,
    tile_count,
    unpack_score_tile,
)

# ---------------------------------------------------------------------------
# Layout parameter tests
# ---------------------------------------------------------------------------


def test_layout_m1_max_tile():
    layout = make_slot_tile_layout(1)
    assert layout.tile_width == SLOT_CAPACITY
    assert layout.active_slots == SLOT_CAPACITY
    assert layout.slot_capacity == SLOT_CAPACITY


def test_layout_m200():
    layout = make_slot_tile_layout(200)
    assert layout.tile_width == 20
    assert layout.active_slots == 4000
    assert layout.slot_capacity == SLOT_CAPACITY


def test_layout_m4096():
    layout = make_slot_tile_layout(SLOT_CAPACITY)
    assert layout.tile_width == 1
    assert layout.active_slots == SLOT_CAPACITY


def test_layout_invalid_inputs():
    with pytest.raises(ValueError, match="batch_size"):
        make_slot_tile_layout(0)
    with pytest.raises(ValueError, match="batch_size"):
        make_slot_tile_layout(SLOT_CAPACITY + 1)
    with pytest.raises(ValueError, match="batch_size"):
        make_slot_tile_layout(True)
    with pytest.raises(ValueError, match="batch_size"):
        make_slot_tile_layout(1.5)


# ---------------------------------------------------------------------------
# Tile count / slice tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "m, w, expected",
    [
        (200, 50, 3),  # current R1
        (200, 483, 25),  # current R2
        (200, 20, 1),
        (200, 21, 2),
        (1, 5, 1),
        (4096, 5, 5),
        (10, 0, 0),
    ],
)
def test_tile_count(m, w, expected):
    layout = make_slot_tile_layout(m)
    assert tile_count(w, layout) == expected


def test_iter_tile_slices_padding():
    layout = make_slot_tile_layout(200)  # T=20
    slices = list(iter_tile_slices(483, layout))
    assert len(slices) == 25
    assert slices[0] == (0, 0, 20)
    assert slices[24] == (24, 480, 3)
    total_valid = sum(v for _, _, v in slices)
    assert total_valid == 483


def test_iter_tile_slices_exact():
    layout = make_slot_tile_layout(200)  # T=20
    slices = list(iter_tile_slices(40, layout))
    assert len(slices) == 2
    assert all(v == 20 for _, _, v in slices)


# ---------------------------------------------------------------------------
# Packing / unpacking tests
# ---------------------------------------------------------------------------


def test_repeat_query_rows():
    layout = make_slot_tile_layout(3)  # T=1365
    matrix = np.arange(15, dtype=np.float64).reshape(3, 5)
    repeated = repeat_query_rows(matrix, layout)
    assert repeated.shape == (3 * layout.tile_width, 5)
    # First T rows should be query 0
    np.testing.assert_array_equal(repeated[0], matrix[0])
    np.testing.assert_array_equal(repeated[layout.tile_width], matrix[1])
    np.testing.assert_array_equal(repeated[2 * layout.tile_width], matrix[2])


def test_repeat_query_rows_wrong_rows():
    layout = make_slot_tile_layout(3)
    with pytest.raises(ValueError, match="batch_size"):
        repeat_query_rows(np.zeros((4, 5)), layout)


def test_expand_plain_tile():
    layout = make_slot_tile_layout(3)  # T=1365
    values = np.arange(layout.tile_width, dtype=np.float64)
    expanded = expand_plain_tile(values, layout)
    assert expanded.shape == (layout.active_slots,)
    # Check first and second group
    np.testing.assert_array_equal(expanded[: layout.tile_width], values)
    np.testing.assert_array_equal(
        expanded[layout.tile_width : 2 * layout.tile_width], values
    )


def test_expand_plain_tile_wrong_length():
    layout = make_slot_tile_layout(3)
    with pytest.raises(ValueError, match="tile_width"):
        expand_plain_tile(np.zeros(layout.tile_width - 1), layout)


def test_unpack_score_tile():
    layout = make_slot_tile_layout(5)  # T=819
    values = np.arange(layout.active_slots, dtype=np.float64)
    valid_width = 17
    unpacked = unpack_score_tile(values, layout, valid_width)
    assert unpacked.shape == (5, valid_width)
    # q-major: row q starts at q*T
    expected = values.reshape(5, layout.tile_width)[:, :valid_width]
    np.testing.assert_array_equal(unpacked, expected)


def test_unpack_score_tile_full_width():
    layout = make_slot_tile_layout(5)
    values = np.arange(layout.active_slots, dtype=np.float64)
    unpacked = unpack_score_tile(values, layout, layout.tile_width)
    assert unpacked.shape == (5, layout.tile_width)
    np.testing.assert_array_equal(unpacked, values.reshape(5, layout.tile_width))


def test_pack_centroid_tile():
    layout = make_slot_tile_layout(200)  # T=20
    centroids = np.arange(50 * 200, dtype=np.float64).reshape(50, 200)
    # Use valid_width that fits inside remaining centroids from start.
    tile = pack_centroid_tile(centroids, layout, start=40, valid_width=10)
    assert tile.shape == (20, 200)
    np.testing.assert_array_equal(tile[:10], centroids[40:50])
    np.testing.assert_array_equal(tile[10:], 0.0)


def test_pack_candidate_tile():
    layout = make_slot_tile_layout(200)
    cluster_matrix = np.arange(50 * 483 * 50, dtype=np.float64).reshape(50, 483, 50)
    tile = pack_candidate_tile(cluster_matrix, layout, start=480, valid_width=3)
    assert tile.shape == (50, 20, 50)
    np.testing.assert_array_equal(tile[:, :3, :], cluster_matrix[:, 480:483, :])
    np.testing.assert_array_equal(tile[:, 3:, :], 0.0)


def test_pack_selector_tile():
    layout = make_slot_tile_layout(4)  # T=1024
    selector = np.arange(20, dtype=np.float64).reshape(4, 5)
    packed = pack_selector_tile(selector, layout)
    assert packed.shape == (4 * layout.tile_width, 5)
    np.testing.assert_array_equal(packed[0], selector[0])
    np.testing.assert_array_equal(packed[layout.tile_width], selector[1])


# ---------------------------------------------------------------------------
# Production-dimension structural tests
# ---------------------------------------------------------------------------


def test_current_params_round1_tiles():
    layout = make_slot_tile_layout(200)
    assert tile_count(50, layout) == 3
    slices = list(iter_tile_slices(50, layout))
    assert [v for _, _, v in slices] == [20, 20, 10]


def test_current_params_round2_tiles():
    layout = make_slot_tile_layout(200)
    assert tile_count(483, layout) == 25
    slices = list(iter_tile_slices(483, layout))
    assert slices[-1] == (24, 480, 3)
