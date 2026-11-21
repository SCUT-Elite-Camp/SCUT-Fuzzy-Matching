"""Tests for Party A batch preparation and online selection logic in party_a/."""

import numpy as np
import pytest
import tenseal as ts

from ckks.batching import encrypt_feature_batch
from ckks.context import create_ckks_context
from party_a.local_prep import (
    encode_query_batch,
    encode_query_vectors,
    prepare_encrypted_query_batch,
)
from party_a.online_querier import (
    check_encrypted_score_batch,
    check_encrypted_score_batch_debug,
    choose_clusters_and_build_batch_request,
)
from protocol.types import (
    BatchFirstRoundRequest,
    BatchPartyALocalState,
    BatchSecondRoundRequest,
)


@pytest.fixture(scope="module")
def mock_scaler():
    """Create mock scaler mean and scale for testing."""
    np.random.seed(42)
    mean = np.random.uniform(-1.0, 1.0, size=200)
    scale = np.random.uniform(0.5, 2.0, size=200)
    return mean, scale


@pytest.fixture(scope="module")
def ckks_ctx():
    """Module level CKKS context."""
    return create_ckks_context()


# ---------------------------------------------------------------------------
# Unit Tests for party_a/local_prep.py Batch APIs
# ---------------------------------------------------------------------------


def test_encode_query_batch_single_vs_multi(mock_scaler):
    """Test encode_query_batch with m=1 and m=3 against single query encoder."""
    mean, scale = mock_scaler
    name = "JOHN SMITH"

    # Single query encoder (1D)
    single_200, single_50 = encode_query_vectors(name, mean, scale)

    # Batch encoder m=1 (2D)
    b1_200, b1_50 = encode_query_batch([name], mean, scale)
    assert b1_200.shape == (1, 200)
    assert b1_50.shape == (1, 50)
    np.testing.assert_allclose(b1_200[0], single_200, atol=1e-6)
    np.testing.assert_allclose(b1_50[0], single_50, atol=1e-6)

    # Batch encoder m=3
    names = ["JOHN SMITH", "ALICE BROWN", "BOB WILSON"]
    b3_200, b3_50 = encode_query_batch(names, mean, scale)
    assert b3_200.shape == (3, 200)
    assert b3_50.shape == (3, 50)
    np.testing.assert_allclose(b3_200[0], single_200, atol=1e-6)


def test_prepare_encrypted_query_batch(mock_scaler, ckks_ctx):
    """Test prepare_encrypted_query_batch structure and ciphertext dimensions."""
    mean, scale = mock_scaler
    names = ["JOHN SMITH", "ALICE BROWN"]

    req, state = prepare_encrypted_query_batch(
        names, mean, scale, context=ckks_ctx
    )

    assert isinstance(req, BatchFirstRoundRequest)
    assert isinstance(state, BatchPartyALocalState)

    assert req.batch_size == 2
    assert state.batch_size == 2

    assert len(req.encrypted_query_200) == 200
    for c in req.encrypted_query_200:
        assert c.size() == 2

    assert len(state.encrypted_query_50) == 50
    for c in state.encrypted_query_50:
        assert c.size() == 2

    assert isinstance(req.public_context_bytes, bytes)


# ---------------------------------------------------------------------------
# Unit Tests for party_a/online_querier.py Batch APIs
# ---------------------------------------------------------------------------


def test_choose_clusters_and_build_batch_request(mock_scaler, ckks_ctx):
    """Test cluster argmax selection per query and one-hot selector generation."""
    mean, scale = mock_scaler
    names = ["QUERY_0", "QUERY_1", "QUERY_2"]
    m = len(names)
    k = 4

    _, state = prepare_encrypted_query_batch(
        names, mean, scale, context=ckks_ctx
    )

    # Create mock sim scores matrix (3, 4)
    # Query 0 argmax = 2 (score 0.9)
    # Query 1 argmax = 0 (score 0.8)
    # Query 2 argmax = 3 (score 0.95)
    sim_matrix = np.array(
        [
            [0.1, 0.3, 0.9, 0.2],
            [0.8, 0.2, 0.1, 0.4],
            [0.3, 0.5, 0.1, 0.95],
        ],
        dtype=np.float64,
    )

    enc_sim_scores = encrypt_feature_batch(sim_matrix, ckks_ctx)

    req2, debug = choose_clusters_and_build_batch_request(
        enc_sim_scores, state, k=k
    )

    assert isinstance(req2, BatchSecondRoundRequest)
    assert req2.batch_size == 3
    assert len(req2.encrypted_selectors) == k
    for s in req2.encrypted_selectors:
        assert s.size() == 3

    # Check debug argmax
    expected_clusters = np.array([2, 0, 3])
    np.testing.assert_array_equal(debug.selected_clusters, expected_clusters)


def test_check_encrypted_score_batch_debug_early_stop(ckks_ctx):
    """Test early stop triggers ONLY when all queries in the batch have hit."""
    m = 3
    k_cols = 4

    # Column scores (4 columns, each of size 3)
    # Col 0: Q0 hits (0.5 > 1e-6), Q1 & Q2 miss (-0.1)
    # Col 1: Q1 & Q2 hit (0.8, 0.9) -> ALL 3 queries now hit!
    # Col 2 & 3: Additional columns
    col0 = np.array([0.5, -0.1, -0.1])
    col1 = np.array([0.1, 0.8, 0.9])
    col2 = np.array([0.2, 0.3, 0.4])
    col3 = np.array([0.1, 0.1, 0.1])

    cols_matrix = np.stack([col0, col1, col2, col3], axis=1)  # (3, 4)
    # Encrypt each column as a ciphertext of size 3 (4 ciphertexts total)
    enc_cols = [
        encrypt_feature_batch(cols_matrix[:, c : c + 1], ckks_ctx)[0]
        for c in range(k_cols)
    ]

    res, debug = check_encrypted_score_batch_debug(
        enc_cols, ckks_ctx, batch_size=m, early_stop=True
    )

    assert np.all(res.catches)
    assert debug.checked_columns == 2  # Stopped after col 1 because all hit
    np.testing.assert_array_equal(debug.first_positive_columns, [0, 1, 1])


def test_check_encrypted_score_batch_debug_partial_hit(ckks_ctx):
    """Test when some query never hits, early stop cannot trigger and checks all columns."""
    m = 3
    k_cols = 4

    # Q0 hits at col 0, Q1 hits at col 1, Q2 NEVER hits (always negative)
    col0 = np.array([0.5, -0.1, -0.2])
    col1 = np.array([0.1, 0.8, -0.5])
    col2 = np.array([-0.1, -0.2, -0.3])
    col3 = np.array([-0.1, -0.2, -0.3])

    cols_matrix = np.stack([col0, col1, col2, col3], axis=1)
    enc_cols = [
        encrypt_feature_batch(cols_matrix[:, c : c + 1], ckks_ctx)[0]
        for c in range(k_cols)
    ]

    res, debug = check_encrypted_score_batch_debug(
        enc_cols, ckks_ctx, batch_size=m, early_stop=True
    )

    np.testing.assert_array_equal(res.catches, [True, True, False])
    assert debug.checked_columns == 4  # Checked ALL 4 columns because Q2 didn't hit
    np.testing.assert_array_equal(debug.first_positive_columns, [0, 1, -1])

    # Check production wrapper check_encrypted_score_batch
    prod_res = check_encrypted_score_batch(
        enc_cols, ckks_ctx, batch_size=m, early_stop=True
    )
    np.testing.assert_array_equal(prod_res.catches, [True, True, False])


def test_check_encrypted_score_batch_no_early_stop(ckks_ctx):
    """Test early_stop=False forces checking all columns even when all hit early."""
    m = 2
    k_cols = 3

    # Both hit at col 0
    col0 = np.array([0.5, 0.6])
    col1 = np.array([0.1, 0.2])
    col2 = np.array([0.3, 0.4])

    cols_matrix = np.stack([col0, col1, col2], axis=1)
    enc_cols = [
        encrypt_feature_batch(cols_matrix[:, c : c + 1], ckks_ctx)[0]
        for c in range(k_cols)
    ]

    _, debug = check_encrypted_score_batch_debug(
        enc_cols, ckks_ctx, batch_size=m, early_stop=False
    )

    assert debug.checked_columns == 3


# ---------------------------------------------------------------------------
# Illegal Inputs & Validation Tests
# ---------------------------------------------------------------------------


def test_illegal_query_names(mock_scaler):
    """Test empty query_names or batch_size > 4096."""
    mean, scale = mock_scaler

    with pytest.raises(ValueError, match="query_names 列表不能为空"):
        encode_query_batch([], mean, scale)

    with pytest.raises(ValueError, match="query_names 列表不能为空"):
        prepare_encrypted_query_batch([], mean, scale)

    huge_names = [f"NAME_{i}" for i in range(4097)]
    with pytest.raises(ValueError, match="batch_size 4097 超过"):
        encode_query_batch(huge_names, mean, scale)


def test_illegal_scaler_params():
    """Test invalid shape or NaN/Inf in scaler mean/scale."""
    bad_mean = np.ones(100)
    good_scale = np.ones(200)

    with pytest.raises(ValueError, match="scaler 参数形状不匹配"):
        encode_query_batch(["JOHN"], bad_mean, good_scale)

    nan_mean = np.ones(200)
    nan_mean[0] = np.nan
    with pytest.raises(ValueError, match="NaN 或 Inf"):
        encode_query_batch(["JOHN"], nan_mean, good_scale)


def test_illegal_online_batch_params(ckks_ctx):
    """Test invalid k or batch_size in online querier."""
    state = BatchPartyALocalState(secret_context=ckks_ctx, encrypted_query_50=[], batch_size=2)

    with pytest.raises(ValueError, match="k 必须为正整数"):
        choose_clusters_and_build_batch_request([], state, k=0)

    with pytest.raises(ValueError, match="batch_size 必须位于"):
        check_encrypted_score_batch_debug([], ckks_ctx, batch_size=0)

    with pytest.raises(ValueError, match="batch_size 必须位于"):
        check_encrypted_score_batch_debug([], ckks_ctx, batch_size=4097)
