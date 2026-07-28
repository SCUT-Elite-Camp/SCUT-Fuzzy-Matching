"""Numerical precision and sign consistency tests for batching primitives across scales m=1, 10, 100, 1000."""

import numpy as np
import pytest

from ckks.batching import (
    batch_dot_ct_ct,
    batch_dot_ct_pt,
    batch_select_ct_pt,
    decrypt_score_batch,
    encrypt_feature_batch,
)
from ckks.context import create_ckks_context
from config.params import BATCH_DECRYPT_EPS
from party_a.online_querier import check_encrypted_score_batch_debug


@pytest.fixture(scope="module")
def ckks_ctx():
    """Module-level CKKS context."""
    return create_ckks_context()


# Safety upper bound for batch decryption error calibrated empirically
NUMERICAL_ERROR_LIMIT = 1e-4


@pytest.mark.parametrize("m", [1, 10, 100, 1000])
def test_batch_algebraic_max_abs_error(ckks_ctx, m):
    """Test max absolute error < 1e-4 for core algebraic operations across batch sizes m."""
    d, n, k = 50, 5, 4
    np.random.seed(42 + m)

    query_matrix = np.random.uniform(-1.0, 1.0, size=(m, d))
    plain_rows = np.random.uniform(-1.0, 1.0, size=(n, d))
    mask = np.random.uniform(1.0, 5.0, size=m)

    selected_clusters = np.random.randint(0, k, size=m)
    selector_matrix = np.zeros((m, k), dtype=np.float64)
    for q in range(m):
        selector_matrix[q, selected_clusters[q]] = 1.0

    plain_column = np.random.uniform(-1.0, 1.0, size=(k, d))

    # Encrypt
    enc_q = encrypt_feature_batch(query_matrix, ckks_ctx)
    enc_sel = encrypt_feature_batch(selector_matrix, ckks_ctx)

    # 1. Round 1 ct-pt dot product
    ct_scores = batch_dot_ct_pt(enc_q, plain_rows)
    dec_scores = decrypt_score_batch(
        ct_scores, ckks_ctx, output_count=n, batch_size=m
    )
    exp_scores = query_matrix @ plain_rows.T
    max_err_r1 = float(np.max(np.abs(dec_scores - exp_scores)))
    assert max_err_r1 < NUMERICAL_ERROR_LIMIT

    # 2. Round 2 select + ct-ct dot product
    sel_ct = batch_select_ct_pt(enc_sel, plain_column, mask)
    final_score_ct = batch_dot_ct_ct(enc_q, sel_ct)

    dec_final = decrypt_score_batch(
        [final_score_ct], ckks_ctx, output_count=1, batch_size=m
    ).reshape(-1)

    exp_selected_names = plain_column[selected_clusters, :]  # (m, d)
    exp_final = (query_matrix * (exp_selected_names * mask[:, None])).sum(axis=1)

    max_err_r2 = float(np.max(np.abs(dec_final - exp_final)))
    assert max_err_r2 < NUMERICAL_ERROR_LIMIT


@pytest.mark.parametrize("m", [1, 10, 100, 1000])
def test_batch_sign_consistency_outside_gray_zone(ckks_ctx, m):
    """Test 100% sign consistency when plaintext difference |diff| >= 5e-4."""
    d = 50
    k = 4
    np.random.seed(123 + m)

    query_matrix = np.random.uniform(-1.0, 1.0, size=(m, d))
    plain_column = np.random.uniform(-1.0, 1.0, size=(k, d))
    mask = np.random.uniform(1.0, 10.0, size=m)
    tau = 0.9

    selected_clusters = np.random.randint(0, k, size=m)
    selector_matrix = np.zeros((m, k), dtype=np.float64)
    for q in range(m):
        selector_matrix[q, selected_clusters[q]] = 1.0

    enc_q = encrypt_feature_batch(query_matrix, ckks_ctx)
    enc_sel = encrypt_feature_batch(selector_matrix, ckks_ctx)

    sel_ct = batch_select_ct_pt(enc_sel, plain_column, mask)
    raw_dot_ct = batch_dot_ct_ct(enc_q, sel_ct)
    final_score_ct = raw_dot_ct + (-tau * mask).tolist()

    dec_scores = decrypt_score_batch(
        [final_score_ct], ckks_ctx, output_count=1, batch_size=m
    ).reshape(-1)

    exp_selected_names = plain_column[selected_clusters, :]
    plain_dots = (query_matrix * exp_selected_names).sum(axis=1)
    plain_diff = plain_dots - tau

    # Filter out gray zone (|plain_diff| < 5e-4)
    clear_zone = np.abs(plain_diff) >= 5e-4
    if np.any(clear_zone):
        dec_signs = np.sign(dec_scores[clear_zone])
        plain_signs = np.sign(plain_diff[clear_zone])
        np.testing.assert_array_equal(dec_signs, plain_signs)


@pytest.mark.parametrize("m", [1, 1000, 4096])
def test_exact_threshold_stays_in_batch_gray_zone(ckks_ctx, m):
    """CKKS noise at exactly tau must not turn a non-positive score into a match."""
    d, k = 50, 1
    tau = 0.9
    rng = np.random.default_rng(20260727)

    query_matrix = np.zeros((m, d), dtype=np.float64)
    query_matrix[:, 0] = tau
    selector_matrix = np.ones((m, k), dtype=np.float64)
    plain_column = np.zeros((k, d), dtype=np.float64)
    plain_column[0, 0] = 1.0
    mask = rng.uniform(1.0, 10.0, size=m)

    enc_query = encrypt_feature_batch(query_matrix, ckks_ctx)
    enc_selector = encrypt_feature_batch(selector_matrix, ckks_ctx)
    selected = batch_select_ct_pt(enc_selector, plain_column, mask)
    score = batch_dot_ct_ct(enc_query, selected) + (-tau * mask).tolist()

    decrypted = decrypt_score_batch(
        [score], ckks_ctx, output_count=1, batch_size=m
    ).reshape(-1)
    assert float(np.max(np.abs(decrypted))) < BATCH_DECRYPT_EPS

    result, _ = check_encrypted_score_batch_debug(
        [score], ckks_ctx, batch_size=m
    )
    assert not np.any(result.catches)
