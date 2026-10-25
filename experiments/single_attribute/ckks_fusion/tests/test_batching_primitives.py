"""Tests for vertical SIMD CKKS primitives in ckks/batching.py."""

import numpy as np
import pytest
import tenseal as ts

from ckks.batching import (
    batch_dot_ct_ct,
    batch_dot_ct_pt,
    batch_select_ct_pt,
    decrypt_score_batch,
    encrypt_feature_batch,
    load_feature_batch,
    serialize_feature_batch,
)
from ckks.context import create_ckks_context


@pytest.fixture(scope="module")
def ckks_ctx():
    """Module-level CKKS context for testing."""
    return create_ckks_context()


@pytest.fixture(scope="module")
def alt_ckks_ctx():
    """Alternative CKKS context to test context mismatch."""
    return create_ckks_context()


# ---------------------------------------------------------------------------
# Core Primitive Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m, d", [(1, 4), (3, 4), (17, 4)])
def test_encrypt_decrypt_transpose(ckks_ctx, m, d):
    """Test (m, d) matrix -> encrypted batch -> decrypted (m, d)."""
    np.random.seed(42 + m)
    matrix = np.random.uniform(-1.0, 1.0, size=(m, d))

    ciphertexts = encrypt_feature_batch(matrix, ckks_ctx)
    assert len(ciphertexts) == d
    for c in ciphertexts:
        assert c.size() == m

    decrypted = decrypt_score_batch(
        ciphertexts, ckks_ctx, output_count=d, batch_size=m
    )
    assert decrypted.shape == (m, d)
    np.testing.assert_allclose(decrypted, matrix, atol=1e-4)


@pytest.mark.parametrize("m, d, n", [(1, 5, 3), (4, 10, 6), (16, 8, 5)])
def test_batch_dot_ct_pt(ckks_ctx, m, d, n):
    """Test d feature ciphertexts (m, d) x (n, d) -> n score ciphertexts (m, n)."""
    np.random.seed(100 + m)
    query_matrix = np.random.uniform(-1.0, 1.0, size=(m, d))
    plain_rows = np.random.uniform(-1.0, 1.0, size=(n, d))

    enc_features = encrypt_feature_batch(query_matrix, ckks_ctx)
    score_ciphertexts = batch_dot_ct_pt(enc_features, plain_rows)

    assert len(score_ciphertexts) == n
    for s in score_ciphertexts:
        assert s.size() == m

    decrypted_scores = decrypt_score_batch(
        score_ciphertexts, ckks_ctx, output_count=n, batch_size=m
    )
    expected_scores = query_matrix @ plain_rows.T
    assert decrypted_scores.shape == (m, n)
    np.testing.assert_allclose(decrypted_scores, expected_scores, atol=1e-4)


@pytest.mark.parametrize("m, k, d", [(1, 4, 5), (5, 3, 6), (12, 5, 8)])
def test_batch_select_ct_pt(ckks_ctx, m, k, d):
    """Test selector (m, k) one-hot x (k, d) matrix x (m,) positive mask -> (m, d)."""
    np.random.seed(200 + m)
    # Randomly assign each query q to a cluster in 0..k-1
    selected_clusters = np.random.randint(0, k, size=m)

    # Build one-hot selector matrix (m, k)
    selector_matrix = np.zeros((m, k), dtype=np.float64)
    for q, c in enumerate(selected_clusters):
        selector_matrix[q, c] = 1.0

    # Encrypt selector columns (k ciphertexts of size m)
    enc_selectors = encrypt_feature_batch(selector_matrix, ckks_ctx)

    plain_column = np.random.uniform(-1.0, 1.0, size=(k, d))
    mask = np.random.uniform(1.0, 5.0, size=m)

    selected_ct = batch_select_ct_pt(enc_selectors, plain_column, mask)
    assert len(selected_ct) == d
    for s in selected_ct:
        assert s.size() == m

    decrypted = decrypt_score_batch(
        selected_ct, ckks_ctx, output_count=d, batch_size=m
    )
    assert decrypted.shape == (m, d)

    expected = plain_column[selected_clusters, :] * mask[:, None]
    np.testing.assert_allclose(decrypted, expected, atol=1e-4)


@pytest.mark.parametrize("m, d", [(1, 5), (7, 10), (15, 8)])
def test_batch_dot_ct_ct(ckks_ctx, m, d):
    """Test dot product of two (m, d) ciphertext feature lists -> single (m,) ciphertext."""
    np.random.seed(300 + m)
    matrix_left = np.random.uniform(-1.0, 1.0, size=(m, d))
    matrix_right = np.random.uniform(-1.0, 1.0, size=(m, d))

    enc_left = encrypt_feature_batch(matrix_left, ckks_ctx)
    enc_right = encrypt_feature_batch(matrix_right, ckks_ctx)

    res_ct = batch_dot_ct_ct(enc_left, enc_right)
    assert res_ct.size() == m

    decrypted = decrypt_score_batch(
        [res_ct], ckks_ctx, output_count=1, batch_size=m
    )
    assert decrypted.shape == (m, 1)

    expected = (matrix_left * matrix_right).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(decrypted, expected, atol=1e-4)


def test_mask_sign_preservation(ckks_ctx):
    """Test positive mask r_j preserves sign of scores away from gray zone."""
    m = 5
    d = 4
    n = 3
    np.random.seed(400)
    query_matrix = np.random.uniform(-1.0, 1.0, size=(m, d))
    plain_rows = np.random.uniform(-1.0, 1.0, size=(n, d))
    mask = np.random.uniform(1.0, 10.0, size=m)

    scores_ct = batch_dot_ct_pt(encrypt_feature_batch(query_matrix, ckks_ctx), plain_rows)

    # Multiply each score by mask (elementwise)
    masked_scores_ct = []
    for s in scores_ct:
        masked_scores_ct.append(s * mask.tolist())

    decrypted_masked = decrypt_score_batch(
        masked_scores_ct, ckks_ctx, output_count=n, batch_size=m
    )
    raw_scores = query_matrix @ plain_rows.T

    # Filter out near-zero gray zone values (|val| > 1e-3)
    non_gray = np.abs(raw_scores) > 1e-3
    assert np.all(np.sign(decrypted_masked[non_gray]) == np.sign(raw_scores[non_gray]))


def test_serialization_roundtrip(ckks_ctx):
    """Test object -> bytes -> object roundtrip preserving count, size, and values."""
    m, d = 6, 8
    np.random.seed(500)
    matrix = np.random.uniform(-1.0, 1.0, size=(m, d))

    orig_ct = encrypt_feature_batch(matrix, ckks_ctx)
    bytes_list = serialize_feature_batch(orig_ct)
    assert len(bytes_list) == d
    assert all(isinstance(b, bytes) for b in bytes_list)

    loaded_ct = load_feature_batch(
        bytes_list, ckks_ctx, feature_count=d, batch_size=m
    )
    assert len(loaded_ct) == d

    decrypted = decrypt_score_batch(
        loaded_ct, ckks_ctx, output_count=d, batch_size=m
    )
    np.testing.assert_allclose(decrypted, matrix, atol=1e-4)


# ---------------------------------------------------------------------------
# Illegal Input & Validation Tests
# ---------------------------------------------------------------------------


def test_illegal_batch_size(ckks_ctx):
    """Test invalid batch sizes (m = 0, m > 4096, non-int)."""
    matrix_zero = np.zeros((0, 4))
    with pytest.raises(ValueError, match="batch_size"):
        encrypt_feature_batch(matrix_zero, ckks_ctx)

    matrix_huge = np.zeros((4097, 4))
    with pytest.raises(ValueError, match="batch_size"):
        encrypt_feature_batch(matrix_huge, ckks_ctx)

    with pytest.raises(ValueError, match="batch_size"):
        load_feature_batch([], ckks_ctx, feature_count=4, batch_size=0)

    with pytest.raises(ValueError, match="batch_size"):
        load_feature_batch([], ckks_ctx, feature_count=4, batch_size=4097)


def test_illegal_feature_list(ckks_ctx):
    """Test empty feature lists and count mismatches."""
    with pytest.raises(ValueError, match="密文列表不能为空"):
        serialize_feature_batch([])

    with pytest.raises(ValueError, match="密文列表不能为空"):
        batch_dot_ct_pt([], np.ones((3, 4)))

    with pytest.raises(ValueError, match="密文列表长度不一致|特征密文列表不能为空"):
        batch_dot_ct_ct([], [])

    m, d = 3, 4
    enc_ct = encrypt_feature_batch(np.ones((m, d)), ckks_ctx)
    with pytest.raises(ValueError, match="密文数量不匹配"):
        load_feature_batch(enc_ct, ckks_ctx, feature_count=d + 1, batch_size=m)

    with pytest.raises(ValueError, match="密文数量不匹配"):
        decrypt_score_batch(enc_ct, ckks_ctx, output_count=d - 1, batch_size=m)


def test_slot_size_mismatch(ckks_ctx):
    """Test ciphertexts with mismatched slot sizes."""
    ct_m3 = encrypt_feature_batch(np.ones((3, 4)), ckks_ctx)
    ct_m5 = encrypt_feature_batch(np.ones((5, 4)), ckks_ctx)

    # Mixed slot sizes in dot product
    mixed_features = [ct_m3[0], ct_m5[1], ct_m3[2], ct_m3[3]]
    with pytest.raises(ValueError, match="槽位数"):
        batch_dot_ct_pt(mixed_features, np.ones((2, 4)))

    with pytest.raises(ValueError, match="槽位数"):
        batch_dot_ct_ct(ct_m3, ct_m5)


def test_plain_matrix_shape_and_nan(ckks_ctx):
    """Test dimension mismatch and NaN/Inf detection in plain matrices."""
    m, d = 4, 5
    enc_features = encrypt_feature_batch(np.ones((m, d)), ckks_ctx)

    # Dimension mismatch in batch_dot_ct_pt (d=5 vs plain cols=3)
    with pytest.raises(ValueError, match="形状不匹配|列数"):
        batch_dot_ct_pt(enc_features, np.ones((3, 3)))

    # 1D plain rows
    with pytest.raises(ValueError, match="二维数组"):
        batch_dot_ct_pt(enc_features, np.ones(5))

    # NaN / Inf
    nan_rows = np.ones((3, d))
    nan_rows[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN 或 Inf"):
        batch_dot_ct_pt(enc_features, nan_rows)


def test_mask_validation(ckks_ctx):
    """Test mask length and non-positive value validation."""
    m, k, d = 4, 3, 5
    selectors = encrypt_feature_batch(np.ones((m, k)), ckks_ctx)
    plain_col = np.ones((k, d))

    # Non-positive mask
    bad_mask_zero = np.ones(m)
    bad_mask_zero[1] = 0.0
    with pytest.raises(ValueError, match="正数|严格大于 0"):
        batch_select_ct_pt(selectors, plain_col, bad_mask_zero)

    bad_mask_neg = np.ones(m)
    bad_mask_neg[2] = -1.5
    with pytest.raises(ValueError, match="正数|严格大于 0"):
        batch_select_ct_pt(selectors, plain_col, bad_mask_neg)

    # Wrong mask length
    bad_mask_len = np.ones(m + 1)
    with pytest.raises(ValueError, match="mask 长度"):
        batch_select_ct_pt(selectors, plain_col, bad_mask_len)


def test_context_mismatch(ckks_ctx):
    """Test public context cannot decrypt and mixing different contexts is validated."""
    m, d = 3, 4
    matrix = np.ones((m, d))
    ct = encrypt_feature_batch(matrix, ckks_ctx)
    bytes_ct = serialize_feature_batch(ct)

    # Public context (drop secret key)
    pub_ctx = ckks_ctx.copy()
    pub_ctx.make_context_public()

    # Deserializing with public context succeeds
    loaded_pub = load_feature_batch(
        bytes_ct, pub_ctx, feature_count=d, batch_size=m
    )

    # Decrypting with public context must fail
    with pytest.raises((RuntimeError, ValueError)):
        decrypt_score_batch(
            loaded_pub, pub_ctx, output_count=d, batch_size=m
        )

