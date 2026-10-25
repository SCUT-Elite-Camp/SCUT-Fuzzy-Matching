"""Tests for Party B batch online responder logic in party_b/."""

import numpy as np
import pytest
import tenseal as ts

from ckks.batching import decrypt_score_batch, encrypt_feature_batch
from ckks.context import create_ckks_context
from party_b.online_responder import (
    column_wise_batch_matching,
    compare_batch_to_centroids,
)
from protocol.types import (
    BatchFirstRoundRequest,
    BatchSecondRoundRequest,
)


@pytest.fixture(scope="module")
def ckks_ctx():
    """Module-level CKKS context with secret key."""
    return create_ckks_context()


@pytest.fixture(scope="module")
def pub_ctx_bytes(ckks_ctx):
    """Public context bytes without secret key."""
    pub = ckks_ctx.copy()
    pub.make_context_public()
    return pub.serialize()


# ---------------------------------------------------------------------------
# Core Unit Tests for party_b/online_responder.py Batch APIs
# ---------------------------------------------------------------------------


def test_compare_batch_to_centroids(ckks_ctx, pub_ctx_bytes):
    """Test compare_batch_to_centroids correctness and shapes."""
    m, k, d = 3, 4, 200
    np.random.seed(100)
    query_200 = np.random.uniform(-1.0, 1.0, size=(m, d))
    centroids = np.random.uniform(-1.0, 1.0, size=(k, d))

    enc_query_200 = encrypt_feature_batch(query_200, ckks_ctx)
    req1 = BatchFirstRoundRequest(
        public_context_bytes=pub_ctx_bytes,
        encrypted_query_200=enc_query_200,
        batch_size=m,
    )

    # Native object output
    scores_ct = compare_batch_to_centroids(req1, centroids, serialize_output=False)
    assert len(scores_ct) == k
    for s in scores_ct:
        assert s.size() == m

    # Decrypt and compare with plaintext matmul Q @ C.T
    decrypted_scores = decrypt_score_batch(
        scores_ct, ckks_ctx, output_count=k, batch_size=m
    )
    expected_scores = query_200 @ centroids.T
    np.testing.assert_allclose(decrypted_scores, expected_scores, atol=1e-4)

    # Serialized bytes output
    bytes_scores = compare_batch_to_centroids(req1, centroids, serialize_output=True)
    assert len(bytes_scores) == k
    assert all(isinstance(b, bytes) for b in bytes_scores)

    decrypted_bytes_scores = decrypt_score_batch(
        bytes_scores, ckks_ctx, output_count=k, batch_size=m
    )
    np.testing.assert_allclose(decrypted_bytes_scores, expected_scores, atol=1e-4)


def test_column_wise_batch_matching_signs(ckks_ctx, pub_ctx_bytes):
    """Test column_wise_batch_matching mask scaling and sign correctness per query."""
    m, k, max_size, d = 3, 4, 5, 50
    np.random.seed(200)

    query_50 = np.random.uniform(-1.0, 1.0, size=(m, d))
    cluster_matrix = np.random.uniform(-1.0, 1.0, size=(k, max_size, d))

    # Selected cluster per query: Q0 -> c2, Q1 -> c0, Q2 -> c3
    selected_clusters = np.array([2, 0, 3])
    selector_matrix = np.zeros((m, k), dtype=np.float64)
    for q in range(m):
        selector_matrix[q, selected_clusters[q]] = 1.0

    enc_query_50 = encrypt_feature_batch(query_50, ckks_ctx)
    enc_selectors = encrypt_feature_batch(selector_matrix, ckks_ctx)

    req2 = BatchSecondRoundRequest(
        encrypted_query_50=enc_query_50,
        encrypted_selectors=enc_selectors,
        batch_size=m,
    )

    tau = 0.5
    generator = column_wise_batch_matching(
        cluster_matrix, req2, pub_ctx_bytes, tau=tau, serialize_output=False
    )

    col_scores_list = list(generator)
    assert len(col_scores_list) == max_size

    for j in range(max_size):
        score_ct_j = col_scores_list[j]
        assert score_ct_j.size() == m

        # Decrypt column j score ciphertext -> (m, 1) -> (m,)
        dec_col_j = decrypt_score_batch(
            [score_ct_j], ckks_ctx, output_count=1, batch_size=m
        ).reshape(-1)

        for q in range(m):
            cq = selected_clusters[q]
            raw_dot = np.dot(query_50[q], cluster_matrix[cq, j])
            diff = raw_dot - tau

            # If diff away from zero (|diff| > 1e-3), positive mask r_j[q] > 0 MUST preserve sign
            if abs(diff) > 1e-3:
                assert np.sign(dec_col_j[q]) == np.sign(diff)


def test_column_wise_batch_matching_serialized(ckks_ctx, pub_ctx_bytes):
    """Test column_wise_batch_matching with serialize_output=True."""
    m, k, max_size, d = 2, 3, 2, 50
    np.random.seed(300)

    query_50 = np.random.uniform(-1.0, 1.0, size=(m, d))
    cluster_matrix = np.random.uniform(-1.0, 1.0, size=(k, max_size, d))
    selector_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)

    req2 = BatchSecondRoundRequest(
        encrypted_query_50=encrypt_feature_batch(query_50, ckks_ctx),
        encrypted_selectors=encrypt_feature_batch(selector_matrix, ckks_ctx),
        batch_size=m,
    )

    generator = column_wise_batch_matching(
        cluster_matrix, req2, pub_ctx_bytes, serialize_output=True
    )
    col_bytes_list = list(generator)
    assert len(col_bytes_list) == max_size
    assert all(isinstance(b, bytes) for b in col_bytes_list)


# ---------------------------------------------------------------------------
# Validation & Boundary Error Tests
# ---------------------------------------------------------------------------


def test_illegal_centroids_shape(pub_ctx_bytes):
    """Test dimension mismatch in centroids matrix."""
    bad_centroids_1d = np.ones(200)
    req1 = BatchFirstRoundRequest(
        public_context_bytes=pub_ctx_bytes,
        encrypted_query_200=[],
        batch_size=2,
    )

    with pytest.raises(ValueError, match="centroids must be 2-D"):
        compare_batch_to_centroids(req1, bad_centroids_1d)

    bad_centroids_cols = np.ones((4, 100))
    with pytest.raises(ValueError, match="second dimension must be 200"):
        compare_batch_to_centroids(req1, bad_centroids_cols)


def test_illegal_cluster_matrix_shape(pub_ctx_bytes):
    """Test dimension mismatch in cluster matrix."""
    bad_matrix_2d = np.ones((4, 50))
    req2 = BatchSecondRoundRequest(
        encrypted_query_50=[], encrypted_selectors=[], batch_size=2
    )

    with pytest.raises(ValueError, match="cluster_matrix must be 3-D"):
        list(column_wise_batch_matching(bad_matrix_2d, req2, pub_ctx_bytes))

    bad_matrix_dim50 = np.ones((4, 5, 20))
    with pytest.raises(ValueError, match="last dimension must be 50"):
        list(column_wise_batch_matching(bad_matrix_dim50, req2, pub_ctx_bytes))


def test_nan_in_b_matrices(pub_ctx_bytes):
    """Test NaN / Inf detection in Party B matrices."""
    nan_centroids = np.ones((4, 200))
    nan_centroids[0, 0] = np.nan
    req1 = BatchFirstRoundRequest(
        public_context_bytes=pub_ctx_bytes,
        encrypted_query_200=[],
        batch_size=2,
    )

    with pytest.raises(ValueError, match="NaN or Inf"):
        compare_batch_to_centroids(req1, nan_centroids)
