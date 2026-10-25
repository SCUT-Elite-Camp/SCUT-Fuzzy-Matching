"""Numerical correctness checks against real modules."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ckks.context import create_ckks_context
from ckks.keys import encrypt
from ckks.operations import add_plain, dot_ct_ct, dot_ct_pt
from minhash.encoder import batch_encode
from party_b.offline_prep import fit_scaler
from preprocessing.normalizer import l2_normalize


def test_minhash_consistency_for_same_name():
    left = batch_encode(["John Smith"], 200)
    right = batch_encode(["john smith"], 200)

    np.testing.assert_array_equal(left, right)


def test_party_a_scaler_formula_matches_b_scaler_parameters():
    normalized = l2_normalize(batch_encode(["john smith", "mary jones"], 200))
    _, standardized, mean, scale = fit_scaler(normalized)
    manual = (normalized[0] - mean) / scale

    np.testing.assert_allclose(manual, standardized[0])


def test_ckks_ct_pt_dot_matches_plain_dot_after_decrypt():
    ctx = create_ckks_context()
    ctx.generate_relin_keys()
    plain_left = np.zeros(50, dtype=np.float64)
    plain_left[0] = 1.0
    plain_right = np.zeros(50, dtype=np.float64)
    plain_right[0] = 0.75

    encrypted_left = encrypt(plain_left, ctx)
    encrypted_score = dot_ct_pt(encrypted_left, plain_right)

    assert abs(encrypted_score.decrypt()[0] - 0.75) < 1e-4


def test_ckks_ct_ct_dot_matches_plain_dot_after_decrypt():
    ctx = create_ckks_context()
    ctx.generate_relin_keys()
    plain_left = np.zeros(50, dtype=np.float64)
    plain_left[0] = 1.0
    plain_right = np.zeros(50, dtype=np.float64)
    plain_right[0] = 0.5

    encrypted_left = encrypt(plain_left, ctx)
    encrypted_right = encrypt(plain_right, ctx)
    encrypted_score = dot_ct_ct(encrypted_left, encrypted_right)

    assert abs(encrypted_score.decrypt()[0] - 0.5) < 1e-4


def test_threshold_sign_preserved_after_positive_mask():
    ctx = create_ckks_context()
    encrypted_score = encrypt(np.array([0.95], dtype=np.float64), ctx)
    shifted = add_plain(encrypted_score, -0.9)
    masked = shifted * 2.0

    assert masked.decrypt()[0] > 0


if __name__ == "__main__":
    test_minhash_consistency_for_same_name()
    test_party_a_scaler_formula_matches_b_scaler_parameters()
    test_ckks_ct_pt_dot_matches_plain_dot_after_decrypt()
    test_ckks_ct_ct_dot_matches_plain_dot_after_decrypt()
    test_threshold_sign_preserved_after_positive_mask()
    print("real correctness tests passed")
