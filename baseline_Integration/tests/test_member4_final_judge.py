import numpy as np
import tenseal as ts

from config.params import COEFF_MOD_BIT_SIZES, POLY_MODULUS_DEGREE, SCALE
from party_a.online_querier import (
    check_encrypted_scores,
    check_encrypted_scores_debug,
)


def _create_secret_context() -> ts.Context:
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
    )
    ctx.global_scale = SCALE
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx


def _encrypt_scores(ctx: ts.Context, values: list[float]):
    return [ts.ckks_vector(ctx, [value]) for value in values]


def test_check_encrypted_scores_returns_true_when_any_score_exceeds_eps():
    ctx = _create_secret_context()
    encrypted_scores = _encrypt_scores(ctx, [-0.2, 0.3, -0.1])

    result = check_encrypted_scores(encrypted_scores, ctx, early_stop=True, eps=1e-6)

    assert result.catch is True


def test_check_encrypted_scores_returns_false_when_all_scores_are_non_positive():
    ctx = _create_secret_context()
    encrypted_scores = _encrypt_scores(ctx, [-0.2, 0.0, -0.1])

    result = check_encrypted_scores(encrypted_scores, ctx, early_stop=False, eps=1e-6)

    assert result.catch is False


def test_check_encrypted_scores_debug_respects_early_stop_and_reports_index():
    ctx = _create_secret_context()
    encrypted_scores = _encrypt_scores(ctx, [-0.2, 0.3, 0.4])

    result, debug = check_encrypted_scores_debug(
        encrypted_scores,
        ctx,
        early_stop=True,
        eps=1e-6,
    )

    assert result.catch is True
    assert debug.checked_columns == 2
    assert debug.first_positive_column == 1


def test_check_encrypted_scores_debug_consumes_all_columns_when_early_stop_disabled():
    ctx = _create_secret_context()
    encrypted_scores = _encrypt_scores(ctx, [-0.2, 0.3, 0.4])

    result, debug = check_encrypted_scores_debug(
        encrypted_scores,
        ctx,
        early_stop=False,
        eps=1e-6,
    )

    assert result.catch is True
    assert debug.checked_columns == 3
    assert debug.first_positive_column == 1
