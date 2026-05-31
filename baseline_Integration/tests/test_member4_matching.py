import numpy as np
import tenseal as ts

from ckks.operations import add_plain, dot_ct_ct, matmul_ct_pt
from config.params import COEFF_MOD_BIT_SIZES, POLY_MODULUS_DEGREE, SCALE
from config.params import DECRYPT_EPS
from party_b.online_responder import column_wise_matching
from protocol.types import SecondRoundRequest


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


def _build_fixture():
    ctx = _create_secret_context()
    query = np.zeros(50, dtype=np.float64)
    query[0] = 1.0

    cluster_matrix = np.zeros((2, 3, 50), dtype=np.float64)
    cluster_matrix[0, 0, 1] = 1.0
    cluster_matrix[1, 0, 0] = 1.0
    cluster_matrix[0, 1, 2] = 1.0
    cluster_matrix[1, 1, 3] = 1.0
    cluster_matrix[0, 2, 4] = 1.0
    cluster_matrix[1, 2, 5] = 1.0

    selector = np.array([0.0, 1.0], dtype=np.float64)
    query_ct = ts.ckks_vector(ctx, query.tolist())
    selector_ct = ts.ckks_vector(ctx, selector.tolist())
    request = SecondRoundRequest(
        encrypted_query_50=query_ct,
        encrypted_selector=selector_ct,
    )
    return ctx, query, cluster_matrix, request


def test_matmul_ct_pt_selects_target_cluster_row():
    ctx, _, cluster_matrix, request = _build_fixture()

    selected = matmul_ct_pt(request.encrypted_selector, cluster_matrix[:, 0, :])
    values = np.array(selected.decrypt(), dtype=np.float64)

    assert values.shape == (50,)
    assert np.allclose(values, cluster_matrix[1, 0, :], atol=1e-4)


def test_column_wise_matching_returns_one_score_per_column_and_positive_match():
    ctx, _, cluster_matrix, request = _build_fixture()

    scores = list(column_wise_matching(cluster_matrix, request, ctx, tau=0.5))

    assert len(scores) == cluster_matrix.shape[1]

    first_plain = float(scores[0].decrypt()[0])
    second_plain = float(scores[1].decrypt()[0])

    assert first_plain > DECRYPT_EPS
    assert second_plain <= DECRYPT_EPS

    selected_name = matmul_ct_pt(request.encrypted_selector, cluster_matrix[:, 0, :])
    cos_score = dot_ct_ct(request.encrypted_query_50, selected_name)
    shifted = add_plain(cos_score, -0.5)
    assert float(shifted.decrypt()[0]) > DECRYPT_EPS
