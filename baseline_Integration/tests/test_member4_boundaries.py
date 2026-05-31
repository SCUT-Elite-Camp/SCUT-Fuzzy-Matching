import inspect

from config.params import RANDOM_MASK_MAX, RANDOM_MASK_MIN
from party_a.online_querier import (
    check_encrypted_scores,
    check_encrypted_scores_debug,
)
from party_b.online_responder import _sample_positive_mask, column_wise_matching
from protocol.types import MatchDebug, MatchResult


def test_column_wise_matching_signature_does_not_accept_secret_context():
    signature = inspect.signature(column_wise_matching)

    assert "secret_context" not in signature.parameters


def test_a_side_judgers_do_not_accept_cluster_matrix():
    for fn in (check_encrypted_scores, check_encrypted_scores_debug):
        signature = inspect.signature(fn)
        assert "cluster_matrix" not in signature.parameters


def test_match_result_only_exposes_catch():
    result = MatchResult(catch=True)

    assert result.catch is True
    assert not hasattr(result, "checked_columns")
    assert not hasattr(result, "first_positive_column")


def test_match_debug_tracks_column_metadata():
    debug = MatchDebug(checked_columns=3, first_positive_column=1)

    assert debug.checked_columns == 3
    assert debug.first_positive_column == 1


def test_mask_sampler_returns_bounded_positive_values():
    values = [_sample_positive_mask() for _ in range(10)]

    assert all(RANDOM_MASK_MIN <= value <= RANDOM_MASK_MAX for value in values)
    assert len(set(values)) > 1
