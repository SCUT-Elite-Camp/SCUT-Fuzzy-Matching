import importlib.util

import numpy as np
import pytest

from multi_attribute import (
    MatchRecord,
    MultiAttributeConfig,
    encode_record_vectors,
    plaintext_similarity,
)

HAS_TENSEAL = importlib.util.find_spec("tenseal") is not None
if HAS_TENSEAL:
    from multi_attribute.protocol import run_multi_attribute_protocol



def test_weighted_vector_shapes_and_self_similarity():
    cfg = MultiAttributeConfig()
    record = MatchRecord("John Smith", "2001-05-17")
    cluster, match = encode_record_vectors([record], cfg)
    assert cluster.shape == (1, cfg.cluster_dim)
    assert match.shape == (1, cfg.match_dim)
    assert np.isclose(np.dot(match[0], match[0]), 1.0, atol=1e-12)


def test_same_name_different_dob_is_penalized():
    cfg = MultiAttributeConfig(
        name_weight=0.70,
        dob_weight=0.30,
        similarity_threshold=0.80,
    )
    same = plaintext_similarity(
        MatchRecord("John Smith", "2001-05-17"),
        MatchRecord("John Smith", "2001/05/17"),
        cfg,
    )
    different_dob = plaintext_similarity(
        MatchRecord("John Smith", "2001-05-17"),
        MatchRecord("John Smith", "1990-01-01"),
        cfg,
    )
    assert same > 0.99
    assert different_dob < cfg.similarity_threshold


@pytest.mark.skipif(not HAS_TENSEAL, reason="TenSEAL not installed")
def test_fuzzy_name_same_dob_matches_in_full_he_protocol():
    cfg = MultiAttributeConfig(
        name_weight=0.70,
        dob_weight=0.30,
        similarity_threshold=0.80,
    )
    records_b = [
        MatchRecord("John Smith", "2001-05-17"),
        MatchRecord("Jane Doe", "1999-08-02"),
        MatchRecord("Mary Johnson", "2003-01-10"),
    ]
    result = run_multi_attribute_protocol(
        records_b,
        MatchRecord("Jon Smith", "2001/05/17"),
        cfg=cfg,
        k_mode=1,
        random_state=42,
    )
    assert result.catch is True


@pytest.mark.skipif(not HAS_TENSEAL, reason="TenSEAL not installed")
def test_exact_name_wrong_dob_rejected_in_full_he_protocol():
    cfg = MultiAttributeConfig(
        name_weight=0.70,
        dob_weight=0.30,
        similarity_threshold=0.80,
    )
    records_b = [
        MatchRecord("John Smith", "2001-05-17"),
        MatchRecord("Jane Doe", "1999-08-02"),
    ]
    result = run_multi_attribute_protocol(
        records_b,
        MatchRecord("John Smith", "1990-01-01"),
        cfg=cfg,
        k_mode=1,
        random_state=42,
        early_stop=False,
    )
    assert result.catch is False
