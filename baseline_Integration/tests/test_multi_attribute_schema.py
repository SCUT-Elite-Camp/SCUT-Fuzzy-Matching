"""声明式属性 schema 的测试。

覆盖四块：
1. schema 校验与维度推导；
2. 编码语义（加权和、缺失属性零贡献、空值、可插拔 kind）；
3. 与 V1 的向后兼容（编码逐位相同）；
4. 真实 HE 协议下 3+ 属性可用，且布局不一致会 fail closed。
"""

import importlib.util

import numpy as np
import pytest

from multi_attribute import (
    AttributeRecord,
    AttributeSchema,
    AttributeSpec,
    MatchRecord,
    MultiAttributeConfig,
    SchemaError,
    attribute_similarities,
    encode_attribute,
    encode_record_vectors,
    get_kind,
    plaintext_similarity,
    record_values,
    register_encoder,
    registered_kinds,
    unregister_encoder,
)
from multi_attribute.registry import AttributeBlock, RegistryError

HAS_TENSEAL = importlib.util.find_spec("tenseal") is not None
if HAS_TENSEAL:
    from multi_attribute.protocol import run_multi_attribute_protocol

V1_FP = dict(name_weight=0.70, dob_weight=0.30, similarity_threshold=0.80)


# ---------------------------------------------------------------------------
# schema 校验与维度推导
# ---------------------------------------------------------------------------


def test_default_schema_dims_match_v1():
    schema = MultiAttributeConfig(**V1_FP).to_schema()
    assert schema.cluster_dim == 456
    assert schema.match_dim == 306
    assert schema.spec("dob").kind == "date"


def test_registered_kinds_are_builtin():
    assert set(registered_kinds()) >= {"fuzzy_text", "exact", "date"}


def test_schema_layout_slices():
    schema = MultiAttributeConfig(**V1_FP).to_schema()
    assert schema.layout() == {"name": slice(0, 200), "dob": slice(200, 456)}
    assert schema.match_layout() == {"name": slice(0, 50), "dob": slice(50, 306)}


def test_kind_aliases_are_canonicalized():
    assert AttributeSpec("d", "dob", 1.0).kind == "date"
    assert AttributeSpec("n", "name", 1.0).kind == "fuzzy_text"
    assert AttributeSpec("c", "categorical", 1.0).kind == "exact"


def test_schema_rejects_unknown_kind_and_lists_available():
    with pytest.raises(SchemaError) as exc:
        AttributeSchema([{"name": "x", "kind": "nope", "weight": 1.0}])
    assert "Available" in str(exc.value)


def test_schema_rejects_weights_not_summing_to_one():
    with pytest.raises(SchemaError, match="must sum to 1.0"):
        AttributeSchema(
            [
                {"name": "a", "kind": "exact", "weight": 0.5},
                {"name": "b", "kind": "exact", "weight": 0.2},
            ]
        )


def test_schema_rejects_duplicate_and_blank_and_non_identifier_names():
    for bad in ("", "1abc", "has space", "a-b"):
        with pytest.raises(SchemaError):
            AttributeSchema([{"name": bad, "kind": "exact", "weight": 1.0}])
    with pytest.raises(SchemaError, match="duplicate"):
        AttributeSchema(
            [
                {"name": "a", "kind": "exact", "weight": 0.5},
                {"name": "a", "kind": "exact", "weight": 0.5},
            ]
        )


def test_schema_rejects_dims_over_ckks_slot_limit():
    # 8 个 exact 属性各 1024 维 -> 8192 > CKKS_SLOT_LIMIT(4096)
    with pytest.raises(SchemaError, match="CKKS slot limit"):
        AttributeSchema.from_weights(
            {f"a{i}": 1 for i in range(8)},
            kinds={f"a{i}": "exact" for i in range(8)},
            params={f"a{i}": {"blocks": 1, "buckets_per_block": 1024} for i in range(8)},
        )


def test_fuzzy_text_dim_constraints():
    with pytest.raises(SchemaError, match="cannot exceed cluster_dim"):
        AttributeSchema(
            [
                {
                    "name": "n",
                    "kind": "fuzzy_text",
                    "weight": 1.0,
                    "cluster_dim": 50,
                    "match_dim": 100,
                }
            ]
        )
    with pytest.raises(SchemaError, match="NUM_PERMUTATIONS_CLUSTER"):
        AttributeSchema(
            [{"name": "n", "kind": "fuzzy_text", "weight": 1.0, "cluster_dim": 201}]
        )


def test_unknown_params_are_rejected_not_ignored():
    with pytest.raises(SchemaError, match="unknown params"):
        AttributeSchema(
            [{"name": "n", "kind": "fuzzy_text", "weight": 1.0, "clustr_dim": 100}]
        )


def test_name_attribute_must_be_fuzzy_text():
    with pytest.raises(SchemaError, match="must use kind 'fuzzy_text'"):
        AttributeSchema(
            [{"name": "dob", "kind": "date", "weight": 1.0}], name_attribute="dob"
        )


def test_from_dict_rejects_unknown_top_level_keys():
    with pytest.raises(SchemaError, match="unknown schema keys"):
        AttributeSchema.from_dict(
            {"attributes": [{"name": "a", "kind": "exact", "weight": 1.0}], "typo": 1}
        )


def test_from_weights_normalizes_and_strict_mode_raises():
    schema = AttributeSchema.from_weights({"name": 3, "dob": 1}, kinds={"name": "fuzzy_text", "dob": "date"})
    assert schema.weights == pytest.approx((0.75, 0.25))
    with pytest.raises(SchemaError, match="must sum to 1.0"):
        AttributeSchema.from_weights({"a": 3, "b": 1}, normalize=False)


def test_json_round_trip_is_stable_and_order_sensitive():
    schema = AttributeSchema.from_weights(
        {"name": 0.5, "dob": 0.3, "country": 0.2},
        kinds={"name": "fuzzy_text", "dob": "date", "country": "exact"},
    )
    assert AttributeSchema.from_json(schema.to_json()) == schema
    assert AttributeSchema.from_json(schema.to_json()).fingerprint() == schema.fingerprint()

    reordered = AttributeSchema(list(reversed(schema.attributes)))
    assert reordered.cluster_dim == schema.cluster_dim
    assert reordered.fingerprint() != schema.fingerprint()


# ---------------------------------------------------------------------------
# 编码语义
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,values",
    [
        ("fuzzy_text", ["alpha", None, "beta", "  "]),
        ("exact", ["alpha", None, "beta", "  "]),
        ("date", ["2001-05-17", None, "2001/05/17", "  "]),
    ],
)
def test_encoder_output_shape_matches_declared_dims(kind, values):
    spec = AttributeSpec("a", kind, 1.0)
    block = encode_attribute(spec, values)
    cluster_dim, match_dim = get_kind(kind).dims(spec)
    assert block.cluster.shape == (len(values), cluster_dim)
    assert block.match.shape == (len(values), match_dim)
    # 缺失行（None / 纯空白）必须是全零块。
    assert not block.match[1].any()
    assert not block.match[3].any()


def test_weighted_sum_identity_holds():
    schema = AttributeSchema.from_weights(
        {"name": 0.5, "dob": 0.3, "country": 0.2},
        kinds={"name": "fuzzy_text", "dob": "date", "country": "exact"},
    )
    left = AttributeRecord({"name": "John Smith", "dob": "2001-05-17", "country": "US"})
    right = AttributeRecord({"name": "Jon Smith", "dob": "2001/05/17", "country": "US"})

    sims = attribute_similarities(left, right, schema)
    expected = sum(
        spec.weight * sims[spec.name] for spec in schema.attributes
    )
    assert plaintext_similarity(left, right, schema) == pytest.approx(expected, abs=1e-12)
    assert sims["dob"] == pytest.approx(1.0)
    assert sims["country"] == pytest.approx(1.0)


def test_self_similarity_is_one_for_fully_present_record():
    schema = AttributeSchema.from_weights(
        {"name": 0.5, "dob": 0.5}, kinds={"name": "fuzzy_text", "dob": "date"}
    )
    record = AttributeRecord({"name": "John Smith", "dob": "2001-05-17"})
    assert plaintext_similarity(record, record, schema) == pytest.approx(1.0)


def test_missing_attribute_contributes_zero():
    schema = AttributeSchema.from_weights(
        {"name": 0.5, "dob": 0.5}, kinds={"name": "fuzzy_text", "dob": "date"}
    )
    both = AttributeRecord({"name": "John Smith", "dob": "2001-05-17"})
    no_dob = AttributeRecord({"name": "John Smith"})

    sims = attribute_similarities(both, no_dob, schema)
    assert sims["dob"] == 0.0
    assert plaintext_similarity(both, no_dob, schema) == pytest.approx(0.5 * sims["name"])


def test_blank_fuzzy_text_yields_zero_block():
    """空/纯空白姓名不得拿到任何分数（V1 的 "<empty>" 哨兵行为已被收紧）。"""

    schema = MultiAttributeConfig(**V1_FP).to_schema()
    blank = AttributeRecord({"name": "   ", "dob": "2001-05-17"})
    other = AttributeRecord({"name": "John Smith", "dob": "2001-05-17"})

    sims = attribute_similarities(blank, other, schema)
    assert sims["name"] == 0.0
    # 全零块的范数为 0，因此自相似度也是 0（不是 1）。
    assert sims["dob"] == pytest.approx(1.0)
    assert plaintext_similarity(blank, blank, schema) == pytest.approx(0.3)


def test_exact_digits_normalization_ignores_formatting():
    schema = AttributeSchema.from_weights(
        {"phone": 1.0},
        kinds={"phone": "exact"},
        params={"phone": {"normalize": "digits", "blocks": 1, "buckets_per_block": 512}},
    )
    a = AttributeRecord({"phone": "+1 (555) 010-0199"})
    b = AttributeRecord({"phone": "15550100199"})
    assert plaintext_similarity(a, b, schema) == pytest.approx(1.0)


def test_date_kind_fails_closed_on_garbage():
    schema = AttributeSchema.from_weights({"dob": 1.0}, kinds={"dob": "date"})
    with pytest.raises(ValueError, match="unsupported DOB format"):
        encode_record_vectors([AttributeRecord({"dob": "not-a-date"})], schema)


def test_registry_custom_kind_is_pluggable():
    """注册一个新 kind 后协议层无需任何改动即可使用。"""

    def dims(spec):
        size = int(spec.params.get("size", 4))
        return size, size

    def encode(values, spec):
        size = int(spec.params.get("size", 4))
        rows = np.zeros((len(values), size), dtype=np.float64)
        for i, value in enumerate(values):
            if value is not None:
                rows[i, hash(str(value)) % size] = 1.0
        return AttributeBlock(cluster=rows, match=rows)

    register_encoder("constant_test", dims=dims, encode=encode)
    try:
        schema = AttributeSchema(
            [{"name": "tag", "kind": "constant_test", "weight": 1.0, "size": 4}],
            name_attribute=None,
        )
        assert schema.cluster_dim == 4
        left = AttributeRecord({"tag": "x"})
        right = AttributeRecord({"tag": "x"})
        assert plaintext_similarity(left, right, schema) == pytest.approx(1.0)
    finally:
        unregister_encoder("constant_test")

    with pytest.raises(RegistryError):
        get_kind("constant_test")


def test_record_values_coercion_precedence():
    schema = AttributeSchema.from_weights(
        {"name": 0.6, "dob": 0.4}, kinds={"name": "fuzzy_text", "dob": "date"}
    )
    legacy = MatchRecord("John Smith", "2001-05-17", "B1")
    assert record_values(legacy, schema) == {"name": "John Smith", "dob": "2001-05-17"}

    generic = AttributeRecord({"name": "Jane Doe", "extra": "ignored"})
    assert record_values(generic, schema) == {"name": "Jane Doe", "dob": None}

    mapping = {"name": "Zhang San", "dob": "2005-03-18"}
    assert record_values(mapping, schema) == {"name": "Zhang San", "dob": "2005-03-18"}


def test_match_record_encodes_identically_to_attribute_record():
    schema = MultiAttributeConfig(**V1_FP).to_schema()
    legacy = MatchRecord("John Smith", "2001-05-17")
    generic = AttributeRecord({"name": "John Smith", "dob": "2001-05-17"})
    assert np.array_equal(
        encode_record_vectors([legacy], schema)[1],
        encode_record_vectors([generic], schema)[1],
    )


# ---------------------------------------------------------------------------
# 与 V1 的向后兼容
# ---------------------------------------------------------------------------


def _v1_records():
    return [
        MatchRecord("John Smith", "2001-05-17", "B001"),
        MatchRecord("Jane Doe", "1999-08-02", "B002"),
        MatchRecord("Jon Smythe", "1988-11-23", "B003"),
        MatchRecord("Mary Johnson", "2003-01-10", "B004"),
        MatchRecord("Zhang San", "2005-03-18", "B005"),
        MatchRecord("  Spaced  Name ", "2001.05.17", "B006"),
        MatchRecord("Ana Maria de la Cruz", "1990/12/01", "B007"),
    ]


def test_legacy_config_encoding_is_bit_identical_to_schema_path():
    records = _v1_records()
    cfg = MultiAttributeConfig(**V1_FP)
    c_cfg, m_cfg = encode_record_vectors(records, cfg)
    c_schema, m_schema = encode_record_vectors(records, cfg.to_schema())
    assert np.array_equal(c_cfg, c_schema)
    assert np.array_equal(m_cfg, m_schema)
    assert c_cfg.shape == (len(records), 456)
    assert m_cfg.shape == (len(records), 306)


def test_legacy_plaintext_similarity_pins_v1_doc_values():
    """MULTI_ATTRIBUTE_V1.md 记录的 0.889 / 0.700 必须保持。"""

    cfg = MultiAttributeConfig(**V1_FP)
    fuzzy = plaintext_similarity(
        MatchRecord("Jon Smith", "2001/05/17"),
        MatchRecord("John Smith", "2001-05-17"),
        cfg,
    )
    wrong_dob = plaintext_similarity(
        MatchRecord("John Smith", "1990-01-01"),
        MatchRecord("John Smith", "2001-05-17"),
        cfg,
    )
    assert fuzzy == pytest.approx(0.889, abs=5e-3)
    assert wrong_dob == pytest.approx(0.700, abs=5e-3)


def test_cfg_accepts_schema_mapping_and_config_interchangeably():
    records = _v1_records()
    cfg = MultiAttributeConfig(**V1_FP)
    baseline, _ = encode_record_vectors(records, cfg)
    for alternative in (cfg.to_schema(), cfg.to_schema().to_dict(), None):
        vectors, _ = encode_record_vectors(records, alternative)
        if alternative is None:
            # None 用默认 schema，权重一致但阈值不同 -> 编码仍应相同
            assert np.array_equal(baseline, vectors)
        else:
            assert np.array_equal(baseline, vectors)


# ---------------------------------------------------------------------------
# 真实 HE 协议
# ---------------------------------------------------------------------------


THREE_ATTR = AttributeSchema.from_weights(
    {"name": 0.4, "dob": 0.3, "country": 0.3},
    kinds={"name": "fuzzy_text", "dob": "date", "country": "exact"},
    similarity_threshold=0.80,
)
THREE_ATTR_RECORDS = [
    AttributeRecord({"name": "John Smith", "dob": "2001-05-17", "country": "US"}),
    AttributeRecord({"name": "Jane Doe", "dob": "1999-08-02", "country": "US"}),
    AttributeRecord({"name": "Mary Johnson", "dob": "2003-01-10", "country": "CA"}),
]


@pytest.mark.skipif(not HAS_TENSEAL, reason="TenSEAL not installed")
def test_three_attribute_he_protocol_catches_full_match():
    result = run_multi_attribute_protocol(
        THREE_ATTR_RECORDS,
        AttributeRecord({"name": "Jon Smith", "dob": "2001/05/17", "country": "US"}),
        cfg=THREE_ATTR,
        k_mode=1,
        random_state=42,
    )
    assert result.catch is True


@pytest.mark.skipif(not HAS_TENSEAL, reason="TenSEAL not installed")
def test_three_attribute_he_protocol_rejects_mismatch():
    """姓名相同但 dob+country 全不同：加权分远低于 tau，必须拒绝。"""

    result = run_multi_attribute_protocol(
        THREE_ATTR_RECORDS,
        AttributeRecord({"name": "John Smith", "dob": "1970-01-01", "country": "DE"}),
        cfg=THREE_ATTR,
        k_mode=1,
        random_state=42,
        early_stop=False,
    )
    assert result.catch is False
    assert result.checked_columns == len(THREE_ATTR_RECORDS)


@pytest.mark.skipif(not HAS_TENSEAL, reason="TenSEAL not installed")
def test_wide_schema_he_protocol_still_discriminates():
    """6 属性、match_dim≈1900 的宽向量下阈值判据仍然可分。"""

    schema = AttributeSchema.from_dict(
        {
            "attributes": [
                {"name": "name", "kind": "fuzzy_text", "weight": 0.30,
                 "cluster_dim": 200, "match_dim": 50},
                {"name": "address", "kind": "fuzzy_text", "weight": 0.20,
                 "cluster_dim": 100, "match_dim": 30},
                {"name": "dob", "kind": "date", "weight": 0.20,
                 "blocks": 2, "buckets_per_block": 128, "seed": 20260904},
                {"name": "ssn", "kind": "exact", "weight": 0.15,
                 "blocks": 2, "buckets_per_block": 512},
                {"name": "postcode", "kind": "exact", "weight": 0.10,
                 "blocks": 1, "buckets_per_block": 512},
                {"name": "state", "kind": "exact", "weight": 0.05,
                 "blocks": 1, "buckets_per_block": 64},
            ],
            "similarity_threshold": 0.80,
        }
    )
    assert schema.cluster_dim == 2156
    assert schema.match_dim == 1936

    records = [
        AttributeRecord({
            "name": "John Smith", "address": "8 Stanley Street Miami",
            "dob": "2001-05-17", "ssn": "5304218", "postcode": "4223", "state": "nsw",
        }),
        AttributeRecord({
            "name": "Jane Doe", "address": "3 Light Street",
            "dob": "1999-08-02", "ssn": "1551941", "postcode": "3212", "state": "vic",
        }),
    ]
    query = AttributeRecord({
        "name": "John Smith", "address": "8 Stanley Street Miami",
        "dob": "2001-05-17", "ssn": "5304218", "postcode": "4223", "state": "nsw",
    })
    result = run_multi_attribute_protocol(
        records, query, cfg=schema, k_mode=1, random_state=42
    )
    assert result.catch is True


@pytest.mark.skipif(not HAS_TENSEAL, reason="TenSEAL not installed")
def test_schema_mismatch_between_a_and_b_fails_closed():
    """总维度相同但属性顺序不同 -> 必须报错，而不是算出看似合理的错值。"""

    from multi_attribute.protocol import (
        MultiPartyAState,
        prepare_party_b_multi_offline,
        prepare_party_a_multi_query,
    )

    specs = [
        {"name": "a", "kind": "exact", "weight": 0.5, "blocks": 1, "buckets_per_block": 64},
        {"name": "b", "kind": "exact", "weight": 0.5, "blocks": 1, "buckets_per_block": 64},
    ]
    schema_a = AttributeSchema(specs, name_attribute=None)
    schema_b = AttributeSchema(list(reversed(specs)), name_attribute=None)

    assert schema_a.cluster_dim == schema_b.cluster_dim
    assert schema_a.fingerprint() != schema_b.fingerprint()

    records = [AttributeRecord({"a": "x", "b": "y"})]
    artifacts = prepare_party_b_multi_offline(records, cfg=schema_a, k_mode=1)
    with pytest.raises(ValueError, match="schema mismatch"):
        prepare_party_a_multi_query(
            AttributeRecord({"a": "x", "b": "y"}), artifacts, cfg=schema_b
        )
