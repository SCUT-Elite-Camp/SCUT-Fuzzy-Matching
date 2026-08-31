"""SAGE cleaning and multilingual query validation tests."""

from __future__ import annotations

import numpy as np

from data_pipeline import build_prepared_dataset
from minhash.encoder import batch_encode, generate_signature
from scripts.demo_sage_cross_script import (
    DEFAULT_QUERY_IDS,
    _ANSI_RE,
    _LRI,
    _PDI,
    _clip,
    _display_query,
    _pad,
    _terminal_width,
)
from scripts.validate_sage_multilingual import validate_prepared_dataset


def test_sage_pipeline_cleans_and_audits_the_full_source():
    prepared = build_prepared_dataset("config/examples/sage_pipeline.json")
    manifest = prepared.manifest

    assert manifest.source_stats["source_rows"] == 123479
    assert manifest.source_stats["unique_name_country_pairs"] == 43206
    assert manifest.source_stats["source_duplicate_rows"] == 80273
    assert manifest.source_stats["countries"] == 23
    assert manifest.output_database_rows == 43206
    assert manifest.unique_database_canonical_names == 43092
    assert manifest.matching_database_entries == 56705
    assert manifest.database_script_counts["Arab"] == 3225
    assert manifest.database_script_counts["Han"] == 2052
    assert manifest.database_script_counts["Mymr"] == 1469
    assert manifest.output_query_rows == 9
    assert manifest.orphan_positive_queries == 0

    database_by_id = {record.record_id: record for record in prepared.database}
    for query in prepared.queries:
        for record_id in query.expected_record_ids:
            record = database_by_id[record_id]
            searchable_values = {record.canonical_name, *record.variants.values()}
            assert query.canonical_name in searchable_values

    assert database_by_id["sage-b81ed53599b31a21"].variants[
        "latin_transliterated"
    ] == "liaoxueguang"
    assert database_by_id["sage-a00c0a6bb1497a23"].variants[
        "latin_transliterated"
    ] == "hbh dl jmy lyby lsdy"


def test_sage_demo_defaults_to_all_nine_validation_queries():
    assert len(DEFAULT_QUERY_IDS) == 9
    assert "zh-romanized" in DEFAULT_QUERY_IDS
    assert "ar-romanized" in DEFAULT_QUERY_IDS
    assert "negative-latin" in DEFAULT_QUERY_IDS


def test_sage_demo_tables_display_canonical_names_not_fixture_ids():
    row = {"query_id": "zh-romanized", "canonical_query": "liaoxueguang"}

    assert _display_query(row) == "liaoxueguang"


def test_sage_demo_table_padding_uses_terminal_cell_width():
    assert _terminal_width("liaoxueguang") == 12
    assert _terminal_width("廖學廣") == 6
    assert _terminal_width("ကျော်") == 2
    assert _terminal_width("a\u0301") == 1
    assert _terminal_width("\x1b[32mY\x1b[0m") == 1

    assert _clip("廖學廣", 5) == "廖學…"
    assert _terminal_width(_pad("廖學廣", 8)) == 8
    assert _terminal_width(
        _pad("ဦးအောင်ဆန်းကျော်ခဦးဘီလီဘိုးကျော်", 28)
    ) == 28

    colored = _clip("\x1b[32m廖學廣\x1b[0m", 5)
    assert _ANSI_RE.sub("", colored) == "廖學…"
    assert colored.startswith("\x1b[32m")


def test_sage_demo_table_isolates_arabic_cells_from_column_layout():
    padded = _pad("هبه عادل", 20)
    plain = _ANSI_RE.sub("", padded)

    assert plain.startswith(_LRI)
    assert _PDI in plain
    assert _terminal_width(padded) == 20


def test_minhash_no_longer_collapses_non_latin_names_to_empty_signature():
    names = [
        "هبه عادل عجمي العيبي الساعدي",
        "廖學廣",
        "ဦးအောင်ဆန်းကျော်ခဦးဘီလီဘိုးကျော်",
    ]
    signatures = batch_encode(names, 50)
    empty_signature = generate_signature("", 50)

    assert all(not np.array_equal(signature, empty_signature) for signature in signatures)
    assert len({signature.tobytes() for signature in signatures}) == len(names)


def test_sage_multilingual_queries_work_through_real_he_batch_protocol():
    prepared = build_prepared_dataset("config/examples/sage_pipeline.json")

    result = validate_prepared_dataset(prepared)

    assert result["validation_queries"] == 9
    assert result["passed_queries"] == 9
    assert result["all_passed"] is True
