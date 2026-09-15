"""CSV → 属性记录的桥接层测试，外加 date kind 的脏值策略。

分五块：
1. CSV 读取细节（FEBRL 那种带前导空格的表头、多列拼接、缺失列报错）；
2. 记录 / 查询的加载与真值标签推导；
3. FEBRL 的 ``rec_id`` 正则配对、孤儿副本处理、库/查询限额分离；
4. ``date`` kind 的 ``day_first`` 与 ``on_invalid`` 两条策略；
5. ``demo_multi_attribute_dataset.py`` 的落盘产物（嵌套字段摊平、写盘失败只告警）。

这一层不碰同态加密，所以全部用 ``tmp_path`` 造小 CSV，不依赖下载好的数据集。
真正下载到的数据另有两处按存在性跳过的冒烟测试。
"""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from multi_attribute import AttributeRecord, AttributeSchema
from multi_attribute.dataset import (
    AttributeQuery,
    apply_labels,
    febrl_entity_id,
    load_attribute_queries,
    load_attribute_records,
    load_dataset_pair,
    load_febrl_pair,
)
from multi_attribute.hashing import normalize_dob
from multi_attribute.encoder import encode_record_vectors

ROOT = Path(__file__).resolve().parents[1]
HAS_TENSEAL = importlib.util.find_spec("tenseal") is not None


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# 1. CSV 读取细节
# ---------------------------------------------------------------------------


def test_reader_tolerates_space_padded_header(tmp_path):
    """FEBRL 的真实表头是 ``rec_id, given_name, surname`` —— 列名前带空格。"""

    path = tmp_path / "febrl_like.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write("rec_id, given_name, surname\n")
        handle.write("rec-1-org, michaela, neumann\n")

    records = load_attribute_records(
        path, {"name": ["given_name", "surname"]}, id_column="rec_id"
    )
    assert len(records) == 1
    assert records[0].record_id == "rec-1-org"
    assert records[0].values["name"] == "michaela neumann"


def test_project_row_joins_columns_and_skips_empties(tmp_path):
    path = _write_csv(
        tmp_path / "a.csv",
        ["id", "street", "addr1", "addr2"],
        [
            {"id": "1", "street": "8", "addr1": "stanley street", "addr2": "miami"},
            {"id": "2", "street": "3", "addr1": "", "addr2": "pinehill"},
        ],
    )
    records = load_attribute_records(
        path, {"address": ["street", "addr1", "addr2"]}, id_column="id"
    )
    assert records[0].values["address"] == "8 stanley street miami"
    # 空列只是被跳过，不会留下多余空格。
    assert records[1].values["address"] == "3 pinehill"


def test_project_row_omits_attribute_when_all_columns_empty(tmp_path):
    """整条属性为空 => 键不写入 => 编码成全零块 => 该属性零贡献。"""

    path = _write_csv(
        tmp_path / "a.csv",
        ["id", "given_name", "surname", "postcode"],
        [{"id": "1", "given_name": "", "surname": "   ", "postcode": "4223"}],
    )
    records = load_attribute_records(
        path, {"name": ["given_name", "surname"], "postcode": "postcode"}, id_column="id"
    )
    assert "name" not in records[0].values
    assert records[0].values["postcode"] == "4223"


def test_row_with_every_attribute_empty_is_skipped(tmp_path):
    """整行都没有可用取值时整条丢弃 —— 全零向量对匹配没有任何信息量。"""

    path = _write_csv(
        tmp_path / "a.csv",
        ["id", "given_name"],
        [{"id": "1", "given_name": "  "}, {"id": "2", "given_name": "kept"}],
    )
    records = load_attribute_records(path, {"name": "given_name"}, id_column="id")
    assert [r.record_id for r in records] == ["2"]


def test_unknown_column_raises(tmp_path):
    path = _write_csv(tmp_path / "a.csv", ["id", "given_name"], [{"id": "1", "given_name": "x"}])
    with pytest.raises(ValueError, match="unknown column 'surname'"):
        load_attribute_records(path, {"name": ["given_name", "surname"]}, id_column="id")


def test_unknown_id_column_raises(tmp_path):
    path = _write_csv(tmp_path / "a.csv", ["id", "given_name"], [{"id": "1", "given_name": "x"}])
    with pytest.raises(ValueError, match="unknown id column"):
        load_attribute_records(path, {"name": "given_name"}, id_column="nope")


def test_empty_csv_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("id,given_name\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no data rows"):
        load_attribute_records(path, {"name": "given_name"}, id_column="id")


# ---------------------------------------------------------------------------
# 2. 记录与查询加载
# ---------------------------------------------------------------------------


def test_generated_ids_when_no_id_column(tmp_path):
    path = _write_csv(
        tmp_path / "a.csv",
        ["given_name"],
        [{"given_name": "a"}, {"given_name": "b"}],
    )
    records = load_attribute_records(path, {"name": "given_name"})
    assert [r.record_id for r in records] == ["row-1", "row-2"]


def test_limit_stops_early(tmp_path):
    path = _write_csv(
        tmp_path / "a.csv",
        ["id", "given_name"],
        [{"id": str(i), "given_name": f"n{i}"} for i in range(10)],
    )
    assert len(load_attribute_records(path, {"name": "given_name"}, id_column="id", limit=3)) == 3


def test_rows_with_no_values_at_all_are_skipped(tmp_path):
    path = _write_csv(
        tmp_path / "a.csv",
        ["id", "given_name"],
        [{"id": "1", "given_name": ""}, {"id": "2", "given_name": "kept"}],
    )
    records = load_attribute_records(path, {"name": "given_name"}, id_column="id")
    assert [r.record_id for r in records] == ["2"]


def test_query_labels_from_label_column(tmp_path):
    path = _write_csv(
        tmp_path / "q.csv",
        ["qid", "given_name", "match"],
        [
            {"qid": "q1", "given_name": "a", "match": "true"},
            {"qid": "q2", "given_name": "b", "match": "0"},
        ],
    )
    queries = load_attribute_queries(
        path, {"name": "given_name"}, id_column="qid", label_column="match"
    )
    assert [q.label for q in queries] == [True, False]
    assert queries[0].record.record_id == "q1"


def test_negative_query_never_carries_expected_ids(tmp_path):
    """label 为假时必须清空 expected —— 否则会写出自相矛盾的用例。"""

    path = _write_csv(
        tmp_path / "q.csv",
        ["qid", "given_name", "match", "targets"],
        [{"qid": "q1", "given_name": "a", "match": "false", "targets": "ent-9"}],
    )
    queries = load_attribute_queries(
        path,
        {"name": "given_name"},
        id_column="qid",
        label_column="match",
        expected_ids_column="targets",
    )
    assert queries[0].label is False
    assert queries[0].expected_record_ids == frozenset()


def test_expected_ids_column_splits_and_implies_label(tmp_path):
    path = _write_csv(
        tmp_path / "q.csv",
        ["qid", "given_name", "targets"],
        [{"qid": "q1", "given_name": "a", "targets": "ent-1 | ent-2"}],
    )
    queries = load_attribute_queries(
        path, {"name": "given_name"}, id_column="qid", expected_ids_column="targets"
    )
    assert queries[0].label is True
    assert queries[0].expected_record_ids == frozenset({"ent-1", "ent-2"})


def test_load_dataset_pair_uses_separate_database_limit(tmp_path):
    database = _write_csv(
        tmp_path / "db.csv",
        ["id", "given_name"],
        [{"id": str(i), "given_name": f"n{i}"} for i in range(10)],
    )
    queries = _write_csv(
        tmp_path / "q.csv",
        ["qid", "given_name"],
        [{"qid": f"q{i}", "given_name": f"n{i}"} for i in range(10)],
    )
    db, qs = load_dataset_pair(
        database,
        queries,
        {"name": "given_name"},
        id_column="id",
        query_id_column="qid",
        limit=2,
        database_limit=7,
    )
    assert len(db) == 7
    assert len(qs) == 2


# ---------------------------------------------------------------------------
# 3. FEBRL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rec_id,expected",
    [
        ("rec-1070-org", "1070"),
        ("rec-1070-dup-0", "1070"),
        ("rec-1070-dup-12", "1070"),
        ("rec-1-org", "1"),
    ],
)
def test_febrl_entity_id_parses(rec_id, expected):
    assert febrl_entity_id(rec_id) == expected


@pytest.mark.parametrize("rec_id", ["1070", "rec-1070", "rec-x-org", "", "dup-1070"])
def test_febrl_entity_id_rejects_malformed(rec_id):
    with pytest.raises(ValueError, match="malformed FEBRL rec_id"):
        febrl_entity_id(rec_id)


def _febrl_files(tmp_path, originals: list[tuple[str, str]], duplicates: list[tuple[str, str]]):
    columns = ["rec_id", "given_name", "surname", "date_of_birth"]
    a = _write_csv(
        tmp_path / "4a.csv",
        columns,
        [
            {"rec_id": rid, "given_name": name, "surname": "smith", "date_of_birth": "19900101"}
            for rid, name in originals
        ],
    )
    b = _write_csv(
        tmp_path / "4b.csv",
        columns,
        [
            {"rec_id": rid, "given_name": name, "surname": "smith", "date_of_birth": "19900101"}
            for rid, name in duplicates
        ],
    )
    return a, b, {"name": ["given_name", "surname"], "dob": "date_of_birth"}


def test_febrl_pair_derives_truth_from_rec_id(tmp_path):
    a, b, columns = _febrl_files(
        tmp_path,
        [("rec-10-org", "alice"), ("rec-20-org", "bob")],
        [("rec-10-dup-0", "alice"), ("rec-20-dup-0", "bobby")],
    )
    database, queries = load_febrl_pair(a, b, columns)
    assert [r.record_id for r in database] == ["rec-10-org", "rec-20-org"]
    assert all(q.label for q in queries)
    assert queries[1].expected_record_ids == frozenset({"rec-20-org"})


def test_febrl_orphan_duplicates_are_dropped_not_mislabelled(tmp_path):
    """原始记录不在库里的副本是**假负例**，必须丢弃而不是标成 False。"""

    a, b, columns = _febrl_files(
        tmp_path,
        [("rec-10-org", "alice")],
        [("rec-10-dup-0", "alice"), ("rec-99-dup-0", "ghost")],
    )
    _, queries = load_febrl_pair(a, b, columns)
    assert [q.record.record_id for q in queries] == ["rec-10-dup-0"]
    assert all(q.label for q in queries)


def test_febrl_orphans_can_be_kept_as_negatives(tmp_path):
    a, b, columns = _febrl_files(
        tmp_path,
        [("rec-10-org", "alice")],
        [("rec-10-dup-0", "alice"), ("rec-99-dup-0", "ghost")],
    )
    _, queries = load_febrl_pair(a, b, columns, drop_orphans=False)
    assert [q.label for q in queries] == [True, False]
    assert queries[1].expected_record_ids == frozenset()


def test_febrl_limit_applies_to_queries_and_database_limit_to_database(tmp_path):
    """``limit`` 只管查询条数 —— 4a 的 rec_id 不是升序，套在库上会制造一堆孤儿。"""

    originals = [(f"rec-{i}-org", f"n{i}") for i in range(1, 9)]
    duplicates = [(f"rec-{i}-dup-0", f"n{i}") for i in range(1, 9)]
    a, b, columns = _febrl_files(tmp_path, originals, duplicates)

    database, queries = load_febrl_pair(a, b, columns, limit=3, database_limit=5)
    assert len(database) == 5
    assert len(queries) == 3
    # 只留下原始记录确实在库内的副本，所以全部是真匹配。
    assert all(q.label for q in queries)


def test_febrl_limit_counts_surviving_queries_not_raw_rows(tmp_path):
    """前若干行全是孤儿时，limit=2 仍应拿到 2 条可用查询。"""

    a, b, columns = _febrl_files(
        tmp_path,
        [("rec-10-org", "alice"), ("rec-20-org", "bob")],
        [
            ("rec-90-dup-0", "ghost1"),
            ("rec-91-dup-0", "ghost2"),
            ("rec-10-dup-0", "alice"),
            ("rec-20-dup-0", "bob"),
        ],
    )
    _, queries = load_febrl_pair(a, b, columns, limit=2)
    assert [q.record.record_id for q in queries] == ["rec-10-dup-0", "rec-20-dup-0"]


# ---------------------------------------------------------------------------
# 4. date kind 的脏值与日在前策略
# ---------------------------------------------------------------------------


def test_normalize_dob_rejects_day_first_by_default():
    """DD/MM/YYYY 与 MM/DD/YYYY 有歧义，默认必须 fail closed。"""

    with pytest.raises(ValueError, match="unsupported DOB format"):
        normalize_dob("05/03/2001")
    assert normalize_dob("05/03/2001", day_first=True) == "2001-03-05"


def test_normalize_dob_accepts_day_first_variants():
    assert normalize_dob("25/12/1990", day_first=True) == "1990-12-25"
    assert normalize_dob("25-12-1990", day_first=True) == "1990-12-25"
    assert normalize_dob("25.12.1990", day_first=True) == "1990-12-25"


def test_date_kind_default_fails_closed_on_garbage():
    schema = AttributeSchema.from_dict(
        {"attributes": [{"name": "dob", "kind": "date", "weight": 1.0, "blocks": 1,
                         "buckets_per_block": 16}]}
    )
    with pytest.raises(ValueError, match="unparseable date value"):
        encode_record_vectors([AttributeRecord({"dob": "19450493"})], schema)


def test_date_kind_on_invalid_missing_zeroes_instead_of_raising():
    """FEBRL 4b 有 1.3% 的非法日期；声明 missing 后应当降级为全零块。"""

    schema = AttributeSchema.from_dict(
        {"attributes": [{"name": "dob", "kind": "date", "weight": 1.0, "blocks": 1,
                         "buckets_per_block": 16, "on_invalid": "missing"}]}
    )
    _, vectors = encode_record_vectors(
        [AttributeRecord({"dob": "19450493"}), AttributeRecord({"dob": "1990-01-01"})], schema
    )
    assert not vectors[0].any(), "unparseable date should contribute nothing"
    assert vectors[1].any()


def test_date_kind_day_first_changes_the_encoding():
    base = {"name": "dob", "kind": "date", "weight": 1.0, "blocks": 1, "buckets_per_block": 64}
    naive = AttributeSchema.from_dict({"attributes": [base]})
    flagged = AttributeSchema.from_dict({"attributes": [{**base, "day_first": True}]})
    pairs = [AttributeRecord({"dob": "1990-12-25"}), AttributeRecord({"dob": "25/12/1990"})]

    # 默认策略下 25/12/1990 连解析都过不了（没有第 25 个月）。
    with pytest.raises(ValueError, match="unparseable date value"):
        encode_record_vectors(pairs, naive)

    # 声明 day_first 之后，两种写法指同一天，编码必须逐位相同。
    _, vectors = encode_record_vectors(pairs, flagged)
    assert (vectors[0] == vectors[1]).all()


def test_date_kind_rejects_unknown_on_invalid_mode():
    with pytest.raises(ValueError, match="on_invalid must be one of"):
        AttributeSchema.from_dict(
            {"attributes": [{"name": "dob", "kind": "date", "weight": 1.0,
                             "on_invalid": "explode"}]}
        )


# ---------------------------------------------------------------------------
# 5. 示例 config 与真实数据（按存在性跳过）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected_cluster,expected_match",
    [
        ("febrl_multi_attribute.json", 2156, 1936),
        ("multi_attribute_schema.json", 1076, 856),
    ],
)
def test_example_job_files_declare_documented_dims(name, expected_cluster, expected_match):
    """示例 config 里写在注释里的维度必须和实际推导出来的一致。"""

    path = ROOT / "config" / "examples" / name
    job = json.loads(path.read_text(encoding="utf-8"))
    schema = AttributeSchema.from_dict(job["schema"])
    assert schema.cluster_dim == expected_cluster
    assert schema.match_dim == expected_match

    # 每个属性都要有列映射，否则 demo 会在运行时才报错。
    assert set(job["attribute_columns"]) == {s.name for s in schema.attributes}


FEBRL_DIR = ROOT / "dataset" / "febrl"
SYNTHETIC_DIR = ROOT / "dataset" / "synthetic"


@pytest.mark.skipif(not (FEBRL_DIR / "dataset4a.csv").exists(), reason="run fetch_dataset.py first")
def test_real_febrl_loads_with_truth():
    job = json.loads((ROOT / "config" / "examples" / "febrl_multi_attribute.json").read_text("utf-8"))
    database, queries = load_febrl_pair(
        FEBRL_DIR / "dataset4a.csv",
        FEBRL_DIR / "dataset4b.csv",
        job["attribute_columns"],
        id_column="rec_id",
        limit=20,
        database_limit=500,
    )
    assert len(database) == 500
    assert all(q.label for q in queries)
    assert all(len(q.expected_record_ids) == 1 for q in queries)


@pytest.mark.skipif(
    not (SYNTHETIC_DIR / "queries.csv").exists(),
    reason="run generate_synthetic_attributes.py first",
)
def test_real_synthetic_dataset_labels_agree_with_db_records():
    job = json.loads((ROOT / "config" / "examples" / "multi_attribute_schema.json").read_text("utf-8"))
    database, queries = load_dataset_pair(
        SYNTHETIC_DIR / "entities.csv",
        SYNTHETIC_DIR / "queries.csv",
        job["attribute_columns"],
        id_column="entity_id",
        query_id_column="query_id",
        limit=50,
    )
    queries = apply_labels(queries, SYNTHETIC_DIR / "labels.csv")
    ids = {r.record_id for r in database}
    positive = [q for q in queries if q.label]
    assert positive, "generator should emit interleaved positives even in a short prefix"
    # 每条正例的真值 id 必须真的在库里 —— 否则是生成器写坏了 labels.csv。
    for query in positive:
        assert query.expected_record_ids <= ids
    assert any(not q.label for q in queries), "interleaving should surface negatives early"


@pytest.mark.skipif(not HAS_TENSEAL, reason="TenSEAL not installed")
@pytest.mark.skipif(
    not (SYNTHETIC_DIR / "queries.csv").exists(),
    reason="run generate_synthetic_attributes.py first",
)
def test_real_synthetic_dataset_runs_through_the_encrypted_protocol():
    """真实 CSV → schema → 加密双轮，端到端跑通一次。"""

    from multi_attribute.protocol import run_multi_attribute_protocol

    job = json.loads((ROOT / "config" / "examples" / "multi_attribute_schema.json").read_text("utf-8"))
    schema = AttributeSchema.from_dict(job["schema"])
    database, queries = load_dataset_pair(
        SYNTHETIC_DIR / "entities.csv",
        SYNTHETIC_DIR / "queries.csv",
        job["attribute_columns"],
        id_column="entity_id",
        query_id_column="query_id",
        limit=12,
    )
    queries = apply_labels(queries, SYNTHETIC_DIR / "labels.csv")
    positives = [q for q in queries if q.label]
    assert positives

    caught = 0
    for query in positives[:3]:
        run = run_multi_attribute_protocol(database, query.record, cfg=schema, k_mode=1, random_state=42)
        caught += int(run.catch)
    # k=1 时不存在簇丢失，正例应当稳定命中。
    assert caught == 3


# ---------------------------------------------------------------------------
# demo 的落盘产物
# ---------------------------------------------------------------------------


@pytest.fixture
def demo():
    """demo 脚本顶部会 import protocol（进而 import tenseal），没装就跳过。"""

    pytest.importorskip("tenseal")
    import scripts.demo_multi_attribute_dataset as module

    return module


def _sample_row(**overrides) -> dict:
    row = {
        "query_id": "q-1",
        "label": True,
        "expected_record_ids": ["e-1"],
        "top1_record_id": "e-1",
        "top1_score": 0.9,
        "impostor_score": 0.4,
        "similarities": {"name": 0.8, "dob": 1.0},
        "encrypted": {
            "catch": True,
            "agrees_with_label": True,
            "selected_cluster": 3,
            "checked_columns": 7,
            "true_match_cluster": 3,
            "cluster_hit": True,
        },
    }
    row.update(overrides)
    return row


def test_artifact_names_follow_sibling_convention(demo):
    """目录去掉 demo_ 前缀、文件名保留全名，与 ncvr_matches / sage_cross_script 一致。"""

    assert demo._ARTIFACT_NAME == "demo_multi_attribute_dataset"
    assert demo._ARTIFACT_DIR_NAME == "multi_attribute_dataset"


def test_csv_rows_flatten_nested_fields(demo):
    out = demo._csv_rows([_sample_row()])[0]

    # 嵌套的 similarities 摊成 sim_<属性名> 列 —— 换 schema 就换列名，可直接画标定图。
    assert out["sim_name"] == 0.8 and out["sim_dob"] == 1.0
    assert "similarities" not in out

    # encrypted 摊成判词 + 独立数值列，而不是把 dict 的 repr 塞进单元格。
    assert out["encrypted"] == "CATCH"
    assert out["enc_checked_columns"] == 7
    assert out["enc_cluster_hit"] is True

    # 真值集合在 JSON 里是列表，CSV 里用 | 连接。
    assert out["expected_record_ids"] == "e-1"


def test_csv_rows_join_multiple_expected_ids(demo):
    out = demo._csv_rows([_sample_row(expected_record_ids=["e-1", "e-2"])])[0]
    assert out["expected_record_ids"] == "e-1|e-2"


def test_csv_rows_blank_encrypted_columns_when_protocol_skipped(demo):
    """--no-encrypted 时加密列留空，明文列照常保留 —— 报表仍然可用。"""

    out = demo._csv_rows([_sample_row(encrypted=None)])[0]
    assert out["encrypted"] == ""
    for key in (
        "enc_agrees_with_label",
        "enc_selected_cluster",
        "enc_checked_columns",
        "enc_true_match_cluster",
        "enc_cluster_hit",
    ):
        assert out[key] == ""
    assert out["top1_score"] == 0.9
    assert out["sim_name"] == 0.8


def test_csv_rows_no_catch_wording(demo):
    row = _sample_row(encrypted={**_sample_row()["encrypted"], "catch": False})
    assert demo._csv_rows([row])[0]["encrypted"] == "no-catch"


def test_save_outputs_round_trips_json_and_csv(demo, tmp_path):
    """两种形态的行混在一起也不能炸 —— DictWriter 对多余键默认是直接报错。"""

    result = {"schema": {"fingerprint": "abc"}, "rows": [_sample_row(), _sample_row(query_id="q-2", encrypted=None)]}
    json_path, csv_path = demo._save_outputs(result, tmp_path)

    assert json_path is not None and csv_path is not None
    assert json.loads(json_path.read_text(encoding="utf-8")) == result

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["query_id"] for r in rows] == ["q-1", "q-2"]
    assert rows[1]["encrypted"] == ""


def test_save_outputs_creates_missing_directory(demo, tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    json_path, csv_path = demo._save_outputs({"rows": [_sample_row()]}, nested)
    assert json_path is not None and json_path.parent == nested


def test_save_outputs_warns_instead_of_raising(demo, tmp_path, capsys):
    """报告已经打到终端了，写盘失败不该让整个 run 白跑。"""

    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    json_path, csv_path = demo._save_outputs({"rows": [_sample_row()]}, blocker)

    assert json_path is None and csv_path is None
    assert "could not save demo artifacts" in capsys.readouterr().out
