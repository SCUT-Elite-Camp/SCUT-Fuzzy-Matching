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
from multi_attribute.encoder import encode_record_vectors, plaintext_similarity

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
        "expected_score": 0.9,
        "expected_score_margin": 0.3,
        "tau": 0.6,
        "should_catch": True,
        "impostor_score": 0.4,
        "similarities": {"name": 0.8, "dob": 1.0},
        "encrypted": {
            "catch": True,
            "agrees_with_label": True,
            "selected_cluster": 3,
            "checked_columns": 7,
            "true_match_cluster": 3,
            "cluster_hit": True,
            "miss_cause": None,
        },
    }
    row.update(overrides)
    return row


def _summary_row(
    demo,
    query_id="q-1",
    *,
    label=True,
    expected_score=0.9,
    tau=0.6,
    catch=True,
    cluster_hit=True,
    impostor_score=0.4,
    encrypted=True,
) -> dict:
    """构造 ``_summarize`` 的最小输入。

    ``should_catch`` / ``expected_score_margin`` / ``miss_cause`` 一律调 demo 自己的
    辅助函数推导，不在测试里重写一遍 —— 重写就等于把断言变成对实现的复述，
    实现改错了测试也跟着改错。
    """

    should_catch = demo._should_catch(expected_score, tau)
    row = _sample_row(
        query_id=query_id,
        label=label,
        expected_record_ids=["e-1"] if label else [],
        top1_record_id="e-1" if label else None,
        top1_score=0.9,
        expected_score=expected_score,
        expected_score_margin=(
            None if expected_score is None else expected_score - tau
        ),
        tau=tau,
        should_catch=should_catch,
        impostor_score=impostor_score,
        encrypted=None,
    )
    if encrypted:
        row["encrypted"] = {
            "catch": catch,
            "agrees_with_label": catch == label,
            "selected_cluster": 3,
            "checked_columns": 7,
            "true_match_cluster": 3 if cluster_hit else 5,
            "cluster_hit": cluster_hit,
            "miss_cause": demo._miss_cause(
                should_catch, catch, cluster_hit, expected_score, tau
            ),
        }
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
        "enc_miss_cause",
    ):
        assert out[key] == ""
    assert out["top1_score"] == 0.9
    assert out["sim_name"] == 0.8


def test_csv_rows_no_catch_wording(demo):
    row = _sample_row(encrypted={**_sample_row()["encrypted"], "catch": False})
    assert demo._csv_rows([row])[0]["encrypted"] == "no-catch"


def test_csv_rows_carry_expected_score_and_tau(demo):
    """新列必须落在 CSV 里 —— 标定时要看的就是 expected_score 而不是 top1_score。"""

    row = _sample_row(encrypted={**_sample_row()["encrypted"], "catch": False, "miss_cause": "round1_cluster"})
    out = demo._csv_rows([row])[0]
    assert out["expected_score"] == 0.9
    assert out["tau"] == 0.6
    assert out["should_catch"] is True
    assert out["expected_score_margin"] == pytest.approx(0.3)
    assert out["enc_miss_cause"] == "round1_cluster"

    # 命中时 miss_cause 是 None，写进 CSV 必须变成空串而不是 "None"。
    assert demo._csv_rows([_sample_row()])[0]["enc_miss_cause"] == ""


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


# ---------------------------------------------------------------------------
# 汇总口径：should_catch 是召回率的分母
#
# 这一组守的是本次改动的核心语义，以及围绕它建的四道防刷分护栏。全部是纯函数，
# 不碰 HE、不碰磁盘。
# ---------------------------------------------------------------------------


def _rows_for_recall(demo):
    """4 条正例：2 条应 catch 且 catch 了，1 条应 catch 却没 catch，1 条不该 catch。"""

    return [
        _summary_row(demo, "q-ident", expected_score=1.0, catch=True),
        _summary_row(demo, "q-caught", expected_score=0.8, catch=True),
        _summary_row(demo, "q-missed", expected_score=0.75, catch=False, cluster_hit=False),
        _summary_row(demo, "q-low", expected_score=0.5, catch=False, cluster_hit=False),
    ]


def test_summary_excludes_below_tau_positives_from_recall(demo):
    """分母是 should_catch，不是"标签为真" —— 低于 tau 的正例两边都不进。"""

    summary = demo._summarize(_rows_for_recall(demo), tau=0.6, encrypted=True)

    assert summary["positives"] == 4
    assert summary["should_catch_positives"] == 3
    assert summary["positives_below_tau"] == 1
    assert summary["positives_below_tau_ids"] == ["q-low"]

    assert summary["encrypted_catches_positives"] == 2
    assert summary["recall_should_catch"] == pytest.approx(2 / 3)

    # q-low 既不进分子也不进分母：它不该出现在失败名单里，那会暗示协议漏了它。
    assert "q-low" not in summary["should_catch_miss_ids"]
    assert summary["should_catch_misses"] == 1

    # 老口径（全正例）作为对照仍然在，分子相同、分母更大。
    assert summary["encrypted_recall_all_positives"] == pytest.approx(2 / 4)


def test_summary_counts_absolute_misses_and_attributes_round1(demo):
    """比率之外必须给出绝对失败条数与归因 —— 只看百分比看不出协议是不是坏了。"""

    summary = demo._summarize(_rows_for_recall(demo), tau=0.6, encrypted=True)

    assert summary["should_catch_misses"] == 1
    assert summary["should_catch_miss_ids"] == ["q-missed"]
    assert summary["misses_round1_cluster"] == 1
    assert summary["misses_boundary"] == 0
    assert summary["misses_unexplained"] == 0
    assert summary["misses_unexplained_ids"] == []

    # 与 tau 无关的对照量：真匹配落没落进选中的簇。
    # 4 条正例里只有 q-ident / q-caught 选对了簇；q-low 低于阈值但仍算一次选簇失败，
    # 所以 cluster_recall 的分母是全部正例，不是 should_catch 子集。
    assert summary["cluster_hits"] == 2
    assert summary["cluster_recall"] == pytest.approx(2 / 4)
    # 这个才是 tau 无关的绝对漏掉条数 —— 判 run 要看它，不是 should_catch_misses。
    assert summary["cluster_misses"] == 2


def test_summary_flags_unexplained_miss(demo):
    """真匹配就在选中簇里、分数又超噪声带，第二轮没有理由漏 —— 必须报 unexplained。"""

    rows = [
        # margin 0.3：离阈值很远，选簇也对，漏了就是协议 bug。
        _summary_row(demo, "q-a", expected_score=0.9, catch=False, cluster_hit=True),
        # margin 5e-5：卡在 (tau, tau + eps] 的 CKKS 噪声带里，两边倒都正常。
        _summary_row(demo, "q-b", expected_score=0.6 + 5e-5, catch=False, cluster_hit=True),
    ]
    summary = demo._summarize(rows, tau=0.6, encrypted=True)

    assert rows[0]["encrypted"]["miss_cause"] == "unexplained"
    assert rows[1]["encrypted"]["miss_cause"] == "boundary"
    assert summary["misses_unexplained"] == 1
    assert summary["misses_unexplained_ids"] == ["q-a"]
    assert summary["misses_boundary"] == 1
    assert summary["misses_round1_cluster"] == 0


def test_summary_identical_query_must_catch(demo):
    """用户点名的硬要求：未扰动的查询必能 catch。违反要能指名道姓。"""

    rows = [_summary_row(demo, "q-x", expected_score=1.0, catch=False, cluster_hit=False)]
    summary = demo._summarize(rows, tau=0.6, encrypted=True)

    assert summary["identical_queries"] == 1
    assert summary["identical_caught"] == 0
    assert summary["identical_violations"] == ["q-x"]

    # 没有完全一致的查询时，"0 条违反"是空跑，必须和"有 6 条全部命中"区分开 ——
    # 否则这个检查绿得毫无意义。
    none_identical = demo._summarize(
        [_summary_row(demo, "q-y", expected_score=0.97)], tau=0.6, encrypted=True
    )
    assert none_identical["identical_queries"] == 0
    assert none_identical["identical_violations"] == []


def test_summary_tau_rise_shrinks_denominator_and_inflates_recall(demo):
    """防刷分护栏的核心断言：涨 tau 会让 headline 数字变好看，而一条错都没修。

    构造上让两条失败的正例分最低 —— 抬阈值恰好把它们从分母里删掉，于是 recall
    从 0.5 跳到 1.0。同一批行的 ``cluster_recall`` 完全不动，因为选簇与 tau 无关；
    它就是"什么都没变好"的那个锚点。
    """

    rows = [
        _summary_row(demo, "q-good-1", expected_score=1.00, catch=True),
        _summary_row(demo, "q-good-2", expected_score=0.90, catch=True),
        _summary_row(demo, "q-bad-1", expected_score=0.74, catch=False, cluster_hit=False),
        _summary_row(demo, "q-bad-2", expected_score=0.70, catch=False, cluster_hit=False),
    ]
    low = demo._summarize(rows, tau=0.6, encrypted=True)
    high = demo._summarize(rows, tau=0.8, encrypted=True)

    # 分母缩水、比率上升、失败条数"归零"。
    assert high["should_catch_positives"] < low["should_catch_positives"]
    assert high["recall_should_catch"] > low["recall_should_catch"]
    assert low["recall_should_catch"] == pytest.approx(2 / 4)
    assert high["recall_should_catch"] == pytest.approx(2 / 2)
    assert high["should_catch_misses"] == 0

    # 但底层失败一条没少：两条查询仍然 catch=False，只是不再被计入分母。
    assert sum(1 for row in rows if not row["encrypted"]["catch"]) == 2
    assert high["positives_below_tau"] == 2

    # 与 tau 无关的锚点：选簇表现一模一样。raising tau 什么都没修好。
    # 注意 should_catch_misses 从 2 掉到 0（上面断言过），而 cluster_misses 纹丝不动 ——
    # 前者的分母含 tau，后者不含。屏幕上标为 tau-DEPENDENT 的就是前者。
    assert high["cluster_recall"] == low["cluster_recall"] == pytest.approx(2 / 4)
    assert high["cluster_misses"] == low["cluster_misses"] == 2
    assert high["should_catch_misses"] != low["should_catch_misses"]

    # 护栏必须在屏幕上把这件事说出来，并且是"有代价"那一档。
    guard = high["tau_guard"]
    assert guard["tau_exceeds_max_impostor"] is True
    assert guard["verdict"] == "above_ceiling_with_cost"
    assert guard["positives_dropped_by_tau"] == 2
    assert guard["positives_dropped_by_tau"] == (
        low["should_catch_positives"] - high["should_catch_positives"]
    )
    assert high["highest_impostor_score"] == pytest.approx(0.4)
    # tau=0.6 时两条失败的正例都还在分母里（0.74 / 0.70 > 0.6），所以那一档是"无代价"。
    assert low["tau_guard"]["verdict"] == "above_ceiling_no_cost"


def test_tau_guard_verdict_is_graded_by_cost(demo):
    """护栏必须按代价分级。

    ``tau > 最高冒充者分`` 本身就是标定正确的样子 —— 阈值就该盖住冒充者上限。把
    它本身当成警告，会让健康 run 和刷分 run 打出同一句话，读者学会忽略它之后，
    真出事那次也就没人看了。
    """

    # 盖住冒充者上限且一条正例没删 —— 健康，不该报警。
    clean = demo._summarize(
        [_summary_row(demo, "q-1", expected_score=0.9, impostor_score=0.4, catch=True)],
        tau=0.6,
        encrypted=True,
    )
    assert clean["tau_guard"]["verdict"] == "above_ceiling_no_cost"

    # 盖住了，但为此删掉了正例 —— 这才是要报警的形态。
    costly = demo._summarize(
        [
            _summary_row(demo, "q-1", expected_score=0.9, impostor_score=0.4, catch=True),
            _summary_row(demo, "q-2", expected_score=0.5, impostor_score=0.4,
                         catch=False, cluster_hit=False),
        ],
        tau=0.6,
        encrypted=True,
    )
    assert costly["tau_guard"]["verdict"] == "above_ceiling_with_cost"
    assert costly["tau_guard"]["positives_dropped_by_tau"] == 1

    # 反方向：阈值没盖过冒充者，冒充者能直接造成假正例。
    leaky = demo._summarize(
        [_summary_row(demo, "q-1", expected_score=0.9, impostor_score=0.8, catch=True)],
        tau=0.6,
        encrypted=True,
    )
    assert leaky["tau_guard"]["verdict"] == "below_ceiling"
    assert leaky["tau_guard"]["tau_exceeds_max_impostor"] is False

    # 没有冒充者样本（阈值比不出来）时不能瞎判。
    unknown = demo._summarize(
        [_summary_row(demo, "q-1", expected_score=None, impostor_score=None)],
        tau=0.6,
        encrypted=False,
    )
    assert unknown["highest_impostor_score"] is None
    assert unknown["tau_guard"]["verdict"] == "unknown"


def test_should_catch_is_strict_at_tau(demo):
    """明文判据必须与第二轮同符号：协议判的是 value > eps，等价于 score > tau。"""

    from config.params import MULTI_ATTRIBUTE_DECRYPT_EPS

    assert demo._should_catch(0.6, 0.6) is False
    assert demo._should_catch(1.0, 1.0) is False
    assert demo._should_catch(0.6001, 0.6) is True
    assert demo._should_catch(None, 0.6) is None
    # 噪声带宽必须取自协议常量，不能就地硬编码一个数。
    assert demo._BOUNDARY_EPS == MULTI_ATTRIBUTE_DECRYPT_EPS


def test_summary_counts_over_reports(demo):
    """负例被 catch 是新口径下的假正例，要单独计数，不能混进召回率。"""

    rows = [
        _summary_row(demo, "q-pos", expected_score=0.9, catch=True),
        _summary_row(demo, "q-low", expected_score=0.5, catch=False, cluster_hit=False),
        _summary_row(demo, "q-neg", label=False, expected_score=None, impostor_score=0.99, catch=True),
    ]
    summary = demo._summarize(rows, tau=0.6, encrypted=True)

    assert summary["negatives"] == 1
    assert summary["over_reports"] == 1
    assert summary["over_report_rate"] == pytest.approx(1.0)
    # 负例不进召回：分子 1、分母 1 都只来自正例。
    assert summary["should_catch_positives"] == 1
    assert summary["encrypted_catches_positives"] == 1
    # 负例的真值集合为空，expected_score 必然缺失 —— 不能因此被当成"真匹配没载入"。
    assert summary["positives_missing_true_match_in_db"] == 0


def test_summary_reports_positives_missing_true_match_in_db(demo):
    """真匹配没载入库里是数据问题，不是匹配失败，绝不能静默混进分母。"""

    summary = demo._summarize(
        [_summary_row(demo, "q-orphan", expected_score=None, catch=False, cluster_hit=False)],
        tau=0.6,
        encrypted=True,
    )

    assert summary["positives_missing_true_match_in_db"] == 1
    assert summary["positives_missing_true_match_in_db_ids"] == ["q-orphan"]
    # None 不是 False：它既不在分母里，也不算"低于阈值"。
    assert summary["should_catch_positives"] == 0
    assert summary["positives_below_tau"] == 0
    assert summary["identical_queries"] == 0
    assert summary["recall_should_catch"] is None


def test_summary_works_without_encrypted_protocol(demo):
    """--no-encrypted 下明文量照常报，加密量整体缺席而不是报 0。"""

    summary = demo._summarize(_rows_for_recall(demo), tau=0.6, encrypted=False)

    assert summary["should_catch_positives"] == 3
    assert summary["positives_below_tau_ids"] == ["q-low"]
    assert summary["identical_queries"] == 1
    assert "recall_should_catch" not in summary
    assert "cluster_recall" not in summary


def test_expected_score_is_the_true_match_score_not_the_argmax(demo):
    """``expected_score`` 必须查真值 id，不能退化成 argmax 的分。

    两者在实测数据上恰好是同一条记录，所以这个 bug 在真实 run 里看不出来；一旦
    argmax 落到冒充者上，旧写法会把冒充者分当成真匹配分，分母从此是错的。

    这里用 ``plaintext_similarity``（encoder.py 里独立的代码路径）当 oracle。
    """

    schema = AttributeSchema.from_dict(
        {
            "attributes": [
                {"name": "code", "kind": "exact", "weight": 0.1, "blocks": 1,
                 "buckets_per_block": 16, "normalize": "casefold"},
                {"name": "name", "kind": "fuzzy_text", "weight": 0.9,
                 "cluster_dim": 40, "match_dim": 16},
            ]
        }
    )
    database = [
        # 真匹配：code 对上，name 差得远。
        AttributeRecord({"code": "AAA", "name": "robert smith"}, record_id="e-1"),
        # 冒充者：code 对不上，name 逐字相同 —— 总分压过真匹配。
        AttributeRecord({"code": "ZZZ", "name": "william jones"}, record_id="e-2"),
    ]
    query = AttributeQuery(
        record=AttributeRecord({"code": "AAA", "name": "william jones"}, record_id="q-1"),
        label=True,
        expected_record_ids=frozenset({"e-1"}),
    )

    score_true = plaintext_similarity(query.record, database[0], schema)
    score_impostor = plaintext_similarity(query.record, database[1], schema)
    assert score_impostor > score_true, (
        "测试构造失效：冒充者没有压过真匹配，这条测试就退化成恒真了"
    )

    db_matrix = encode_record_vectors(database, schema)[1]
    db_index = {record.record_id: i for i, record in enumerate(database)}
    result = demo._best_plaintext_match(database, db_matrix, query, schema, db_index)

    assert result.top1_record_id == "e-2"
    assert result.top1_score == pytest.approx(score_impostor)
    assert result.expected_score == pytest.approx(score_true)
    assert result.expected_score < result.top1_score
    # 冒充者分是"非真值记录里的最高分"，这里就是 argmax。
    assert result.impostor_score == pytest.approx(score_impostor)

    # 真匹配恰好就是 argmax 时两个量重合 —— 说明它们各自都算对了，而不是恒不相等。
    same = AttributeQuery(
        record=AttributeRecord({"code": "AAA", "name": "robert smith"}, record_id="q-2"),
        label=True,
        expected_record_ids=frozenset({"e-1"}),
    )
    hit = demo._best_plaintext_match(database, db_matrix, same, schema, db_index)
    assert hit.top1_record_id == "e-1"
    assert hit.expected_score == pytest.approx(hit.top1_score)


def test_expected_score_uses_the_best_of_multiple_true_matches(demo):
    """真值集合可能是多条（CSV 里 "|" 分隔），取其中最高的那条。"""

    schema = AttributeSchema.from_dict(
        {
            "attributes": [
                {"name": "code", "kind": "exact", "weight": 1.0, "blocks": 1,
                 "buckets_per_block": 16, "normalize": "casefold"},
            ]
        }
    )
    database = [
        AttributeRecord({"code": "AAA"}, record_id="e-1"),
        AttributeRecord({"code": "AAA"}, record_id="e-2"),
        AttributeRecord({"code": "ZZZ"}, record_id="e-3"),
    ]
    query = AttributeQuery(
        record=AttributeRecord({"code": "AAA"}, record_id="q-1"),
        label=True,
        expected_record_ids=frozenset({"e-1", "e-3"}),
    )
    db_matrix = encode_record_vectors(database, schema)[1]
    db_index = {record.record_id: i for i, record in enumerate(database)}

    result = demo._best_plaintext_match(database, db_matrix, query, schema, db_index)
    assert result.expected_score == pytest.approx(1.0)
    assert result.expected_score > 0.0, "取到 e-3 而不是 e-1 就会是 0 分"


def test_expected_score_is_none_when_true_match_is_absent(demo):
    """真值 id 不在库里 ⇒ None。返回 0.0 会把"没载入"伪装成"匹配失败"。"""

    schema = AttributeSchema.from_dict(
        {
            "attributes": [
                {"name": "code", "kind": "exact", "weight": 1.0, "blocks": 1,
                 "buckets_per_block": 16, "normalize": "casefold"},
            ]
        }
    )
    database = [AttributeRecord({"code": "AAA"}, record_id="e-1")]
    query = AttributeQuery(
        record=AttributeRecord({"code": "AAA"}, record_id="q-1"),
        label=True,
        expected_record_ids=frozenset({"e-missing"}),
    )
    db_matrix = encode_record_vectors(database, schema)[1]

    result = demo._best_plaintext_match(
        database, db_matrix, query, schema, {"e-1": 0}
    )
    assert result.expected_score is None
    # 没给 db_index 时同样拿不到真匹配分，不能退化成用 argmax 顶上。
    assert demo._best_plaintext_match(
        database, db_matrix, query, schema
    ).expected_score is None
