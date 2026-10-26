"""从融合数据集（data/fused_multilingual/）派生面向现场演示的 10 条查询子集。

产出 ``data/fused_multilingual/demo10/queries.csv`` + ``labels.csv``（7 正 3 负），
数据库直接复用上级目录的 300 条（job 配置里以仓库根目录为基准写
``data/fused_multilingual/database.csv`` —— ``resolve_path`` 相对路径一律按
``baseline_Integration/`` 解析，**不是**按 job 文件所在目录）。

选样原则（与 ``demo10/README`` 及 job ``_comment`` 保持一致）：

* 文字系统多样性：拉丁重音 / 去重音罗马化 / 汉字 / 汉罗混合 / 阿拉伯 / 缅甸
  各覆盖若干；3 条负例分属不同文字系统。
* 扰动多样性：姓名 typo、dob 留空、phone 留空、dob 年份对调 / 格式变换、
  城市大小写等不同情况都有代表。
* 正例的 true 分与 tau=0.485 的关系有讲头：既有高分的"稳命中"，也有
  0.5–0.7 区间"靠多属性互补拉回来"的案例（姓名-only 到不了阈值，但
  dob/city/phone 把总分拉过线）。

清单是**显式人工策划**的（``SELECTED``），不是自动优化出来的 —— 每条都带
``expect`` 校验（label / 文字系统 / 姓名形式 / 扰动标签 / 分数区间），父数据集
若被重建导致某条不再满足预期，脚本直接报错退出，而不是悄悄换样。

负例说明：父集 50 条里的 10 条负例恰好**全是拉丁文字**（构建时从 SAGE 剩余
姓名池随机抽的，拉丁池最大），凑不出"3 条负例分属不同文字系统"。所以清单里的
阿拉伯 / 缅甸负例是**用构建脚本同款管线新造**的（``_generate_negative_rows``，
固定种子）：从 SAGE 姓名池取与库、与已有查询任何姓名形式都不重合的名字，属性
由同一个 ``_make_db_record`` 合成，并校验冒充者分明显低于 tau。

用法::

    python scripts/build_fused_multilingual_demo10.py --report   # 看全部 50 条统计
    python scripts/build_fused_multilingual_demo10.py            # 校验清单并生成 demo10/
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# classify_name 等定义在同级构建脚本里；直接跑（sys.path[0]=scripts/）与
# 被 tests 以 ``scripts.`` 包名 import 两种情况下都能拿到。
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from anyascii import anyascii  # noqa: E402

import numpy as np  # noqa: E402

from build_fused_multilingual_dataset import (  # noqa: E402
    _make_db_record,
    classify_name,
    load_sage_groups,
    split_forms,
)
from data_pipeline.normalizers import _fold_latin  # noqa: E402

from multi_attribute.dataset import (  # noqa: E402
    AttributeQuery,
    _project_row,
)
from multi_attribute.encoder import (  # noqa: E402
    attribute_similarities,
    encode_record_vectors,
)
from multi_attribute.jobfile import (  # noqa: E402
    load_data,
    load_job,
    resolve_path,
)
from multi_attribute.model import AttributeRecord  # noqa: E402

JOB_PATH = ROOT / "config" / "examples" / "fused_multilingual_multi_attribute.json"
OUT_DIR = ROOT / "data" / "fused_multilingual" / "demo10"

# 负例分数护栏：冒充者分必须明显低于 tau（父数据集 50 条负例的冒充者分在
# 0.318–0.452 之间），防止新造的负例本身就是个假正例。
NEGATIVE_IMPOSTOR_CEILING = 0.45

# 人工策划的演示清单：7 正例（取自父数据集 50 条）+ 3 负例。
# expect 字段在生成前逐条校验：score_band 对正例校验 expected_score（真匹配分）、
# 对负例校验 impostor_score（冒充者分），均为闭区间。why 是现场讲解词，也是
# README 的素材。负例里 "generated" 的条目不由父集选出，而是从 SAGE 姓名池
# 按构建脚本同款管线新造（父集的 10 条负例恰好全是拉丁文字，凑不出 3 个
# 不同文字系统 —— 见模块 docstring）。
SELECTED: list[dict] = [
    {
        "query_id": "q13",
        "why": "latin_plain 基线：dob 改格式 + 城市变大小写仍满分（1.000）命中",
        "expect": {"label": True, "script": "latin_plain", "carry": "local",
                   "perturbations": ["dob_format", "city_case"], "score_band": (0.95, 1.0)},
    },
    {
        "query_id": "q02",
        "why": "去重音拉丁查询：Árgyelán 型重音名的 romanized-only 形式，dob 留空",
        "expect": {"label": True, "script": "latin_accented", "carry": "romanized",
                   "perturbations": ["dob_missing"], "score_band": (0.70, 0.80)},
    },
    {
        "query_id": "q34",
        "why": "汉罗混合：汉字+拉丁混合名的本地形式，姓名 typo 后仍 0.84",
        "expect": {"label": True, "script": "han_mixed", "carry": "local",
                   "perturbations": ["name_typo"], "score_band": (0.80, 0.90)},
    },
    {
        "query_id": "q27",
        "why": "互补案例：汉字名只匹配 0.56，dob 留空，靠 city+phone 两个 exact 属性拉到阈值上",
        "expect": {"label": True, "script": "han", "carry": "local",
                   "perturbations": ["dob_missing"], "score_band": (0.55, 0.62)},
    },
    {
        "query_id": "q47",
        "why": "阿拉伯本地文字查询跨文字匹配罗马化库名（name=0.71），dob 年份对调 + 城市大小写",
        "expect": {"label": True, "script": "arabic", "carry": "local",
                   "perturbations": ["dob_transpose", "city_case"], "score_band": (0.60, 0.68)},
    },
    {
        "query_id": "q16",
        "why": "缅甸文 + 姓名 typo + dob 改格式：name 仅 0.75，其余属性补到 0.90",
        "expect": {"label": True, "script": "myanmar", "carry": "local",
                   "perturbations": ["name_typo", "dob_format"], "score_band": (0.85, 0.95)},
    },
    {
        "query_id": "q06",
        "why": "汉字名 typo + phone 留空：0.4*0.74+0.25+0.15 ≈ 0.69，多重扰动下的稳定命中",
        "expect": {"label": True, "script": "han", "carry": "local",
                   "perturbations": ["name_typo", "phone_missing"], "score_band": (0.65, 0.75)},
    },
    {
        "query_id": "q15",
        "why": "拉丁负例：库里不存在的不相干重音名记录",
        "expect": {"label": False, "script": "latin_accented", "score_band": (0.30, 0.40)},
    },
    {
        "query_id": "neg-arabic-01",
        "generated": "arabic",
        "why": "阿拉伯文负例：与库完全不搭的伊拉克姓名",
        "expect": {"label": False, "script": "arabic", "score_band": (0.0, 0.45)},
    },
    {
        "query_id": "neg-myanmar-01",
        "generated": "myanmar",
        "why": "缅甸文负例：与库完全不搭的缅甸姓名",
        "expect": {"label": False, "script": "myanmar", "score_band": (0.0, 0.45)},
    },
]


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        reader.fieldnames = [name.strip() for name in reader.fieldnames or []]
        return [dict(row) for row in reader]


def _entity_script(entity_row: dict[str, str]) -> str:
    """按构建脚本的口径给**库记录**分文字系统组。

    纯汉字行的 romanized 是 ``anyascii(本地形式)`` 的机器转写，而汉罗混合行的
    romanized 是源数据里的真实拉丁部分 —— 两者都使 ``local+romanized`` 呈
    "Han+Latn"，所以只靠 classify_name 分不出来，用 anyascii 相等性区分。
    """

    local = (entity_row.get("name_local") or "").strip()
    romanized = (entity_row.get("name_romanized") or "").strip()
    script = classify_name(local or romanized)
    if script == "han" and romanized and _fold_latin(anyascii(local)) != romanized:
        return "han_mixed"
    return script


def _detect_perturbations(query_row: dict[str, str], entity_row: dict[str, str]) -> list[str]:
    """比对查询行与其真实体行，反推应用了哪些扰动（生成器不记录扰动类型）。"""

    tags: list[str] = []
    name_col = "name_local" if (query_row.get("name_local") or "").strip() else "name_romanized"
    if (query_row.get(name_col) or "") != (entity_row.get(name_col) or ""):
        tags.append("name_typo")

    q_dob = (query_row.get("dob") or "").strip()
    e_dob = (entity_row.get("dob") or "").strip()
    if not q_dob and e_dob:
        tags.append("dob_missing")
    elif q_dob and q_dob != e_dob:
        tags.append("dob_format" if ("/" in q_dob or "." in q_dob) else "dob_transpose")

    if not (query_row.get("phone") or "").strip() and (entity_row.get("phone") or "").strip():
        tags.append("phone_missing")

    q_city = (query_row.get("city") or "").strip()
    e_city = (entity_row.get("city") or "").strip()
    if q_city and q_city != e_city and q_city.casefold() == e_city.casefold():
        tags.append("city_case")
    return tags


def _score_against_db(record, schema, db_matrix, database, db_index, true_ids) -> dict:
    """一条记录对全库的点积分：expected / impostor / top-1（与 demo 的明文参考同口径）。"""

    vector = encode_record_vectors([record], schema)[1][0]
    scores = db_matrix @ vector
    order = [int(i) for i in np.argsort(scores)[::-1]]
    expected = (
        max((float(scores[db_index[t]]) for t in true_ids if t in db_index), default=None)
        if true_ids
        else None
    )
    impostor = next(
        (
            float(scores[i])
            for i in order
            if not true_ids or database[i].record_id not in true_ids
        ),
        None,
    )
    return {
        "expected_score": expected,
        "impostor_score": impostor,
        "top1_id": database[order[0]].record_id,
    }


def _query_stats(query, row, db_csv, database, db_index, schema, db_matrix) -> dict:
    true_ids = sorted(query.expected_record_ids)
    stats = {
        "label": bool(query.label),
        "carry": "both",
        "perturbations": [],
        "attr_vs_true": {},
        "name": (row.get("name_local") or "").strip() or (row.get("name_romanized") or "").strip(),
    }
    stats.update(_score_against_db(query.record, schema, db_matrix, database, db_index, true_ids))
    if true_ids:
        entity = db_csv[true_ids[0]]
        stats["script"] = _entity_script(entity)
        stats["carry"] = "local" if (row.get("name_local") or "").strip() else "romanized"
        stats["perturbations"] = _detect_perturbations(row, entity)
        stats["attr_vs_true"] = attribute_similarities(
            query.record, database[db_index[true_ids[0]]], schema
        )
    else:
        stats["script"] = _entity_script(row)
    return stats


def compute_stats(job, database, db_matrix, db_index, extra_rows: list[dict] | None = None) -> dict[str, dict]:
    """对全部 50 条查询（外加新造负例行）算明文统计。

    逐属性分是**对真实体**算的（不是对 top-1）——"多属性互补"的故事要看真匹配
    那条记录上各属性各贡献了多少。
    """

    schema = job["schema_object"]
    db_csv = {row["record_id"]: row for row in _read_csv(resolve_path(job["database"]))}
    q_csv = {row["query_id"]: row for row in _read_csv(resolve_path(job["queries"]))}
    _, queries = load_data(job, limit=None, database_limit=None)

    stats: dict[str, dict] = {}
    for query in queries:
        qid = query.record.record_id
        stats[qid] = _query_stats(query, q_csv[qid], db_csv, database, db_index, schema, db_matrix)

    for row in extra_rows or []:
        values = _project_row(row, job["attribute_columns"], path="<generated negative>")
        record = AttributeRecord(values=values, record_id=row["query_id"])
        query = AttributeQuery(record=record, label=False, expected_record_ids=frozenset())
        stats[row["query_id"]] = _query_stats(query, row, db_csv, database, db_index, schema, db_matrix)
    return stats


def _load_db(job: dict):
    """编码整个库一次，供 50 条统计与新造负例的逐个打分共用。"""

    database, _ = load_data(job, limit=None, database_limit=None)
    schema = job["schema_object"]
    db_matrix = encode_record_vectors(database, schema)[1]
    db_index = {record.record_id: i for i, record in enumerate(database)}
    return database, db_matrix, db_index


# ---------------------------------------------------------------------------
# 报告与生成
# ---------------------------------------------------------------------------


def _impostor_of_row(row: dict, job: dict, schema, db_matrix, database, db_index) -> float:
    values = _project_row(row, job["attribute_columns"], path="<generated negative>")
    record = AttributeRecord(values=values, record_id=row["query_id"])
    return _score_against_db(record, schema, db_matrix, database, db_index, ())["impostor_score"]


def _generate_negative_rows(job, schema, db_matrix, database, db_index) -> list[dict]:
    """按构建脚本同款管线，从 SAGE 姓名池为 SELECTED 里 ``generated`` 的负例新造查询行。

    父集的 10 条负例恰好全是拉丁文字，凑不出"3 条负例分属不同文字系统"。新造行
    与父集负例同构（姓名两列都带、属性由同一个 ``_make_db_record`` 合成），
    固定种子逐字节可复现。候选姓名必须：与库/已有查询的任何姓名形式（去重音+
    转写+小写归一后）不重合；逐个算冒充者分，**取第一个低于安全线**（留噪声余量，
    别贴着脸蹭 tau）的候选；一个都没有才退而取最低分者，交给 ``_validate`` 兜底。
    """

    needed = [e for e in SELECTED if e.get("generated")]
    if not needed:
        return []
    rng = random.Random(20261014)  # 与父集同种子，独立 RNG 流，互不干扰
    groups = load_sage_groups()
    taken: set[str] = set()
    for row in [
        *_read_csv(resolve_path(job["database"])),
        *_read_csv(resolve_path(job["queries"])),
    ]:
        for col in ("name_local", "name_romanized"):
            value = (row.get(col) or "").strip()
            if value:
                taken.add(_fold_latin(anyascii(value)).casefold())

    rows = []
    ordinal = 900  # 只用于内部 record_id 生成，查询行不带 record_id
    for entry in needed:
        group = entry["generated"]
        best: tuple[float, dict] | None = None
        for name, country in groups[group]:
            local, romanized = split_forms(name, group)
            forms = {
                _fold_latin(anyascii(value)).casefold()
                for value in (local, romanized)
                if value.strip()
            }
            if forms & taken:
                continue
            record = _make_db_record(rng, name, country, group, ordinal)
            ordinal += 1
            candidate = {
                "query_id": entry["query_id"],
                "name_local": record["name_local"],
                "name_romanized": record["name_romanized"],
                "dob": record["dob"],
                "city": record["city"],
                "phone": record["phone"],
                "country": record["country"],
            }
            impostor = _impostor_of_row(candidate, job, schema, db_matrix, database, db_index)
            if best is None or impostor < best[0]:
                best = (impostor, candidate)
            if impostor <= NEGATIVE_IMPOSTOR_CEILING - 0.02:  # 安全线，留噪声余量
                break
        if best is None:
            raise SystemExit(f"no unused SAGE name left for generated {group!r} negative")
        taken.add(
            _fold_latin(anyascii(best[1]["name_local"] or best[1]["name_romanized"])).casefold()
        )
        print(f"  generated {best[1]['query_id']}: impostor={best[0]:.3f} name={best[1]['name_local'][:24]!r}")
        rows.append(best[1])
    return rows


def report(stats: dict[str, dict]) -> None:
    print(f"{'qid':<5}{'label':<7}{'script':<15}{'carry':<10}{'exp_score':>10} {'impostor':>9}  perturbations / attr vs true")
    print("-" * 110)
    for qid, s in stats.items():
        exp = f"{s['expected_score']:.3f}" if s['expected_score'] is not None else "-"
        imp = f"{s['impostor_score']:.3f}" if s['impostor_score'] is not None else "-"
        attr = " ".join(f"{k}={v:.2f}" for k, v in s["attr_vs_true"].items())
        pert = "+".join(s["perturbations"]) if s["perturbations"] else "-"
        print(
            f"{qid:<5}{str(s['label']):<7}{s['script']:<15}{s['carry']:<10}{exp:>10} {imp:>9}  "
            f"{pert}   {attr}"
        )


def _validate(stats: dict[str, dict]) -> None:
    if len(SELECTED) != 10:
        raise SystemExit(f"SELECTED must contain exactly 10 entries, has {len(SELECTED)}")
    positives = [e for e in SELECTED if e["expect"]["label"]]
    negatives = [e for e in SELECTED if not e["expect"]["label"]]
    if len(positives) != 7 or len(negatives) != 3:
        raise SystemExit(f"need 7 positives + 3 negatives, got {len(positives)}/{len(negatives)}")
    scripts = {stats[e["query_id"]]["script"] for e in SELECTED if not e["expect"]["label"]}
    if len(scripts) != 3:
        raise SystemExit(f"negatives must span 3 script systems, got {sorted(scripts)}")

    problems: list[str] = []
    for entry in SELECTED:
        qid, expect = entry["query_id"], entry["expect"]
        if qid not in stats:
            problems.append(f"{qid}: not in dataset")
            continue
        actual = stats[qid]
        if actual["label"] != expect["label"]:
            problems.append(f"{qid}: label {actual['label']} != expected {expect['label']}")
        if actual["script"] != expect["script"]:
            problems.append(f"{qid}: script {actual['script']} != expected {expect['script']}")
        if entry.get("generated") and entry["generated"] != expect["script"]:
            problems.append(f"{qid}: generated from {entry['generated']} but expect {expect['script']}")
        if "carry" in expect and actual["carry"] != expect["carry"]:
            problems.append(f"{qid}: carry {actual['carry']} != expected {expect['carry']}")
        missing = set(expect.get("perturbations", [])) - set(actual["perturbations"])
        if missing:
            problems.append(f"{qid}: missing perturbations {sorted(missing)} (has {actual['perturbations']})")
        # 正例看真匹配分，负例看冒充者分（库里最高分）——都必须落在宣称的区间。
        score = actual["expected_score"] if expect["label"] else actual["impostor_score"]
        if score is None or not (expect["score_band"][0] <= score <= expect["score_band"][1]):
            field = "expected_score" if expect["label"] else "impostor_score"
            problems.append(f"{qid}: {field} {score} outside {expect['score_band']}")
    if problems:
        raise SystemExit("selection no longer matches the dataset:\n  " + "\n  ".join(problems))


def write_subset(job: dict, stats: dict[str, dict], generated: list[dict]) -> None:
    _validate(stats)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    queries = _read_csv(resolve_path(job["queries"]))
    labels = _read_csv(resolve_path(job["labels"]))
    q_fields = list(queries[0])
    l_fields = list(labels[0])
    parent_q = {row["query_id"]: row for row in queries}
    parent_l = {row["query_id"]: row for row in labels}
    extra_q = {row["query_id"]: row for row in generated}

    # 按 SELECTED 的顺序写 —— 清单顺序就是现场讲解顺序。
    q_out: list[dict] = []
    l_out: list[dict] = []
    for entry in SELECTED:
        qid = entry["query_id"]
        if qid in parent_q:
            q_out.append(parent_q[qid])
            l_out.append(parent_l[qid])
        elif qid in extra_q:
            q_out.append({key: extra_q[qid][key] for key in q_fields})
            l_out.append({key: (qid if key == "query_id" else "") for key in l_fields})
        else:
            raise SystemExit(f"{qid}: neither in parent queries nor generated")

    with (OUT_DIR / "queries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=q_fields)
        writer.writeheader()
        writer.writerows(q_out)
    with (OUT_DIR / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=l_fields)
        writer.writeheader()
        writer.writerows(l_out)

    print(f"wrote {OUT_DIR}/queries.csv + labels.csv ({len(q_out)} queries)")
    for entry in SELECTED:
        s = stats[entry["query_id"]]
        exp = f"{s['expected_score']:.3f}" if s['expected_score'] is not None else "-"
        imp = f"{s['impostor_score']:.3f}" if s['impostor_score'] is not None else "-"
        print(
            f"  {entry['query_id']:<16}{str(s['label']):<7}{s['script']:<15}{s['carry']:<10}"
            f"true={exp:<6}imp={imp:<6}{'+'.join(s['perturbations']) or '-':<28}# {entry['why']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="print stats for all queries and exit")
    args = parser.parse_args()

    job = load_job(JOB_PATH)
    if args.report:
        database, db_matrix, db_index = _load_db(job)
        report(compute_stats(job, database, db_matrix, db_index))
        return 0
    database, db_matrix, db_index = _load_db(job)
    generated = _generate_negative_rows(job, job["schema_object"], db_matrix, database, db_index)
    stats = compute_stats(job, database, db_matrix, db_index, generated)
    write_subset(job, stats, generated)
    return 0


if __name__ == "__main__":
    sys.exit(main())
