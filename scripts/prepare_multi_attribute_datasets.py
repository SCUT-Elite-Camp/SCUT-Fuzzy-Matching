#!/usr/bin/env python3
"""把仓库里已有的两套多属性数据整理成统一的 A/B/ground-truth 三件套。

为什么不用 ``recordlinkage.load_febrl3``
--------------------------------------
仓库自己的多属性管线（``baseline_Integration/multi_attribute/``）吃的不是"裸位串"，
而是 **job 文件**（schema + attribute_columns + database/queries 路径）。它已经自带
FEBRL 的读取分支（``link_mode: "febrl"``，真值来自 ``rec-<N>-org`` 与
``rec-<N>-dup-<M>`` 同实体），并用 ``scripts/fetch_dataset.py`` 从 jsdelivr 拉
dataset4a/4b。再引一份 recordlinkage 的 febrl3 只会得到第三套互不兼容的字段，
故本脚本走仓库原生数据，**不引入 recordlinkage**。

三个来源
--------
1. FEBRL 4a / 4b（真实脏数据）：库 5000 条、查询 5000 条，属性
   ``given_name, surname, street_number, address_1, address_2, suburb, postcode,
   state, date_of_birth, soc_sec_id`` —— 6 个 schema 属性（name/address/dob/ssn/
   postcode/state）。
2. fused_multilingual（SAGE 真实姓名 + 合成属性）：库 300 条、查询 50 条
   （40 正 / 10 负），属性 ``name_local, name_romanized, dob, city, phone``。
3. NCVR 10K：库里只有 ``ncid, full_name`` 两列 —— **单属性**，做不了多属性实验，
   本脚本只记录证据、不生成多属性数据。

输出（``data/multi_attr/``）
----------------------------
``febrl4_A.csv`` / ``febrl4_B.csv`` / ``febrl4_ground_truth.csv``
``multiling_A.csv`` / ``multiling_B.csv`` / ``multiling_ground_truth.csv``

ground truth 统一两列 ``A_id, B_id``（多语言额外带 ``label``），一行一条真匹配。

用法::

    cd ~/projects/SCUT-Fuzzy-Matching
    python3 scripts/fetch_dataset.py --dataset febrl        # 若 dataset/ 还没下
    python3 scripts/prepare_multi_attribute_datasets.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BI = ROOT / "baseline_Integration"
OUT = ROOT / "data" / "multi_attr"

FEBRL_DIR = BI / "dataset" / "febrl"
FUSED_DIR = BI / "data" / "fused_multilingual"
NCVR_DIR = BI / "data" / "ncvr_10k"

# FEBRL rec_id 的实体号：rec-4096-org / rec-4096-dup-0 -> 4096
_ENTITY = re.compile(r"^rec-(\d+)-")


def _read_csv(path: Path) -> list[dict]:
    """读 CSV 并**去掉字段名与值两侧空白**。

    FEBRL 原始 CSV 的表头写成了 ``rec_id, given_name, surname, ...``（逗号后有空格），
    直接 DictReader 会得到 ``" given_name"`` 这种键，下游 ``row["given_name"]`` 全部
    取空 —— 这个坑会让两个不同的记录编码成同一个空串，必须在这里清掉。
    """

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw in reader:
            rows.append(
                {
                    (k or "").strip(): (v.strip() if isinstance(v, str) else v)
                    for k, v in raw.items()
                }
            )
        return rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {path.relative_to(ROOT)}  ({len(rows)} rows, {path.stat().st_size / 1024:.0f} KB)")


def build_febrl() -> bool:
    """FEBRL 4a/4b -> A/B/ground truth。真值按 rec_id 里的实体号对齐。"""

    db_path = FEBRL_DIR / "dataset4a.csv"
    query_path = FEBRL_DIR / "dataset4b.csv"
    if not db_path.exists() or not query_path.exists():
        print("[SKIP] FEBRL 4a/4b 未下载；先跑 scripts/fetch_dataset.py --dataset febrl")
        return False

    database = _read_csv(db_path)
    queries = _read_csv(query_path)
    fields = list(database[0].keys())

    entity_of: dict[str, str] = {}
    for row in database:
        match = _ENTITY.match(row["rec_id"])
        if match:
            entity_of[row["rec_id"]] = match.group(1)

    # 一条查询可能对应多条库记录（4b 里同一实体的多条 dup），这里全部展开。
    truth = []
    for row in queries:
        match = _ENTITY.match(row["rec_id"])
        if not match:
            continue
        entity = match.group(1)
        for db_id, db_entity in entity_of.items():
            if db_entity == entity:
                truth.append({"A_id": db_id, "B_id": row["rec_id"]})

    # 统一 id 列名：A 侧用 record_id，B 侧用 query_id，与多语言那套保持一致。
    a_rows = [{"record_id": r["rec_id"], **{k: r[k] for k in fields if k != "rec_id"}} for r in database]
    b_rows = [{"query_id": r["rec_id"], **{k: r[k] for k in fields if k != "rec_id"}} for r in queries]

    print("[OK] FEBRL 4a/4b")
    _write_csv(OUT / "febrl4_A.csv", a_rows, list(a_rows[0].keys()))
    _write_csv(OUT / "febrl4_B.csv", b_rows, list(b_rows[0].keys()))
    _write_csv(OUT / "febrl4_ground_truth.csv", truth, ["A_id", "B_id"])

    per_query = Counter(r["B_id"] for r in truth)
    print(
        f"  A={len(a_rows)} B={len(b_rows)} truth={len(truth)} "
        f"每查询真匹配数 min/max={min(per_query.values())}/{max(per_query.values())}"
    )
    return True


def build_multilingual() -> bool:
    """fused_multilingual -> A(库)/B(查询)/ground truth（含 10 条负例）。"""

    db_path = FUSED_DIR / "database.csv"
    query_path = FUSED_DIR / "queries.csv"
    label_path = FUSED_DIR / "labels.csv"
    if not all(p.exists() for p in (db_path, query_path, label_path)):
        print("[SKIP] fused_multilingual 不全；应有 database/queries/labels 三个 CSV")
        return False

    database = _read_csv(db_path)
    queries = _read_csv(query_path)
    labels = _read_csv(label_path)

    label_of = {r["query_id"]: (r["true_entity_id"], str(r.get("label", "true")).strip().lower()) for r in labels}

    truth = []
    for query_id, (entity, label) in label_of.items():
        if label in ("true", "1", "yes"):
            truth.append({"A_id": entity, "B_id": query_id, "label": "true"})
        else:
            # 负例也写进去，label=false 时 A_id 记为空 —— 训练/评测时按 label 过滤。
            truth.append({"A_id": "", "B_id": query_id, "label": label})

    a_rows = [{"record_id": r["record_id"], **{k: v for k, v in r.items() if k != "record_id"}} for r in database]
    b_rows = [{"query_id": r["query_id"], **{k: v for k, v in r.items() if k != "query_id"}} for r in queries]

    print("[OK] fused_multilingual")
    _write_csv(OUT / "multiling_A.csv", a_rows, list(a_rows[0].keys()))
    _write_csv(OUT / "multiling_B.csv", b_rows, list(b_rows[0].keys()))
    _write_csv(OUT / "multiling_ground_truth.csv", truth, ["A_id", "B_id", "label"])

    counts = Counter(r["label"] for r in truth)
    print(f"  A={len(a_rows)} B={len(b_rows)} truth={len(truth)} 正/负={counts.get('true', 0)}/{counts.get('false', 0)}")
    return True


def report_ncvr() -> None:
    """NCVR 10K 的证据：只有姓名列，无法做多属性。"""

    db_path = NCVR_DIR / "ncvr_10k_database.csv"
    query_path = NCVR_DIR / "ncvr_10k_queries.csv"
    if not db_path.exists():
        print("[SKIP] NCVR 10K 不存在")
        return
    with db_path.open(encoding="utf-8") as handle:
        header = handle.readline().strip()
    print(f"[N/A] NCVR 10K 只有列 {header} —— 无 dob/address/postcode，单属性，不进多属性表")
    _ = query_path


def write_manifest(febrl_ok: bool, multi_ok: bool) -> None:
    manifest = {
        "generator": "scripts/prepare_multi_attribute_datasets.py",
        "seed": "n/a（FEBRL / fused 均为仓库既有数据，未重新采样）",
        "datasets": {
            "febrl4": {
                "available": febrl_ok,
                "source": "baseline_Integration/dataset/febrl/dataset4a.csv + dataset4b.csv",
                "attributes": ["name(given_name,surname)", "address(street_number,address_1,address_2)",
                               "dob(date_of_birth)", "ssn(soc_sec_id)", "postcode", "state"],
                "ground_truth": "rec-<N>-org 与 rec-<N>-dup-<M> 同实体",
            },
            "multiling": {
                "available": multi_ok,
                "source": "baseline_Integration/data/fused_multilingual/{database,queries,labels}.csv",
                "attributes": ["name(name_local,name_romanized)", "dob", "city", "phone"],
                "ground_truth": "labels.csv 的 true_entity_id（40 正 / 10 负）",
            },
            "ncvr10k": {
                "available": False,
                "reason": "ncvr_10k_database.csv 只有 ncid,full_name 两列，是单属性数据",
            },
        },
    }
    path = OUT / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  -> {path.relative_to(ROOT)}")


def main() -> int:
    global OUT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT), help="output dir (default data/multi_attr)")
    args = parser.parse_args()

    OUT = Path(args.out).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {OUT}")

    febrl_ok = build_febrl()
    multi_ok = build_multilingual()
    report_ncvr()
    write_manifest(febrl_ok, multi_ok)

    if not (febrl_ok or multi_ok):
        print("[FAIL] 一个数据集都没生成")
        return 1
    print("[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
