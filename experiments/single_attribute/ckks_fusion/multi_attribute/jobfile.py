"""job 文件（schema + 数据映射）的读取与加载。

抽出来是因为它有两个消费者，而其中一个不该被拖上 HE 依赖：``scripts/demo_multi_
attribute_dataset.py`` 要走加密双轮，``scripts/calibrate_multi_attribute.py`` 只
算明文分数 —— 后者在一个没装 TenSEAL 的环境里也应当能跑完标定。

本模块只碰 CSV / JSON / schema，不 import tenseal。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dataset import (
    apply_labels,
    load_attribute_queries,
    load_attribute_records,
    load_febrl_pair,
)
from .schema import AttributeSchema

__all__ = ["ROOT", "load_data", "load_job", "resolve_path"]

# 相对路径一律按仓库根目录（baseline_Integration/）解析。
ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


def load_job(path: str | Path) -> dict:
    """读 job 文件。

    两种形态都认：

    * ``{"attributes": [...]}`` —— 裸 schema，列名默认与属性名同名；
    * ``{"schema": {...}, "attribute_columns": {...}, ...}`` —— 带数据映射的完整 job。

    下划线开头的键（``_comment`` 之类）一律当注释忽略。
    """

    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: job file must be a JSON object")

    if "attributes" in raw:
        job: dict[str, Any] = {"schema": raw, "attribute_columns": None}
    else:
        job = {k: v for k, v in raw.items() if not k.startswith("_")}
        if "schema" not in job:
            raise SystemExit(f"{path}: job file needs either 'attributes' or 'schema'")

    schema = AttributeSchema.from_dict(job["schema"])
    columns = job.get("attribute_columns") or {spec.name: spec.name for spec in schema.attributes}

    missing = [spec.name for spec in schema.attributes if spec.name not in columns]
    if missing:
        raise SystemExit(f"{path}: attribute_columns is missing entries for {missing}")

    job["schema_object"] = schema
    job["attribute_columns"] = columns
    return job


def load_data(job: dict, limit: int | None, database_limit: int | None):
    schema: AttributeSchema = job["schema_object"]
    columns = job["attribute_columns"]
    database_path = job.get("database")
    queries_path = job.get("queries")
    if not database_path or not queries_path:
        raise SystemExit("job file needs 'database' and 'queries' paths")

    database_path, queries_path = resolve_path(database_path), resolve_path(queries_path)

    if job.get("link_mode") == "febrl":
        database, queries = load_febrl_pair(
            database_path,
            queries_path,
            columns,
            id_column=job.get("id_column", "rec_id"),
            limit=limit,
            database_limit=database_limit,
        )
    else:
        database = load_attribute_records(
            database_path, columns, id_column=job.get("id_column"), limit=database_limit
        )
        queries = load_attribute_queries(
            queries_path,
            columns,
            id_column=job.get("query_id_column"),
            label_column=job.get("label_column"),
            expected_ids_column=job.get("expected_ids_column"),
            limit=limit,
        )
        if job.get("labels"):
            queries = apply_labels(queries, resolve_path(job["labels"]))

    if not database:
        raise SystemExit(f"no usable records loaded from {database_path}")
    if not queries:
        raise SystemExit(f"no usable queries loaded from {queries_path}")
    return database, queries
