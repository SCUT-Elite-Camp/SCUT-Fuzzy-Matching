"""把 CSV 数据集读成 ``AttributeRecord``，喂给多属性协议。

刻意不去改 ``data_pipeline``：那套契约（``NameRecord``/``PreparedName``）是纯姓名
导向的，为它加属性字段会波及 adapters / builder / io / manifest 四处却零收益。
``MULTI_ATTRIBUTE_V1.md`` 开篇也写明多属性模块不改变现有的 name-only 生产路径。
所以属性数据在这里直接读 CSV。

用法::

    records_b, queries = load_dataset_pair(
        "dataset/febrl/dataset4a.csv",
        "dataset/febrl/dataset4b.csv",
        attribute_columns={
            "name": ["given_name", "surname"],
            "dob": "date_of_birth",
            "address": ["street_number", "address_1", "address_2"],
        },
        link_mode=True,
    )
"""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model import AttributeRecord

# FEBRL 的 rec_id 形如 ``rec-1070-org`` / ``rec-1070-dup-0``，实体号即真值。
_FEBRL_ID = re.compile(r"^rec-(?P<entity>\d+)-(?P<role>org|dup)(?:-\d+)?$")
_FEBRL_SEPARATOR = "-dup-"


@dataclass(frozen=True)
class AttributeQuery:
    """一条带标签的查询记录。"""

    record: AttributeRecord
    label: bool
    expected_record_ids: frozenset[str] = field(default_factory=frozenset)


def _read_csv(path: str | Path, encoding: str) -> list[dict[str, str]]:
    """读 CSV，并把表头与值的前导空格去掉。

    FEBRL 的表头就是 ``rec_id, given_name, surname, ...`` —— 每个列名前面都有
    一个空格。不处理的话 ``row["given_name"]`` 会查不到，报出很误导人的
    「unknown column」。
    """

    with Path(path).open(newline="", encoding=encoding) as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {path}")
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        return [dict(row) for row in reader]


def _project_row(
    row: Mapping[str, str],
    attribute_columns: Mapping[str, str | Sequence[str]],
    *,
    path: str | Path,
) -> dict[str, Any]:
    """按 ``{属性名: 列名或列名列表}`` 投影一行。

    多列以空格拼接并跳过空值；整行该属性为空时**不写入 dict**，于是编码阶段
    得到全零块（缺失属性零贡献）。未知列名直接报错，避免拼写错误被静默吞掉。
    """

    values: dict[str, Any] = {}
    for name, columns in attribute_columns.items():
        if isinstance(columns, str):
            column_list = [columns]
        else:
            column_list = list(columns)

        parts: list[str] = []
        for column in column_list:
            if column not in row:
                raise ValueError(f"{path}: unknown column {column!r} for attribute {name!r}")
            text = (row[column] or "").strip()
            if text:
                parts.append(text)
        if parts:
            values[name] = " ".join(parts)
    return values


def load_attribute_records(
    path: str | Path,
    attribute_columns: Mapping[str, str | Sequence[str]],
    *,
    id_column: str | None = None,
    encoding: str = "utf-8",
    limit: int | None = None,
    id_prefix: str | None = None,
) -> list[AttributeRecord]:
    """读一个 CSV，返回 ``AttributeRecord`` 列表。"""

    rows = _read_csv(path, encoding)
    if not rows:
        raise ValueError(f"CSV file has no data rows: {path}")

    records: list[AttributeRecord] = []
    for index, row in enumerate(rows):
        if limit is not None and len(records) >= limit:
            break
        values = _project_row(row, attribute_columns, path=path)
        if not values:
            continue
        if id_column:
            if id_column not in row:
                raise ValueError(f"{path}: unknown id column {id_column!r}")
            record_id = (row[id_column] or "").strip() or None
        else:
            record_id = f"{id_prefix or 'row'}-{index + 1}"
        records.append(AttributeRecord(values=values, record_id=record_id))
    return records


def load_attribute_queries(
    path: str | Path,
    attribute_columns: Mapping[str, str | Sequence[str]],
    *,
    id_column: str | None = None,
    label_column: str | None = None,
    expected_ids_column: str | None = None,
    expected_ids_separator: str = "|",
    encoding: str = "utf-8",
    limit: int | None = None,
) -> list[AttributeQuery]:
    """读一个带标签的查询 CSV。"""

    rows = _read_csv(path, encoding)
    if not rows:
        raise ValueError(f"CSV file has no data rows: {path}")

    queries: list[AttributeQuery] = []
    for index, row in enumerate(rows):
        if limit is not None and len(queries) >= limit:
            break
        values = _project_row(row, attribute_columns, path=path)
        if not values:
            continue

        expected: frozenset[str] = frozenset()
        if expected_ids_column:
            if expected_ids_column not in row:
                raise ValueError(f"{path}: unknown expected-ids column {expected_ids_column!r}")
            expected = frozenset(
                part.strip()
                for part in (row[expected_ids_column] or "").split(expected_ids_separator)
                if part.strip()
            )

        if label_column:
            if label_column not in row:
                raise ValueError(f"{path}: unknown label column {label_column!r}")
            label = (row[label_column] or "").strip().lower() in {"true", "1", "yes", "y"}
        else:
            label = bool(expected)
        if not label:
            expected = frozenset()

        record_id = (row[id_column] or "").strip() if id_column else f"query-{index + 1}"
        queries.append(
            AttributeQuery(
                record=AttributeRecord(values=values, record_id=record_id or None),
                label=label,
                expected_record_ids=expected,
            )
        )
    return queries


def apply_labels(
    queries: Sequence[AttributeQuery],
    path: str | Path,
    *,
    id_column: str = "query_id",
    label_column: str = "label",
    expected_column: str = "true_entity_id",
    expected_ids_separator: str = "|",
    encoding: str = "utf-8",
) -> list[AttributeQuery]:
    """把单独的 labels CSV 贴到已加载的查询上。

    生成器把真值写在独立的 ``labels.csv`` 里（``query_id,true_entity_id,label``），
    查询 CSV 本身不带标签。labels 里**没有出现的查询原样保留** —— 不猜、不默认。
    """

    truth = _read_csv(path, encoding)
    by_id: dict[str, tuple[bool, frozenset[str]]] = {}
    for row in truth:
        query_id = (row.get(id_column) or "").strip()
        if not query_id:
            continue
        label = (row.get(label_column) or "").strip().lower() in {"true", "1", "yes", "y"}
        raw = (row.get(expected_column) or "").strip()
        expected = (
            frozenset(part.strip() for part in raw.split(expected_ids_separator) if part.strip())
            if raw
            else frozenset()
        )
        by_id[query_id] = (label, expected if label else frozenset())

    out: list[AttributeQuery] = []
    for query in queries:
        entry = by_id.get(query.record.record_id or "")
        if entry is None:
            out.append(query)
        else:
            out.append(
                AttributeQuery(
                    record=query.record, label=entry[0], expected_record_ids=entry[1]
                )
            )
    return out


def febrl_entity_id(rec_id: str) -> str:
    """从 FEBRL 的 ``rec-<N>-org`` / ``rec-<N>-dup-<M>`` 里取出实体号。"""

    match = _FEBRL_ID.match(str(rec_id).strip())
    if not match:
        raise ValueError(
            f"malformed FEBRL rec_id {rec_id!r}; expected rec-<N>-org or rec-<N>-dup-<M>"
        )
    return match.group("entity")


def load_febrl_pair(
    originals_path: str | Path,
    duplicates_path: str | Path,
    attribute_columns: Mapping[str, str | Sequence[str]],
    *,
    id_column: str = "rec_id",
    encoding: str = "utf-8",
    limit: int | None = None,
    database_limit: int | None = None,
    drop_orphans: bool = True,
) -> tuple[list[AttributeRecord], list[AttributeQuery]]:
    """读 FEBRL 4a/4b 这一对，利用 ``rec_id`` 免费拿到真值配对。

    返回 ``(database, queries)``：数据库是原始记录（``-org``），查询是扰动副本
    （``-dup``），每条查询 ``label=True`` 且指向同实体的原始 ``record_id``。

    ``limit`` 只限制**查询**条数，``database_limit`` 才限制库规模 —— 两者必须分开：
    4a 的 ``rec_id`` 不是升序的，把 ``limit`` 同时套在库上会让绝大部分副本的原始
    记录落在库外。FEBRL 4a/4b 里**没有负例**，所有副本都是真匹配。

    Args:
        drop_orphans: 原始记录不在库内的副本直接丢弃。默认丢弃而不是标成负例 ——
            那是**假负例**（真匹配确实存在，只是没被载入），比丢掉更有害。
    """

    database = load_attribute_records(
        originals_path,
        attribute_columns,
        id_column=id_column,
        encoding=encoding,
        limit=database_limit,
    )

    by_entity: dict[str, str] = {}
    for record in database:
        entity = febrl_entity_id(record.record_id or "")
        by_entity.setdefault(entity, record.record_id or "")

    rows = _read_csv(duplicates_path, encoding)
    queries: list[AttributeQuery] = []
    for index, row in enumerate(rows):
        # 先过滤再计数，这样 limit 拿到的是 limit 条**可用**查询，而不是一堆孤儿。
        if limit is not None and len(queries) >= limit:
            break
        values = _project_row(row, attribute_columns, path=duplicates_path)
        if not values:
            continue
        rec_id = (row[id_column] or "").strip()
        entity = febrl_entity_id(rec_id)
        target = by_entity.get(entity)
        if target is None and drop_orphans:
            continue
        queries.append(
            AttributeQuery(
                record=AttributeRecord(values=values, record_id=rec_id or f"dup-{index + 1}"),
                label=target is not None,
                expected_record_ids=frozenset({target}) if target else frozenset(),
            )
        )
    return database, queries


def load_dataset_pair(
    database_path: str | Path,
    queries_path: str | Path,
    attribute_columns: Mapping[str, str | Sequence[str]],
    *,
    id_column: str | None = None,
    query_id_column: str | None = None,
    label_column: str | None = None,
    expected_ids_column: str | None = None,
    encoding: str = "utf-8",
    limit: int | None = None,
    database_limit: int | None = None,
) -> tuple[list[AttributeRecord], list[AttributeQuery]]:
    """通用的一对 CSV 加载（库 + 查询）。

    ``limit`` 限制查询条数，``database_limit`` 限制库规模（见 ``load_febrl_pair``）。
    """

    return (
        load_attribute_records(
            database_path,
            attribute_columns,
            id_column=id_column,
            encoding=encoding,
            limit=database_limit,
        ),
        load_attribute_queries(
            queries_path,
            attribute_columns,
            id_column=query_id_column,
            label_column=label_column,
            expected_ids_column=expected_ids_column,
            encoding=encoding,
            limit=limit,
        ),
    )


__all__ = [
    "AttributeQuery",
    "apply_labels",
    "febrl_entity_id",
    "load_attribute_queries",
    "load_attribute_records",
    "load_dataset_pair",
    "load_febrl_pair",
]
