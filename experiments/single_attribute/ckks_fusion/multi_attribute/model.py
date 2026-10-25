from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .schema import AttributeSchema


@dataclass(frozen=True)
class MatchRecord:
    """One entity record used by the first multi-attribute prototype.

    ``dob`` may be omitted. Missing attributes contribute zero score rather than
    being encoded as a literal string such as ``"null"``.

    这是 V1 的固定两属性记录类型，**字段与顺序保持不变**，因此
    ``MatchRecord("John Smith", "2001-05-17")`` 这类位置参数调用继续有效。
    属性集合可变的场景请使用 :class:`AttributeRecord`。
    """

    name: str
    dob: str | None = None
    record_id: str | None = None


@dataclass(frozen=True)
class AttributeRecord:
    """任意属性集合的记录，配合 ``AttributeSchema`` 使用。

    ``values`` 里没有出现的属性名视为缺失，编码为全零块（贡献 0 分）。
    """

    values: Mapping[str, Any] = field(default_factory=dict)
    record_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", dict(self.values or {}))

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)

    @classmethod
    def from_match_record(
        cls,
        record: MatchRecord,
        *,
        name_attribute: str = "name",
        dob_attribute: str = "dob",
    ) -> "AttributeRecord":
        values: dict[str, Any] = {name_attribute: record.name}
        if record.dob is not None:
            values[dob_attribute] = record.dob
        return cls(values=values, record_id=record.record_id)


def record_values(record: Any, schema: AttributeSchema) -> dict[str, Any]:
    """把一条记录强制转换成 ``{属性名: 取值}``，只保留 schema 声明的属性。

    转换优先级：

    1. :class:`AttributeRecord` -> 直接取 ``values``。
    2. ``Mapping`` -> 直接当字典用。
    3. 其它对象 -> 逐属性 ``getattr(record, spec.name, None)``。
       另外做一层 **legacy 桥接**：对象只要有 ``name`` 字段，就把它填到
       ``schema.name_attribute`` 上。这样 ``MatchRecord("John Smith", ...)``
       不用知道 schema 的存在也能被正确编码。

    schema 里没有声明的键会被忽略 —— schema 是布局的唯一权威，而 CSV 行天然
    带一堆用不上的列。
    """

    if isinstance(record, AttributeRecord):
        source: Mapping[str, Any] = record.values
    elif isinstance(record, Mapping):
        source = record
    else:
        source = {}
        for spec in schema.attributes:
            value = getattr(record, spec.name, None)
            if value is not None:
                source[spec.name] = value
        if schema.name_attribute and schema.name_attribute not in source:
            legacy_name = getattr(record, "name", None)
            if legacy_name is not None:
                source = {**source, schema.name_attribute: legacy_name}

    return {spec.name: source.get(spec.name) for spec in schema.attributes}


__all__ = ["AttributeRecord", "MatchRecord", "record_values"]
