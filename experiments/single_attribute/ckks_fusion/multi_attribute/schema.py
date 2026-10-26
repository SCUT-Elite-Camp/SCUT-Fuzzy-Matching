"""声明式属性 schema：描述「哪些属性、什么类型、多少权重、占多少维」。

这是把写死的 name+DOB 原型泛化为任意属性集合的核心。协议层不认识任何具体属性，
只认识 ``cluster_dim`` 和 ``match_dim`` 两个数字 —— 而这两个数字由本模块从属性
声明推导出来。

最终加密得分保持 V1 的语义::

    S = Σ wᵢ · simᵢ(attributeᵢ)

之所以成立：每个属性的块编码后都是 L2 归一化的，再乘以 ``sqrt(wᵢ)`` 后拼接，
CKKS 点积展开即得上述加权和。

JSON 表示用**列表**承载属性顺序（``{"attributes": [...]}``），因为布局顺序是载荷
的一部分，而 dict 的键序经 ``sort_keys=True`` 序列化后会丢失。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from config.params import (
    ATTRIBUTE_WEIGHT_SUM_TOLERANCE,
    CKKS_SLOT_LIMIT,
    MULTI_ATTRIBUTE_THRESHOLD,
    NAME_ATTRIBUTE,
)

from .registry import RegistryError, get_kind, registered_kinds

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# from_dict 接受的顶层键。多出来的键一律报错，防止拼写错误被静默忽略。
_SCHEMA_KEYS = frozenset(
    {"attributes", "similarity_threshold", "name_attribute", "standardize_cluster"}
)


class SchemaError(ValueError):
    """属性 schema 配置非法。"""


@dataclass(frozen=True, eq=True)
class AttributeSpec:
    """单个属性的声明。

    ``params`` 存放 kind 专属参数（例如 ``fuzzy_text`` 的 ``cluster_dim``、
    ``exact`` 的 ``buckets_per_block``）。放进 dict 而不是展开成可选字段，
    这样注册一个新的 kind 可以自由定义自己的参数名，不必修改本类。
    """

    name: str
    kind: str
    weight: float
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", MappingProxyType(dict(self.params or {})))
        # 别名（如 "dob" -> "date"）统一成规范名，保证 JSON 往返与指纹稳定。
        name = str(self.kind).strip().lower()
        try:
            canonical = get_kind(name).name
        except RegistryError as exc:
            raise SchemaError(str(exc)) from exc
        object.__setattr__(self, "kind", canonical)

    def __hash__(self) -> int:
        # params 是不可哈希的 dict，所以按规范化 JSON 取哈希。
        return hash(self.canonical_json())

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "weight": self.weight, **self.params}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttributeSpec":
        missing = [k for k in ("name", "kind", "weight") if k not in value]
        if missing:
            raise SchemaError(f"attribute entry is missing keys {missing}: {dict(value)}")
        params = {k: v for k, v in value.items() if k not in ("name", "kind", "weight")}
        return cls(
            name=str(value["name"]),
            kind=str(value["kind"]),
            weight=float(value["weight"]),
            params=params,
        )


@dataclass(frozen=True, eq=True)
class AttributeSchema:
    """一整套属性声明，决定向量布局与维度。"""

    attributes: Sequence[AttributeSpec | Mapping[str, Any]]
    similarity_threshold: float = MULTI_ATTRIBUTE_THRESHOLD
    name_attribute: str | None = None
    standardize_cluster: bool = True

    def __post_init__(self) -> None:
        specs: list[AttributeSpec] = []
        for item in self.attributes:
            if isinstance(item, AttributeSpec):
                specs.append(item)
            elif isinstance(item, Mapping):
                specs.append(AttributeSpec.from_dict(item))
            else:
                raise SchemaError(
                    f"attribute must be AttributeSpec or mapping, got {type(item)}"
                )
        object.__setattr__(self, "attributes", tuple(specs))

        # 未显式指定 name_attribute 时自动识别：只有当声明里真的有名为
        # NAME_ATTRIBUTE 的模糊文本属性时才启用它。否则留 None —— 不能因为一个
        # 恰好叫 "name" 的精确属性就把字符串名字硬塞进去。
        # 显式指定了却对不上，则在下方的 _validate 里报错（fail closed）。
        if self.name_attribute is None:
            for spec in specs:
                if spec.name == NAME_ATTRIBUTE and spec.kind == "fuzzy_text":
                    object.__setattr__(self, "name_attribute", NAME_ATTRIBUTE)
                    break

        self._validate()

    def __hash__(self) -> int:
        return hash(self.fingerprint())

    # -- 校验 ---------------------------------------------------------------

    def _validate(self) -> None:
        if not self.attributes:
            raise SchemaError("schema must declare at least one attribute")

        seen: set[str] = set()
        for spec in self.attributes:
            if not _IDENTIFIER.match(spec.name):
                raise SchemaError(
                    f"attribute name {spec.name!r} must match [A-Za-z_][A-Za-z0-9_]*"
                )
            if spec.name in seen:
                raise SchemaError(f"duplicate attribute name: {spec.name!r}")
            seen.add(spec.name)

            try:
                kind = get_kind(spec.kind)
            except RegistryError as exc:
                raise SchemaError(str(exc)) from exc
            try:
                kind.validate(spec)
            except ValueError as exc:
                raise SchemaError(str(exc)) from exc

            if not _is_finite(spec.weight):
                raise SchemaError(f"attribute {spec.name!r}: weight must be finite")
            if spec.weight < 0:
                raise SchemaError(f"attribute {spec.name!r}: weight must be >= 0")

        total = sum(s.weight for s in self.attributes)
        if abs(total - 1.0) > ATTRIBUTE_WEIGHT_SUM_TOLERANCE:
            raise SchemaError(
                f"attribute weights must sum to 1.0, got {total!r}. "
                "Use AttributeSchema.from_weights(..., normalize=True) to "
                "normalize automatically."
            )
        if not any(s.weight > 0 for s in self.attributes):
            raise SchemaError("at least one attribute weight must be > 0")

        if not _is_finite(self.similarity_threshold) or not (
            0.0 <= self.similarity_threshold <= 1.0
        ):
            raise SchemaError("similarity_threshold must be in [0, 1]")

        if self.name_attribute is not None:
            spec = self.spec(self.name_attribute)
            canonical = get_kind(spec.kind).name
            if canonical != "fuzzy_text":
                raise SchemaError(
                    f"name_attribute {self.name_attribute!r} must use kind "
                    f"'fuzzy_text', got {canonical!r}"
                )

        if self.cluster_dim <= 0 or self.match_dim <= 0:
            raise SchemaError("cluster_dim and match_dim must be positive")
        for label, dim in (("cluster_dim", self.cluster_dim), ("match_dim", self.match_dim)):
            if dim > CKKS_SLOT_LIMIT:
                raise SchemaError(
                    f"{label} ({dim}) exceeds the CKKS slot limit "
                    f"({CKKS_SLOT_LIMIT}); reduce attribute dims or bucket counts"
                )

    # -- 维度与布局 ---------------------------------------------------------

    @property
    def cluster_dim(self) -> int:
        return sum(get_kind(s.kind).dims(s)[0] for s in self.attributes)

    @property
    def match_dim(self) -> int:
        return sum(get_kind(s.kind).dims(s)[1] for s in self.attributes)

    @property
    def weights(self) -> tuple[float, ...]:
        return tuple(s.weight for s in self.attributes)

    def _layout(self, index: int) -> dict[str, slice]:
        out: dict[str, slice] = {}
        offset = 0
        for spec in self.attributes:
            dim = get_kind(spec.kind).dims(spec)[index]
            out[spec.name] = slice(offset, offset + dim)
            offset += dim
        return out

    def layout(self) -> dict[str, slice]:
        """各属性在 cluster 向量里的切片。"""
        return self._layout(0)

    def match_layout(self) -> dict[str, slice]:
        """各属性在 match 向量里的切片。"""
        return self._layout(1)

    def spec(self, name: str) -> AttributeSpec:
        for spec in self.attributes:
            if spec.name == name:
                return spec
        raise SchemaError(f"unknown attribute: {name!r}")

    # -- 序列化 -------------------------------------------------------------

    def fingerprint(self) -> str:
        """布局指纹：布局不同（含顺序不同）则指纹不同。"""

        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "similarity_threshold": self.similarity_threshold,
            "name_attribute": self.name_attribute,
            "standardize_cluster": self.standardize_cluster,
            "attributes": [s.to_dict() for s in self.attributes],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | Sequence[Any]) -> "AttributeSchema":
        if isinstance(value, Mapping):
            unknown = sorted(set(value) - _SCHEMA_KEYS)
            if unknown:
                raise SchemaError(
                    f"unknown schema keys {unknown}; allowed: {sorted(_SCHEMA_KEYS)}"
                )
            attributes = value.get("attributes")
            if attributes is None:
                raise SchemaError("schema object requires an 'attributes' list")
            return cls(
                attributes=attributes,
                similarity_threshold=float(
                    value.get("similarity_threshold", MULTI_ATTRIBUTE_THRESHOLD)
                ),
                name_attribute=value.get("name_attribute"),
                standardize_cluster=bool(value.get("standardize_cluster", True)),
            )
        return cls(attributes=value)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "AttributeSchema":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_json_file(cls, path: str | Path) -> "AttributeSchema":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    # -- 便捷构造 -----------------------------------------------------------

    @classmethod
    def from_weights(
        cls,
        weights: Mapping[str, float],
        kinds: Mapping[str, str] | None = None,
        params: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        normalize: bool = True,
        **schema_kwargs: Any,
    ) -> "AttributeSchema":
        """按 ``{属性名: 权重}`` 快速构造 schema。

        Args:
            normalize: ``True``（默认）时把权重按总和归一化 —— 适合
                ``{"name": 3, "dob": 1}`` 这种整数写法；``False`` 时要求
                调用方自己保证权重和为 1。
        """

        if not weights:
            raise SchemaError("weights cannot be empty")
        kinds = dict(kinds or {})
        params = dict(params or {})
        total = sum(float(v) for v in weights.values())
        if total <= 0:
            raise SchemaError("weights must sum to a positive value")
        specs = [
            AttributeSpec(
                name=name,
                kind=kinds.get(name, "exact"),
                weight=(float(weight) / total) if normalize else float(weight),
                params=params.get(name, {}),
            )
            for name, weight in weights.items()
        ]
        return cls(attributes=tuple(specs), **schema_kwargs)

    @classmethod
    def default(cls) -> "AttributeSchema":
        """V1 的 name+DOB 配置，等价于默认的 ``MultiAttributeConfig``。"""

        from .config import MultiAttributeConfig

        return MultiAttributeConfig().to_schema()


SchemaLike = "AttributeSchema | MultiAttributeConfig | Mapping[str, Any] | None"


def resolve_schema(value: Any = None) -> AttributeSchema:
    """把各种可接受的输入统一成 ``AttributeSchema``。

    接受：``None``（用默认 name+DOB）、``AttributeSchema``、mapping、
    或任何带 ``to_schema()`` 的对象（例如 ``MultiAttributeConfig``）。
    """

    if value is None:
        return AttributeSchema.default()
    if isinstance(value, AttributeSchema):
        return value
    if isinstance(value, Mapping):
        return AttributeSchema.from_dict(value)
    to_schema = getattr(value, "to_schema", None)
    if callable(to_schema):
        return to_schema()
    raise TypeError(
        "cfg must be None, AttributeSchema, a mapping, or expose to_schema(); "
        f"got {type(value).__name__}"
    )


def schema_from_json_file(path: str | Path) -> AttributeSchema:
    return AttributeSchema.from_json_file(path)


def _is_finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


__all__ = [
    "AttributeSchema",
    "AttributeSpec",
    "SchemaError",
    "SchemaLike",
    "resolve_schema",
    "registered_kinds",
    "schema_from_json_file",
]
