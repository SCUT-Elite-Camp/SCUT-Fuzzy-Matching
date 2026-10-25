"""属性编码器注册表：把「一种属性」映射到「一个向量块」。

设计要点
--------
* ``AttributeSpec`` 只声明 ``kind`` 和 ``params``，具体维度与编码方式由注册表里
  对应的 ``AttributeKind`` 决定。新增一种属性类型只需 ``register_encoder``，
  协议层完全不需要改动。
* 每个 block 返回**未加权**的 ``(cluster, match)`` 两个矩阵。权重由调用方以
  ``sqrt(weight)`` 乘上去，保证 CKKS 点积近似等于 ``Σ wᵢ · simᵢ``。
* ``encode_attribute`` 会断言输出形状与 ``kind.dims()`` 声明的维度一致 —— 这条
  断言是「声明一个 kind 就能得到正确协议」的可信来源。

本模块刻意不在模块层 import ``schema``（只用于类型标注），否则会与
``schema -> registry`` 形成循环导入。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型标注
    from .schema import AttributeSpec


class RegistryError(ValueError):
    """注册表相关错误（未知 kind、重复注册、形状不符）。"""


@dataclass(frozen=True)
class AttributeBlock:
    """单个属性编码后的两个块，形状 ``(n, cluster_dim)`` / ``(n, match_dim)``。"""

    cluster: np.ndarray
    match: np.ndarray


DimsFn = Callable[["AttributeSpec"], "tuple[int, int]"]
EncodeFn = Callable[[Sequence[Any], "AttributeSpec"], AttributeBlock]
ValidateFn = Callable[["AttributeSpec"], None]


@dataclass(frozen=True)
class AttributeKind:
    """一种属性类型：如何算维度、如何编码、参数是否合法。"""

    name: str
    dims: DimsFn
    encode: EncodeFn
    validate: ValidateFn


_KINDS: dict[str, AttributeKind] = {}
_ALIASES: dict[str, str] = {}
_BUILTINS_LOADED = False


def _ensure_builtins() -> None:
    """惰性加载内置 kind，避免 registry <-> kinds 的循环导入。

    无论调用方先 import ``multi_attribute``、``multi_attribute.encoder`` 还是
    ``multi_attribute.registry``，首次查表时都会走到这里完成注册。
    """

    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    from importlib import import_module

    import_module(".kinds", __package__)


def _canonical_kind(name: str) -> str:
    key = str(name).strip().lower()
    return _ALIASES.get(key, key)


def register_encoder(
    name: str,
    *,
    dims: DimsFn,
    encode: EncodeFn,
    validate: ValidateFn | None = None,
    aliases: Sequence[str] = (),
    replace: bool = False,
) -> None:
    """注册一种属性类型。

    Args:
        name: kind 名称（大小写不敏感）。
        dims: ``spec -> (cluster_dim, match_dim)``。
        encode: ``(values, spec) -> AttributeBlock``。
        validate: 额外的参数校验，抛 ``ValueError`` 表示配置非法。
        aliases: 额外的别名。
        replace: 允许覆盖已注册的同名 kind（仅供测试使用）。
    """

    _ensure_builtins()
    key = str(name).strip().lower()
    if not key:
        raise RegistryError("Attribute kind name cannot be empty")
    if key in _KINDS and not replace:
        raise RegistryError(f"Attribute kind already registered: {key}")

    _KINDS[key] = AttributeKind(
        name=key,
        dims=dims,
        encode=encode,
        validate=validate or (lambda spec: None),
    )
    for alias in aliases:
        alias_key = str(alias).strip().lower()
        if alias_key and alias_key != key:
            _ALIASES[alias_key] = key


def unregister_encoder(name: str, *, aliases: Sequence[str] = ()) -> None:
    """移除一个 kind（主要供测试在 fixture 里复原注册表）。"""

    _ensure_builtins()
    _KINDS.pop(_canonical_kind(name), None)
    for alias in aliases:
        _ALIASES.pop(str(alias).strip().lower(), None)


def get_kind(name: str) -> AttributeKind:
    """按名字或别名取回 kind，未注册时列出所有可用 kind。"""

    _ensure_builtins()
    key = _canonical_kind(name)
    try:
        return _KINDS[key]
    except KeyError as exc:
        raise RegistryError(
            f"Unknown attribute kind: {name!r}. Available: {sorted(_KINDS)}"
        ) from exc


def registered_kinds() -> tuple[str, ...]:
    """返回所有已注册的规范 kind 名（不含别名），已排序。"""

    _ensure_builtins()
    return tuple(sorted(_KINDS))


def encode_attribute(
    spec: "AttributeSpec",
    values: Sequence[Any],
) -> AttributeBlock:
    """编码一个属性列，并校验输出形状与声明维度一致。"""

    kind = get_kind(spec.kind)
    expected_cluster, expected_match = kind.dims(spec)
    block = kind.encode(list(values), spec)

    n = len(values)
    if block.cluster.shape != (n, expected_cluster):
        raise RegistryError(
            f"kind {kind.name!r} declared cluster dim {expected_cluster} but "
            f"encoded {block.cluster.shape} for attribute {spec.name!r}"
        )
    if block.match.shape != (n, expected_match):
        raise RegistryError(
            f"kind {kind.name!r} declared match dim {expected_match} but "
            f"encoded {block.match.shape} for attribute {spec.name!r}"
        )
    return block


__all__ = [
    "AttributeBlock",
    "AttributeKind",
    "RegistryError",
    "encode_attribute",
    "get_kind",
    "register_encoder",
    "registered_kinds",
    "unregister_encoder",
]
