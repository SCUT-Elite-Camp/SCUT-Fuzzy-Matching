"""内置属性类型：``fuzzy_text`` / ``exact`` / ``date``。

三种 kind 全部复用仓库既有实现，没有引入新的相似度定义：

* ``fuzzy_text`` -> ``minhash.encoder.batch_encode`` + ``preprocessing.normalizer.l2_normalize``
* ``exact``      -> ``hashing.exact_hash_vector``
* ``date``       -> ``hashing.normalize_dob`` + ``hashing.exact_hash_vector``

**空值语义（与 V1 一致）**：``None`` 或纯空白的取值一律编码为全零块，该属性对
最终得分的贡献为 0。注意这是对 V1 的一处刻意收紧 —— V1 里空姓名会经 MinHash 的
``"<empty>"`` 哨兵拿到一个单位向量，等于空姓名拿满权重分；新实现统一按缺失处理。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from config.params import (
    DEFAULT_ATTRIBUTE_HASH_SEED,
    DEFAULT_EXACT_BUCKETS_PER_BLOCK,
    DEFAULT_EXACT_HASH_BLOCKS,
    FUZZY_TEXT_CLUSTER_DIM,
    FUZZY_TEXT_MATCH_DIM,
    NUM_PERMUTATIONS_CLUSTER,
)
from minhash.encoder import batch_encode
from preprocessing.normalizer import l2_normalize

from .hashing import exact_hash_vector, normalize_dob
from .registry import AttributeBlock, register_encoder

_DIGITS_ONLY = re.compile(r"\D")

_NORMALIZE_MODES = ("strip", "casefold", "digits")


def _is_missing(value: Any) -> bool:
    """``None`` 或纯空白视为缺失；其余值照常参与编码。"""

    if value is None:
        return True
    return isinstance(value, str) and not value.strip()


def _reject_unknown_params(spec: Any, allowed: Mapping[str, Any]) -> None:
    unknown = sorted(set(spec.params) - set(allowed))
    if unknown:
        raise ValueError(
            f"attribute {spec.name!r} (kind {spec.kind!r}) received unknown "
            f"params {unknown}; allowed: {sorted(allowed)}"
        )


def _param(spec: Any, key: str, default: Any) -> Any:
    value = spec.params.get(key, default)
    return default if value is None else value


def _blocks(spec: Any) -> tuple[int, int, int]:
    """取出精确类属性共用的 ``(blocks, buckets_per_block, seed)``。"""

    return (
        int(_param(spec, "blocks", DEFAULT_EXACT_HASH_BLOCKS)),
        int(_param(spec, "buckets_per_block", DEFAULT_EXACT_BUCKETS_PER_BLOCK)),
        int(_param(spec, "seed", DEFAULT_ATTRIBUTE_HASH_SEED)),
    )


def _validate_hash_params(spec: Any) -> None:
    blocks, buckets, _ = _blocks(spec)
    if blocks < 1:
        raise ValueError(f"attribute {spec.name!r}: blocks must be >= 1")
    if buckets < 2:
        raise ValueError(f"attribute {spec.name!r}: buckets_per_block must be >= 2")


# ---------------------------------------------------------------------------
# fuzzy_text
# ---------------------------------------------------------------------------


def _fuzzy_text_dims(spec: Any) -> tuple[int, int]:
    return (
        int(_param(spec, "cluster_dim", FUZZY_TEXT_CLUSTER_DIM)),
        int(_param(spec, "match_dim", FUZZY_TEXT_MATCH_DIM)),
    )


def _validate_fuzzy_text(spec: Any) -> None:
    _reject_unknown_params(spec, {"cluster_dim": None, "match_dim": None})
    cluster_dim, match_dim = _fuzzy_text_dims(spec)
    if cluster_dim <= 0 or match_dim <= 0:
        raise ValueError(f"attribute {spec.name!r}: fuzzy_text dims must be positive")
    if match_dim > cluster_dim:
        raise ValueError(
            f"attribute {spec.name!r}: match_dim ({match_dim}) cannot exceed "
            f"cluster_dim ({cluster_dim})"
        )
    if cluster_dim > NUM_PERMUTATIONS_CLUSTER:
        raise ValueError(
            f"attribute {spec.name!r}: cluster_dim ({cluster_dim}) cannot exceed "
            f"NUM_PERMUTATIONS_CLUSTER ({NUM_PERMUTATIONS_CLUSTER})"
        )


def _encode_fuzzy_text(values: Sequence[Any], spec: Any) -> AttributeBlock:
    """缺失行留零，其余行走 MinHash；与 V1 的 ``_encode_name_blocks`` 等价。"""

    cluster_dim, match_dim = _fuzzy_text_dims(spec)
    n = len(values)
    raw_cluster = np.zeros((n, cluster_dim), dtype=np.float64)

    present_idx = [i for i, value in enumerate(values) if not _is_missing(value)]
    if present_idx:
        encoded = batch_encode([values[i] for i in present_idx], cluster_dim)
        raw_cluster[present_idx] = encoded

    raw_match = raw_cluster[:, :match_dim].copy()
    return AttributeBlock(
        cluster=l2_normalize(raw_cluster),
        match=l2_normalize(raw_match),
    )


# ---------------------------------------------------------------------------
# exact
# ---------------------------------------------------------------------------


def _exact_dims(spec: Any) -> tuple[int, int]:
    blocks, buckets, _ = _blocks(spec)
    dim = blocks * buckets
    return dim, dim


def _validate_exact(spec: Any) -> None:
    _reject_unknown_params(
        spec,
        {"blocks": None, "buckets_per_block": None, "seed": None, "normalize": None},
    )
    _validate_hash_params(spec)
    mode = _param(spec, "normalize", "casefold")
    if mode not in _NORMALIZE_MODES:
        raise ValueError(
            f"attribute {spec.name!r}: normalize must be one of "
            f"{list(_NORMALIZE_MODES)}, got {mode!r}"
        )


def _apply_normalize(value: Any, mode: str) -> str | None:
    text = str(value).strip()
    if not text:
        return None
    if mode == "casefold":
        return text.casefold()
    if mode == "digits":
        # 电话号码等：丢掉所有非数字字符，使 "+1 (555) 010-0199" 与
        # "15550100199" 编码到同一个桶。
        digits = _DIGITS_ONLY.sub("", text)
        return digits or None
    return text


def _encode_exact(values: Sequence[Any], spec: Any) -> AttributeBlock:
    blocks, buckets, seed = _blocks(spec)
    mode = _param(spec, "normalize", "casefold")
    rows = [
        exact_hash_vector(
            None if _is_missing(v) else _apply_normalize(v, mode),
            blocks=blocks,
            buckets_per_block=buckets,
            seed=seed,
        )
        for v in values
    ]
    matrix = np.stack(rows, axis=0)
    return AttributeBlock(cluster=matrix, match=matrix)


# ---------------------------------------------------------------------------
# date
# ---------------------------------------------------------------------------


_INVALID_MODES = ("raise", "missing")


def _validate_date(spec: Any) -> None:
    _reject_unknown_params(
        spec,
        {
            "blocks": None,
            "buckets_per_block": None,
            "seed": None,
            "day_first": None,
            "on_invalid": None,
        },
    )
    _validate_hash_params(spec)
    mode = _param(spec, "on_invalid", "raise")
    if mode not in _INVALID_MODES:
        raise ValueError(
            f"attribute {spec.name!r}: on_invalid must be one of "
            f"{list(_INVALID_MODES)}, got {mode!r}"
        )


def _encode_date(values: Sequence[Any], spec: Any) -> AttributeBlock:
    blocks, buckets, seed = _blocks(spec)
    day_first = bool(_param(spec, "day_first", False))
    on_invalid = _param(spec, "on_invalid", "raise")

    normalized_values: list[str | None] = []
    invalid: list[tuple[str, str]] = []
    for value in values:
        if _is_missing(value):
            normalized_values.append(None)
            continue
        try:
            # day_first 默认 False —— DD/MM 与 MM/DD 有歧义，必须显式声明。
            normalized_values.append(normalize_dob(value, day_first=day_first))
        except ValueError as exc:
            # 默认 fail closed：静默把垃圾值当成"缺失"会悄悄抹掉一条记录的全部
            # 该属性信息。真实脏数据（FEBRL 4b 有 1.3% 的非法日期）需要显式
            # 声明 on_invalid="missing" 才降级处理。
            #
            # 无论哪种策略都记下首个原始错因：批量报错只给个列表的话，调用方还得
            # 自己去猜到底是格式不对还是日期本身不存在。
            normalized_values.append(None)
            if on_invalid == "raise":
                invalid.append((str(value), str(exc)))

    if invalid:
        preview = [value for value, _ in invalid[:5]]
        raise ValueError(
            f"attribute {spec.name!r}: {len(invalid)} unparseable date value(s), "
            f"e.g. {preview}. First error: {invalid[0][1]} "
            'Set on_invalid="missing" to treat them as missing.'
        )

    matrix = np.stack(
        [
            exact_hash_vector(value, blocks=blocks, buckets_per_block=buckets, seed=seed)
            for value in normalized_values
        ],
        axis=0,
    )
    return AttributeBlock(cluster=matrix, match=matrix)


register_encoder(
    "fuzzy_text",
    dims=_fuzzy_text_dims,
    encode=_encode_fuzzy_text,
    validate=_validate_fuzzy_text,
    aliases=("text", "name"),
)
register_encoder(
    "exact",
    dims=_exact_dims,
    encode=_encode_exact,
    validate=_validate_exact,
    aliases=("categorical", "category"),
)
register_encoder(
    "date",
    dims=_exact_dims,
    encode=_encode_date,
    validate=_validate_date,
    aliases=("dob",),
)

__all__ = ["AttributeBlock"]
