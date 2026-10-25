"""把记录编码成 (cluster_vector, match_vector)。

本模块现在由 ``AttributeSchema`` 驱动：逐个属性调用注册表里的编码器，块内先
L2 归一化，再乘以 ``sqrt(weight)`` 后按 schema 顺序拼接。展开后 CKKS 点积
即近似::

    S = Σ wᵢ · simᵢ

对默认的 name+DOB schema，输出与 V1 的实现**逐位相同**：同样的属性顺序、同样
先取 200 维 MinHash 再各自 L2、同样的日期哈希块、同样的 ``sqrt(w)`` 缩放。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

# kinds 必须先于编码逻辑被 import，以触发内置 kind 的注册。
from . import kinds  # noqa: F401
from .hashing import _exact_hash_vector, exact_hash_vector, normalize_dob  # noqa: F401
from .model import AttributeRecord, MatchRecord, record_values
from .registry import AttributeBlock, encode_attribute
from .schema import AttributeSchema, resolve_schema

__all__ = [
    "AttributeBlock",
    "AttributeRecord",
    "MatchRecord",
    "encode_attribute_matrix",
    "encode_record_vectors",
    "attribute_similarities",
    "exact_hash_vector",
    "normalize_dob",
    "plaintext_similarity",
    "_exact_hash_vector",
]


RecordLike = "MatchRecord | AttributeRecord | Mapping[str, Any]"


def encode_attribute_matrix(
    records: Iterable[Any],
    cfg: Any = None,
) -> dict[str, AttributeBlock]:
    """返回每个属性的**未加权**编码块，便于调试与分数拆解。"""

    schema = resolve_schema(cfg)
    values = [record_values(record, schema) for record in records]
    return {
        spec.name: encode_attribute(spec, [v.get(spec.name) for v in values])
        for spec in schema.attributes
    }


def encode_record_vectors(
    records: Iterable[Any],
    cfg: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode records into weighted cluster and final-match vectors.

    Args:
        records: ``MatchRecord`` / ``AttributeRecord`` / mapping 的序列。
        cfg: ``None``（默认 name+DOB）、``AttributeSchema``、
            ``MultiAttributeConfig``，或等价的 mapping。

    Returns:
        cluster_vectors: shape ``(n, schema.cluster_dim)``
        match_vectors: shape ``(n, schema.match_dim)``

    两个向量的点积（跨任意 schema）等于::

        Σ wᵢ · simᵢ
    """

    schema = resolve_schema(cfg)
    rows = list(records)
    if not rows:
        return (
            np.empty((0, schema.cluster_dim), dtype=np.float64),
            np.empty((0, schema.match_dim), dtype=np.float64),
        )

    values = [record_values(record, schema) for record in rows]

    cluster_parts: list[np.ndarray] = []
    match_parts: list[np.ndarray] = []
    for spec in schema.attributes:
        block = encode_attribute(spec, [v.get(spec.name) for v in values])
        scale = math.sqrt(spec.weight)
        cluster_parts.append(scale * block.cluster)
        match_parts.append(scale * block.match)

    return (
        np.concatenate(cluster_parts, axis=1).astype(np.float64),
        np.concatenate(match_parts, axis=1).astype(np.float64),
    )


def attribute_similarities(
    left: Any,
    right: Any,
    cfg: Any = None,
) -> dict[str, float]:
    """逐属性的**未加权**相似度，用于测试与 demo 的分数拆解。

    ``plaintext_similarity(left, right, cfg) == Σ wᵢ · simᵢ`` 恒成立。
    """

    schema = resolve_schema(cfg)
    blocks = encode_attribute_matrix([left, right], schema)
    return {
        spec.name: float(np.dot(blocks[spec.name].match[0], blocks[spec.name].match[1]))
        for spec in schema.attributes
    }


def plaintext_similarity(
    left: Any,
    right: Any,
    cfg: Any = None,
) -> float:
    """Plaintext reference score for tests/debugging only."""

    schema = resolve_schema(cfg)
    _, vectors = encode_record_vectors([left, right], schema)
    return float(np.dot(vectors[0], vectors[1]))
