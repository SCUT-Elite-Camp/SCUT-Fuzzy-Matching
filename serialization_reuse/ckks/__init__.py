"""CKKS 子包 —— 同态加密基础原语。

提供上下文创建、密钥管理、序列化与同态运算的完整接口。
"""

from ckks.context import create_ckks_context
from ckks.keys import (
    decrypt,
    deserialize_ct,
    encrypt,
    serialize_ct,
    serialize_public_context,
)
from ckks.operations import add_plain, dot_ct_ct, dot_ct_pt, mul_plain

__all__ = [
    "create_ckks_context",
    "encrypt",
    "decrypt",
    "serialize_ct",
    "deserialize_ct",
    "serialize_public_context",
    "dot_ct_pt",
    "dot_ct_ct",
    "add_plain",
    "mul_plain",
]
