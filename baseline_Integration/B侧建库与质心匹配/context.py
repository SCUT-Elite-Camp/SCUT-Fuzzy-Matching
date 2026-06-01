# ckks/context.py
"""
TenSEAL CKKS context 创建工具。
"""

from __future__ import annotations
from typing import Any

from config.params import POLY_MODULUS_DEGREE, COEFF_MOD_BIT_SIZES, SCALE

try:
    import tenseal as ts
    _TENSEAL_AVAILABLE = True
except ImportError:
    _TENSEAL_AVAILABLE = False
    ts = None


def create_ckks_context() -> Any:
    """
    创建包含 secret key 的完整 TenSEAL CKKS context。
    生成 Galois keys 和 relin keys。

    返回：
        TenSEAL Context 对象（含 secret key，仅 A 侧持有）
    """
    if not _TENSEAL_AVAILABLE:
        raise ImportError("TenSEAL 未安装，请执行: pip install tenseal")

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
    )
    context.global_scale = SCALE
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context


def get_public_context_bytes(context: Any) -> bytes:
    """
    从完整 context 导出仅含公钥的 context bytes。

    严格约束：public context 不得包含 secret key。
    B 侧只能持有此 bytes。
    """
    if not _TENSEAL_AVAILABLE:
        raise ImportError("TenSEAL 未安装")

    # 序列化时指定不包含 secret key
    return context.serialize(save_secret_key=False)


def load_public_context(public_context_bytes: bytes) -> Any:
    """
    从 bytes 加载仅含公钥的 context，供 B 侧使用。
    """
    if not _TENSEAL_AVAILABLE:
        raise ImportError("TenSEAL 未安装")

    return ts.context_from(public_context_bytes)
