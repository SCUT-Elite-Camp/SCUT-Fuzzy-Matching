# ckks/operations.py
"""
CKKS 同态加密操作封装。

所有密文向量操作必须通过此模块，不得在业务模块中假设可以自由访问 E(x)[i]。
支持 TenSEAL CKKSVector 操作。

当前实现：correctness-first，每个向量元素作为独立密文处理。
若切换到 packed CKKSVector 优化，必须同时更新成员二、成员四及测试。
"""

from __future__ import annotations
from typing import Any, List

import numpy as np

try:
    import tenseal as ts
    _TENSEAL_AVAILABLE = True
except ImportError:
    _TENSEAL_AVAILABLE = False
    ts = None


def _require_tenseal():
    if not _TENSEAL_AVAILABLE:
        raise ImportError("TenSEAL 未安装，请执行: pip install tenseal")


# ─── 基础 HE 操作 ─────────────────────────────────────────────────────────────

def dot_ct_pt(encrypted_vector: Any, plain_vector: np.ndarray) -> Any:
    """
    密文向量与明文向量的点积（CT-PT dot product）。

    对应论文 Algorithm 4 DotProduct(ctm1, ptm2)。

    参数：
        encrypted_vector: list of TenSEAL CKKSVector（每个元素对应向量的一个维度）
                          或单个 CKKSVector（packed 表示）
        plain_vector:     np.ndarray，形状 (d,)

    返回：
        加密标量（密文），表示点积结果
    """
    _require_tenseal()

    if isinstance(encrypted_vector, list):
        # scalar list 表示：逐元素相乘后求和
        assert len(encrypted_vector) == len(plain_vector), (
            f"长度不匹配: encrypted={len(encrypted_vector)}, plain={len(plain_vector)}"
        )
        result = None
        for i, (ct, pt) in enumerate(zip(encrypted_vector, plain_vector)):
            term = ct * float(pt)
            if result is None:
                result = term
            else:
                result = result + term
        return result
    else:
        # packed CKKSVector 表示：直接调用 TenSEAL 内置 dot
        return encrypted_vector.dot(plain_vector.tolist())


def dot_ct_ct(encrypted_left: Any, encrypted_right: Any) -> Any:
    """
    两个密文向量的点积（CT-CT dot product）。

    对应论文 Algorithm 4 DotProduct(ctm1, ctm2)。

    参数：
        encrypted_left:  list of CKKSVector 或 单个 CKKSVector
        encrypted_right: list of CKKSVector 或 单个 CKKSVector

    返回：
        加密标量（密文），表示点积结果
    """
    _require_tenseal()

    if isinstance(encrypted_left, list):
        assert isinstance(encrypted_right, list), "两侧表示必须一致"
        assert len(encrypted_left) == len(encrypted_right), (
            f"长度不匹配: left={len(encrypted_left)}, right={len(encrypted_right)}"
        )
        result = None
        for ct1, ct2 in zip(encrypted_left, encrypted_right):
            term = ct1 * ct2
            if result is None:
                result = term
            else:
                result = result + term
        return result
    else:
        # packed: 逐元素乘后 sum
        product = encrypted_left * encrypted_right
        # TenSEAL CKKSVector 没有内置 sum_all，用 sum()
        return product.sum()


def add_plain(encrypted_value: Any, plain_value: float) -> Any:
    """
    密文加明文常数（CKKSAddconst）。

    参数：
        encrypted_value: CKKSVector 或加密标量
        plain_value:     float

    返回：
        新密文
    """
    _require_tenseal()
    return encrypted_value + plain_value


def mul_plain(encrypted_value: Any, plain_value: float) -> Any:
    """
    密文乘明文常数（CKKSMultconst）。

    参数：
        encrypted_value: CKKSVector 或加密标量
        plain_value:     float（必须为正数，用于随机 mask）

    返回：
        新密文
    """
    _require_tenseal()
    return encrypted_value * plain_value


# ─── 向量加密 / 解密工具 ──────────────────────────────────────────────────────

def encrypt_vector_as_list(
    context: Any,
    vector: np.ndarray,
) -> List[Any]:
    """
    将明文向量逐元素加密为密文列表（scalar list 表示）。

    参数：
        context: TenSEAL context（含 secret key，用于加密）
        vector:  np.ndarray，形状 (d,)

    返回：
        list of CKKSVector，长度 d
    """
    _require_tenseal()
    return [ts.ckks_vector(context, [float(v)]) for v in vector]


def encrypt_vector_packed(
    context: Any,
    vector: np.ndarray,
) -> Any:
    """
    将明文向量打包加密为单个 CKKSVector（packed 表示）。

    参数：
        context: TenSEAL context
        vector:  np.ndarray，形状 (d,)

    返回：
        单个 CKKSVector
    """
    _require_tenseal()
    return ts.ckks_vector(context, vector.tolist())


def decrypt_scalar(context: Any, encrypted_scalar: Any) -> float:
    """
    解密加密标量，返回 float。

    参数：
        context:          TenSEAL context（含 secret key）
        encrypted_scalar: 单个 CKKSVector（表示标量）

    返回：
        float
    """
    _require_tenseal()
    result = encrypted_scalar.decrypt()
    if isinstance(result, (list, np.ndarray)):
        return float(result[0])
    return float(result)


def decrypt_vector(context: Any, encrypted_list: List[Any]) -> np.ndarray:
    """
    解密密文列表为明文向量。

    参数：
        context:        TenSEAL context（含 secret key）
        encrypted_list: list of CKKSVector

    返回：
        np.ndarray，形状 (d,)
    """
    _require_tenseal()
    values = []
    for ct in encrypted_list:
        val = ct.decrypt()
        values.append(float(val[0]) if isinstance(val, (list, np.ndarray)) else float(val))
    return np.array(values, dtype=np.float64)


def serialize_ciphertext(ct: Any) -> bytes:
    """序列化密文为 bytes，用于网络传输和通信量评估。"""
    _require_tenseal()
    return ct.serialize()


def deserialize_ciphertext(context: Any, data: bytes) -> Any:
    """从 bytes 反序列化密文。"""
    _require_tenseal()
    return ts.lazy_ckks_vector_from(data)
