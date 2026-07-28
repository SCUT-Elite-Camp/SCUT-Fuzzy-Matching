"""CKKS 纵向 SIMD (SoA) 批处理原语模块。

实现基于槽位的纵向向量化计算：
- 一个特征对应一个 CKKSVector (长度 m)
- 查询编号占据 slot 0 .. m-1
"""

from collections.abc import Sequence

import numpy as np
import tenseal as ts

from config.params import POLY_MODULUS_DEGREE
from protocol.types import CipherLike

MAX_BATCH_SIZE = POLY_MODULUS_DEGREE // 2  # 4096


def _validate_batch_size(batch_size: int) -> None:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise ValueError(f"batch_size 必须为整数，当前类型: {type(batch_size)}")
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须为正整数，当前值: {batch_size}")
    if batch_size > MAX_BATCH_SIZE:
        raise ValueError(f"batch_size {batch_size} 超过最大允许值 {MAX_BATCH_SIZE}")


def _validate_finite(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 包含 NaN 或 Inf 非法数值")


def encrypt_feature_batch(
    matrix: np.ndarray, context: ts.Context
) -> list[ts.CKKSVector]:
    """(m, d) -> d 个长度 m 的密文。"""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"matrix 必须为二维数组，当前维度: {arr.ndim}")
    _validate_finite(arr, "matrix")

    m, d = arr.shape
    _validate_batch_size(m)
    if d <= 0:
        raise ValueError(f"特征数 d 必须为正整数，当前值: {d}")

    # Transpose to (d, m) for SoA layout: each row corresponds to feature f across m queries
    transposed = arr.T
    ciphertexts = []
    try:
        for f in range(d):
            vec = ts.ckks_vector(context, transposed[f].tolist())
            ciphertexts.append(vec)
    except Exception as e:
        raise RuntimeError(f"TenSEAL 批量加密过程失败: {e}") from e

    return ciphertexts


def serialize_feature_batch(
    ciphertexts: Sequence[CipherLike],
) -> list[bytes]:
    """保持特征顺序逐个序列化；bytes 输入保持不变。"""
    if len(ciphertexts) == 0:
        raise ValueError("密文列表不能为空")

    serialized: list[bytes] = []
    for idx, ciphertext in enumerate(ciphertexts):
        if isinstance(ciphertext, bytes):
            serialized.append(ciphertext)
            continue
        if not isinstance(ciphertext, ts.CKKSVector):
            raise ValueError(f"第 {idx} 个密文类型不支持序列化: {type(ciphertext)}")
        try:
            serialized.append(ciphertext.serialize())
        except Exception as e:
            raise RuntimeError(f"TenSEAL 第 {idx} 个密文序列化失败: {e}") from e
    return serialized


def load_feature_batch(
    ciphertexts: Sequence[CipherLike],
    context: ts.Context,
    *,
    feature_count: int,
    batch_size: int,
) -> list[ts.CKKSVector]:
    """反序列化并做数量、slot 数检查。"""
    _validate_batch_size(batch_size)
    if feature_count <= 0:
        raise ValueError(f"feature_count 必须为正整数，当前值: {feature_count}")

    if len(ciphertexts) != feature_count:
        raise ValueError(
            f"密文数量不匹配: 期望 {feature_count}，实际 {len(ciphertexts)}"
        )

    loaded = []
    for idx, item in enumerate(ciphertexts):
        if isinstance(item, bytes):
            try:
                vec = ts.ckks_vector_from(context, item)
            except Exception as e:
                raise RuntimeError(
                    f"从 bytes 反序列化第 {idx} 个 CKKSVector 失败: {e}"
                ) from e
        elif isinstance(item, ts.CKKSVector):
            vec = item
        else:
            raise ValueError(f"不支持的密文类型: {type(item)}")

        if vec.size() != batch_size:
            raise ValueError(
                f"第 {idx} 个密文槽位数 ({vec.size()}) 与期望 batch_size ({batch_size}) 不符"
            )
        loaded.append(vec)

    return loaded


def decrypt_score_batch(
    ciphertexts: Sequence[CipherLike],
    secret_context: ts.Context,
    *,
    output_count: int,
    batch_size: int,
) -> np.ndarray:
    """n 个长度 m 的密文 -> (m, n)。"""
    _validate_batch_size(batch_size)
    if output_count <= 0:
        raise ValueError(f"output_count 必须为正整数，当前值: {output_count}")

    if len(ciphertexts) != output_count:
        raise ValueError(
            f"密文数量不匹配: 期望 output_count={output_count}，实际 {len(ciphertexts)}"
        )

    loaded_vectors = load_feature_batch(
        ciphertexts,
        secret_context,
        feature_count=output_count,
        batch_size=batch_size,
    )

    decrypted_list = []
    for idx, vec in enumerate(loaded_vectors):
        try:
            decrypted_list.append(vec.decrypt())
        except Exception as e:
            raise RuntimeError(f"解密第 {idx} 个密文失败: {e}") from e

    # raw has shape (n, m), transpose to (m, n) so row q corresponds to query q
    raw = np.array(decrypted_list, dtype=np.float64)
    return raw.T


def batch_dot_ct_pt(
    encrypted_features: Sequence[ts.CKKSVector],
    plain_rows: np.ndarray,
) -> list[ts.CKKSVector]:
    """d 个特征密文 × (n, d) -> n 个长度 m 的分数密文。"""
    if len(encrypted_features) == 0:
        raise ValueError("encrypted_features 密文列表不能为空")

    d = len(encrypted_features)
    m = encrypted_features[0].size()
    _validate_batch_size(m)

    for idx, c in enumerate(encrypted_features):
        if c.size() != m:
            raise ValueError(
                f"第 {idx} 个特征密文槽位数 ({c.size()}) 与首个密文 ({m}) 不一致"
            )

    arr = np.asarray(plain_rows, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"plain_rows 必须为二维数组，当前维度: {arr.ndim}")
    _validate_finite(arr, "plain_rows")

    n_rows, n_cols = arr.shape
    if n_cols != d:
        raise ValueError(f"形状不匹配: plain_rows 列数 ({n_cols}) != 特征密文数 ({d})")
    if n_rows <= 0:
        raise ValueError(f"plain_rows 行数必须大于 0，当前值: {n_rows}")

    scores = []
    for i in range(n_rows):
        row = arr[i]
        # Start sum from the first product temporary object
        acc = encrypted_features[0] * float(row[0])
        for f in range(1, d):
            acc = acc + (encrypted_features[f] * float(row[f]))
        scores.append(acc)

    return scores


def batch_select_ct_pt(
    encrypted_selectors: Sequence[ts.CKKSVector],
    plain_column: np.ndarray,
    mask: np.ndarray,
) -> list[ts.CKKSVector]:
    """k 个 selector × (k, d) × (m,) -> d 个长度 m 的密文。"""
    if len(encrypted_selectors) == 0:
        raise ValueError("encrypted_selectors 列表不能为空")

    k = len(encrypted_selectors)
    m = encrypted_selectors[0].size()
    _validate_batch_size(m)

    for idx, s in enumerate(encrypted_selectors):
        if s.size() != m:
            raise ValueError(
                f"第 {idx} 个 selector 密文槽位数 ({s.size()}) 与首个密文 ({m}) 不一致"
            )

    plain = np.asarray(plain_column, dtype=np.float64)
    if plain.ndim != 2:
        raise ValueError(f"plain_column 必须为二维数组，当前维度: {plain.ndim}")
    _validate_finite(plain, "plain_column")

    pk, d = plain.shape
    if pk != k:
        raise ValueError(f"形状不匹配: plain_column 行数 ({pk}) != selector 数量 ({k})")
    if d <= 0:
        raise ValueError(f"特征数 d 必须为正整数，当前值: {d}")

    msk = np.asarray(mask, dtype=np.float64)
    if msk.ndim != 1:
        raise ValueError(f"mask 必须为一维数组，当前维度: {msk.ndim}")
    if msk.shape[0] != m:
        raise ValueError(f"mask 长度 ({msk.shape[0]}) 不等于 batch_size ({m})")
    _validate_finite(msk, "mask")
    if np.any(msk <= 0):
        raise ValueError("mask 元素必须严格大于 0")

    selected_features = []
    for f in range(d):
        # plain_column[i, f] * msk gives a (m,) ndarray for each cluster i
        # Multiply with encrypted_selectors[i] (size m) slot-wise
        acc_f = encrypted_selectors[0] * (plain[0, f] * msk).tolist()
        for i in range(1, k):
            acc_f = acc_f + (encrypted_selectors[i] * (plain[i, f] * msk).tolist())
        selected_features.append(acc_f)

    return selected_features


def batch_dot_ct_ct(
    left_features: Sequence[ts.CKKSVector],
    right_features: Sequence[ts.CKKSVector],
) -> ts.CKKSVector:
    """两个 d 特征密文列表 -> 一个长度 m 的分数密文。"""
    if len(left_features) == 0 or len(right_features) == 0:
        raise ValueError("特征密文列表不能为空")

    d = len(left_features)
    if len(right_features) != d:
        raise ValueError(
            f"特征密文列表长度不一致: left ({d}) vs right ({len(right_features)})"
        )

    m = left_features[0].size()
    _validate_batch_size(m)

    for idx in range(d):
        if left_features[idx].size() != m:
            raise ValueError(
                f"left_features[{idx}] 槽位数 ({left_features[idx].size()}) 与首个密文 ({m}) 不一致"
            )
        if right_features[idx].size() != m:
            raise ValueError(
                f"right_features[{idx}] 槽位数 ({right_features[idx].size()}) 与首个密文 ({m}) 不一致"
            )

    acc = left_features[0] * right_features[0]
    for f in range(1, d):
        acc = acc + (left_features[f] * right_features[f])

    return acc
