"""CKKS 同态运算模块。

提供密文-明文、密文-密文之间的点积、加法和乘法运算。
所有操作使用 TenSEAL API，输入输出类型与协议类型定义一致。
"""

from typing import Union

import numpy as np
import tenseal as ts


def dot_ct_pt(enc_vec: ts.CKKSVector, plain_vec: np.ndarray) -> ts.CKKSVector:
    """计算加密向量与明文向量的点积（内积）。

    实现：逐元素密文-明文乘法后，对所有槽位求和。
    (enc_vec[i] * plain_vec[i]) → sum over i

    Args:
        enc_vec: 加密向量。
        plain_vec: 明文向量，形状须与加密向量的槽位数一致。

    Returns:
        ts.CKKSVector: 加密的点积结果（标量密文）。
    """
    return (enc_vec * plain_vec.tolist()).sum()


def dot_ct_ct(enc_left: ts.CKKSVector, enc_right: ts.CKKSVector) -> ts.CKKSVector:
    """计算两个加密向量的点积（内积）。

    使用 TenSEAL 内置 dot 方法，内部执行 slot-wise 乘加。

    Args:
        enc_left: 左侧加密向量。
        enc_right: 右侧加密向量。

    Returns:
        ts.CKKSVector: 加密的点积结果（标量密文）。
    """
    return enc_left.dot(enc_right)


def add_plain(enc_val: ts.CKKSVector, plain_val: Union[float, np.ndarray]) -> ts.CKKSVector:
    """将明文值加到密文上（同态加法）。

    Args:
        enc_val: 加密值。
        plain_val: 明文标量或数组，形状须与密文槽位兼容。

    Returns:
        ts.CKKSVector: 加密的加法结果。
    """
    return enc_val + plain_val


def mul_plain(enc_val: ts.CKKSVector, plain_val: Union[float, np.ndarray]) -> ts.CKKSVector:
    """将密文与明文值相乘（同态标量/逐元素乘法）。

    Args:
        enc_val: 加密值。
        plain_val: 明文标量或数组，形状须与密文槽位兼容。

    Returns:
        ts.CKKSVector: 加密的乘法结果。
    """
    return enc_val * plain_val
