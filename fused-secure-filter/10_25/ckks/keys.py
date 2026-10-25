"""CKKS 加密/解密/序列化操作模块。

提供向量加密、解密、密文与上下文序列化等基础原语。
"""

import numpy as np
import tenseal as ts


def serialize_public_context(ctx: ts.Context) -> bytes:
    """复制上下文并移除私钥后序列化为公开字节流。

    公开上下文字节流包含 public key、relinearization keys 和 galois keys，
    供 B 侧密文运算使用，绝不包含 secret key。

    Args:
        ctx: 含私钥的完整 TenSEAL 上下文。

    Returns:
        bytes: 不含 secret key 的公开上下文字节流。

    Raises:
        RuntimeError: 序列化或反序列化上下文失败时抛出。
    """
    try:
        # 通过序列化/反序列化创建副本，避免修改原始上下文
        ctx_data = ctx.serialize()
        ctx_copy = ts.Context.load(ctx_data)
        # 移除私钥，生成仅含公钥材料的公开上下文
        ctx_copy.make_context_public()
        return ctx_copy.serialize()
    except Exception as e:
        raise RuntimeError(f"Failed to serialize public context: {e}") from e


def encrypt(vec: np.ndarray, ctx: ts.Context) -> ts.CKKSVector:
    """使用 CKKS 方案批量加密 numpy 向量。

    自动将非 float64 类型转换为 float64 以匹配 CKKS 编码精度。

    Args:
        vec: 待加密的明文数值向量，形状 (n,)。
        ctx: 已配置的 CKKS 上下文（含公钥与加密参数）。

    Returns:
        ts.CKKSVector: CKKS 密文向量对象，可参与同态运算或序列化。
    """
    if vec.dtype != np.float64:
        vec = vec.astype(np.float64)
    return ts.ckks_vector(ctx, vec.tolist())


def serialize_ct(ct: ts.CKKSVector) -> bytes:
    """序列化密文向量为字节流。

    Args:
        ct: 待序列化的 CKKS 密文向量。

    Returns:
        bytes: 密文的紧凑字节流表示。
    """
    return ct.serialize()


def decrypt(ct: ts.CKKSVector, sk) -> np.ndarray:
    """解密密文向量，返回 numpy 数组。

    CKKS 为近似加密方案，解密结果存在微小舍入误差（通常 <= 1e-6）。

    Args:
        ct: 待解密的 CKKS 密文向量对象。
        sk: 私钥对象。

    Returns:
        np.ndarray: 解密后的明文数值数组。
    """
    decrypted = ct.decrypt(sk)
    return np.array(decrypted)


def deserialize_ct(data: bytes, ctx: ts.Context) -> ts.CKKSVector:
    """从字节流反序列化为可运算的 CKKS 密文向量。

    Args:
        data: 序列化密文字节流。
        ctx: 已配置的 CKKS 上下文（参数须与序列化时一致）。

    Returns:
        ts.CKKSVector: 可参与同态运算的密文向量。
    """
    return ts.ckks_vector_from(ctx, data)
