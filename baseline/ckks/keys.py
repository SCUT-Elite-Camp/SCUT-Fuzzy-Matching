import numpy as np
import tenseal as ts

# ============================================================
# 核心功能接口
# ============================================================

def generate_keys(ctx):
    """生成公钥、私钥及重线性化密钥，并绑定至上下文。

    基于给定上下文生成重线性化密钥（同态乘法降维所需），
    提取私钥字节流，上下文自身作为公钥载体返回。

    功能:
        - 生成并绑定重线性化密钥（relinearization keys）
        - 提取私钥字节流
        - 返回可用于加密的上下文（含公钥材料）及私钥

    参数:
        ctx (tenseal.Context): 已初始化（含 Galois 密钥）的 CKKS 上下文

    返回值:
        tuple (tenseal.Context, SecretKey):
            - ctx:       绑定了重线性化密钥的上下文（承载公钥与参数）
            - SecretKey: 私钥对象，仅用于解密操作

    示例:
        >>> ctx = build_context()
        >>> pk_ctx, sk_bytes = generate_keys(ctx)
        >>> len(sk_bytes) > 0
        True

    注意事项:
        - 必须先调用 build_context() 创建上下文。
        - 重线性化密钥是执行同态乘法的必要条件，缺省将导致乘法失败。
        - 私钥字节流应妥善保管，不得泄露至逻辑层外部。
    """
    # 生成重线性化密钥：同态乘法后密文维度升高，需此密钥降维
    ctx.generate_relin_keys()

    # 提取私钥为字节流（仅解密时使用）
    secret_key = ctx.secret_key()

    return ctx, secret_key


def encrypt(vec, ctx):
    """使用 CKKS 方案批量加密 numpy 向量，返回密文向量。

    通过 pack_ckks 方法将 numpy 向量打包为单个明文多项式并加密，
    充分利用单指令多数据（SIMD）特性，一次加密即可操作整个向量。

    功能:
        将任意长度的 numpy 数值向量批量编码并加密为 CKKS 密文。

    参数:
        vec (numpy.ndarray): 待加密的明文数值向量（支持 float64/float32/int64 等）
        ctx (tenseal.Context): 已配置的 CKKS 上下文（含公钥与加密参数）

    返回值:
        tenseal.CKKSVector: CKKS 密文向量对象，可参与同态运算或序列化

    示例:
        >>> ctx = build_context()
        >>> ctx, sk = generate_keys(ctx)
        >>> data = np.array([0.5, -1.2, 3.14], dtype=np.float64)
        >>> ct = encrypt(data, ctx)
        >>> type(ct).__name__
        'CKKSVector'

    注意事项:
        - 输入向量长度须 ≤ poly_modulus_degree / 2 = 4096。
        - 非 float64 类型会自动转换为 float64，以匹配 CKKS 编码精度。
        - 内部调用 CKKSVector.pack_ckks 实现批量打包加密。
    """
    # 统一转换为 float64，保证 CKKS 定点编码精度一致
    if vec.dtype != np.float64:
        vec = vec.astype(np.float64)

    # pack_ckks: 将向量各元素打包为单个多项式的不同槽位，一次加密整条向量
    ct = ts.ckks_vector(ctx, vec.tolist())

    return ct

def decrypt(ct, sk):
    """解密密文向量，返回与原始输入格式一致的 numpy 数组。

    使用私钥字节流解密 CKKS 密文，还原为明文 numpy 数组。
    CKKS 是近似加密方案，解密结果存在微小舍入误差（通常 ≤ 10^-6）。

    功能:
        使用私钥解密 CKKS 密文，输出 numpy 数组。

    参数:
        ct (tenseal.CKKSVector): 待解密的 CKKS 密文向量对象
        sk (SecretKey): 私钥对象（由 generate_keys 生成）

    返回值:
        numpy.ndarray: 解密后的明文数值数组

    示例:
        >>> ctx = build_context()
        >>> ctx, sk = generate_keys(ctx)
        >>> data = np.array([1.0, 2.0, 3.0])
        >>> ct = encrypt(data, ctx)
        >>> result = decrypt(ct, sk)
        >>> np.allclose(data, result, atol=1e-6)
        True

    注意事项:
        - CKKS 为近似加密，解密误差通常在 10^-6 量级以内。
        - 私钥必须与加密时的上下文严格对应。
        - 私钥仅在本函数作用域内使用，不输出或记录至外部。
    """
    # 使用私钥解密；decrypt() 可接受 secret_key 参数覆盖上下文内密钥
    decrypted = ct.decrypt(sk)

    # 确保返回 numpy 数组，与输入格式一致
    return np.array(decrypted)


def serialize_ct(ct):
    """序列化密文向量为字节流，用于网络传输与持久化存储。

    调用 TenSEAL 原生序列化方法将密文转为字节流，
    保证密文完整性，反序列化后可正常执行同态运算。

    功能:
        将 CKKS 密文对象转换为字节流。

    参数:
        ct (tenseal.CKKSVector): 待序列化的 CKKS 密文向量

    返回值:
        bytes: 密文的紧凑字节流表示

    示例:
        >>> ctx = build_context()
        >>> ctx, sk = generate_keys(ctx)
        >>> ct = encrypt(np.array([1.0, 2.0]), ctx)
        >>> buf = serialize_ct(ct)
        >>> isinstance(buf, bytes)
        True

    注意事项:
        - 使用 TenSEAL 原生 serialize() 方法，保证跨平台兼容。
        - 字节流可用于磁盘持久化或网络传输。
    """
    return ct.serialize()


def deserialize_ct(data, ctx):
    """从字节流反序列化为可运算的 CKKS 密文向量。

    将序列化密文字节流还原为 CKKS 密文对象，
    反序列化后的密文可直接参与同态加、减、乘运算。

    功能:
        从字节流恢复可运算的 CKKS 密文向量。

    参数:
        data (bytes): 序列化密文字节流（由 serialize_ct 生成）
        ctx (tenseal.Context): 已配置的 CKKS 上下文（参数须与序列化时一致）

    返回值:
        tenseal.CKKSVector: 可参与同态运算的密文向量

    示例:
        >>> ctx = build_context()
        >>> ctx, sk = generate_keys(ctx)
        >>> ct = encrypt(np.array([1.0, 2.0, 3.0]), ctx)
        >>> buf = serialize_ct(ct)
        >>> ct2 = deserialize_ct(buf, ctx)
        >>> np.allclose(decrypt(ct2, sk), [1.0, 2.0, 3.0], atol=1e-6)
        True

    注意事项:
        - 上下文加密参数必须与序列化时完全一致（poly_modulus_degree 等）。
        - 反序列化密文可立即参与同态运算，无需额外初始化。
    """
    return ts.ckks_vector_from(ctx, data)