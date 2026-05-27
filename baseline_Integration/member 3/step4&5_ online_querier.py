"""
party_a/online_querier.py

成员三负责：
Step 4:
    A 侧解密成员一返回的 encrypted_sim_scores，
    选择最相似的 cluster。

Step 5:
    构造 one-hot selector，
    加密 selector，
    生成 SecondRoundRequest(encrypted_query_50, encrypted_selector)。

注意：
1. 本文件只实现成员三模块，不实现 B 侧列式匹配。
2. 不接收 cluster_matrix。
3. 不把 selected_cluster 放进 SecondRoundRequest。
4. 不把 secret_context 传给 B 侧接口。
5. 当前 baseline 只支持单查询 m=1。
6. 若 protocol/types.py 或 ckks/operations.py 暂时缺失，
   本文件提供最小兼容适配层，方便先跑通成员三模块。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


# =========================================================
# 1. 优先复用项目统一类型
# =========================================================

try:
    from protocol.types import (
        PartyALocalState,
        SecondRoundRequest,
        ClusterSelectionDebug,
        EncryptedVectorK,
    )
except ImportError:
    # -----------------------------------------------------
    # 最小兼容适配层：
    # 只有在上游 protocol/types.py 尚未实现时才启用。
    # 字段名严格按照接口规范，避免改变协议语义。
    # -----------------------------------------------------

    EncryptedVectorK = Any

    @dataclass
    class PartyALocalState:
        secret_context: object
        encrypted_query_50: Any

    @dataclass
    class SecondRoundRequest:
        encrypted_query_50: Any
        encrypted_selector: EncryptedVectorK

    @dataclass
    class ClusterSelectionDebug:
        selected_cluster: int


# =========================================================
# 2. 优先复用 ckks/operations.py
# =========================================================

try:
    import ckks.operations as ckks_ops
except ImportError:
    ckks_ops = None


# =========================================================
# 3. TenSEAL 最小兜底
# =========================================================

try:
    import tenseal as ts
except ImportError:
    ts = None


# =========================================================
# 4. 内部工具函数
# =========================================================

def _load_ckks_vector(cipher: bytes, context: object):
    """
    将 bytes 反序列化为 TenSEAL CKKSVector。

    说明：
    - 网络通信或通信量评估时，密文通常是 bytes。
    - 同进程端到端测试时，密文可能已经是 TenSEAL 对象。
    """
    if ts is None:
        raise ImportError(
            "TenSEAL is required to deserialize CKKS ciphertext bytes. "
            "Please install tenseal or provide ckks.operations wrappers."
        )

    return ts.ckks_vector_from(context, cipher)


def _decrypt_one_cipher(cipher: Any, secret_context: object) -> list[float]:
    """
    解密一个密文对象或密文字节。

    返回 list[float]，兼容：
    - 加密 scalar：decrypt 后长度为 1
    - packed vector：decrypt 后长度为 k
    """
    if isinstance(cipher, bytes):
        cipher = _load_ckks_vector(cipher, secret_context)

    if not hasattr(cipher, "decrypt"):
        raise TypeError(
            f"Unsupported ciphertext type: {type(cipher)}. "
            "Expected TenSEAL CKKSVector object or serialized bytes."
        )

    values = cipher.decrypt()

    if isinstance(values, (float, int)):
        return [float(values)]

    return [float(x) for x in values]


def _encrypt_vector_plain(vector: np.ndarray, secret_context: object):
    """
    使用 TenSEAL 直接加密向量的兜底实现。

    优先级低于 ckks/operations.py。
    只有当 ckks.operations 没有提供统一 encrypt 接口时才使用。
    """
    if ts is None:
        raise ImportError(
            "TenSEAL is required to encrypt selector. "
            "Please install tenseal or implement ckks.operations.encrypt_vector."
        )

    return ts.ckks_vector(secret_context, vector.astype(float).tolist())


# =========================================================
# 5. Step 4: 解密 centroid 相似度
# =========================================================

def decrypt_sim_scores(
    encrypted_sim_scores: Any,
    secret_context: object,
) -> np.ndarray:
    """
    Step 4 的第一部分：
    A 侧使用 secret_context 解密成员一返回的质心相似度。

    参数：
        encrypted_sim_scores:
            可以是：
            1. list[TenSEAL CKKSVector]
            2. list[bytes]
            3. 单个 packed CKKSVector
            4. 单个 packed bytes

        secret_context:
            A 侧私钥上下文，只能留在 A 侧。

    返回：
        sim_scores: np.ndarray, shape = (k,)
    """

    # -----------------------------------------------------
    # 优先使用 ckks/operations.py 中的统一解密接口
    # -----------------------------------------------------
    if ckks_ops is not None:
        for fn_name in (
            "decrypt_vector",
            "decrypt_ckks_vector",
            "decrypt_cipher",
            "decrypt",
        ):
            if hasattr(ckks_ops, fn_name):
                fn = getattr(ckks_ops, fn_name)
                try:
                    values = fn(encrypted_sim_scores, secret_context)
                    arr = np.asarray(values, dtype=float).reshape(-1)
                    if arr.size > 0:
                        return arr
                except TypeError:
                    pass

    # -----------------------------------------------------
    # 兼容：encrypted_sim_scores 是 list
    # -----------------------------------------------------
    if isinstance(encrypted_sim_scores, (list, tuple)):
        plain_values: list[float] = []

        for cipher in encrypted_sim_scores:
            decrypted = _decrypt_one_cipher(cipher, secret_context)

            # 常见情况：每个密文是一个 scalar，取第一个值
            if len(decrypted) == 1:
                plain_values.append(decrypted[0])
            else:
                # 如果上游返回 list of packed vector，也保持兼容
                plain_values.extend(decrypted)

        return np.asarray(plain_values, dtype=float).reshape(-1)

    # -----------------------------------------------------
    # 兼容：encrypted_sim_scores 是单个 packed ciphertext
    # -----------------------------------------------------
    decrypted = _decrypt_one_cipher(encrypted_sim_scores, secret_context)
    return np.asarray(decrypted, dtype=float).reshape(-1)


# =========================================================
# 6. Step 4: 选择 cluster 并构造 one-hot
# =========================================================

def build_selector(
    sim_scores: np.ndarray,
    k: int,
) -> tuple[int, np.ndarray]:
    """
    Step 4 的第二部分：
    根据解密后的 centroid 相似度选择最大值对应的 cluster，
    并构造 one-hot selector。

    参数：
        sim_scores: shape = (k,)
        k: cluster 数量

    返回：
        selected_cluster: int
            只能留在 A 侧 debug / test / log。
        selector: np.ndarray
            one-hot 向量，shape = (k,)
    """

    sim_scores = np.asarray(sim_scores, dtype=float).reshape(-1)

    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")

    if sim_scores.shape != (k,):
        raise ValueError(
            f"sim_scores shape mismatch: expected ({k},), "
            f"got {sim_scores.shape}"
        )

    if not np.all(np.isfinite(sim_scores)):
        raise ValueError(
            "sim_scores contains NaN or Inf. "
            "Please check Step 3 CKKS dot-product / deserialization."
        )

    selected_cluster = int(np.argmax(sim_scores))

    selector = np.zeros(k, dtype=float)
    selector[selected_cluster] = 1.0

    return selected_cluster, selector


# =========================================================
# 7. Step 5: 加密 selector
# =========================================================

def encrypt_selector(
    selector: np.ndarray,
    secret_context: object,
) -> EncryptedVectorK:
    """
    Step 5：
    将 one-hot selector 加密为 encrypted_selector。

    注意：
    - 这里必须加密 selector。
    - 不能把 selected_cluster 明文传给 B。
    """

    selector = np.asarray(selector, dtype=float).reshape(-1)

    if selector.ndim != 1:
        raise ValueError("selector must be a 1-D vector")

    if not np.isclose(selector.sum(), 1.0):
        raise ValueError(
            "selector must be one-hot: sum(selector) should be 1"
        )

    if not np.all((selector == 0.0) | (selector == 1.0)):
        raise ValueError(
            "selector must be one-hot: values should only be 0 or 1"
        )

    # -----------------------------------------------------
    # 优先使用 ckks/operations.py 中的统一加密接口
    # -----------------------------------------------------
    if ckks_ops is not None:
        for fn_name in (
            "encrypt_vector",
            "encrypt_ckks_vector",
            "encrypt_plain_vector",
            "encrypt",
        ):
            if hasattr(ckks_ops, fn_name):
                fn = getattr(ckks_ops, fn_name)
                try:
                    return fn(selector, secret_context)
                except TypeError:
                    pass

    # -----------------------------------------------------
    # 最小兜底：TenSEAL 直接加密 packed vector
    # -----------------------------------------------------
    return _encrypt_vector_plain(selector, secret_context)


# =========================================================
# 8. 成员三主接口：Step 4 + Step 5
# =========================================================

def choose_cluster_and_build_request(
    encrypted_sim_scores: Any,
    party_a_state: PartyALocalState,
    k: int,
) -> tuple[SecondRoundRequest, ClusterSelectionDebug]:
    """
    成员三对外主接口。

    输入：
        encrypted_sim_scores:
            成员一 Step 3 返回的加密 centroid 相似度。

        party_a_state:
            成员二 Step 2 留在 A 侧的本地状态。
            必须包含：
                - secret_context
                - encrypted_query_50

        k:
            成员一 artifacts 中的 cluster 数量。

    输出：
        second_round_request:
            只包含：
                - encrypted_query_50
                - encrypted_selector

            这是允许发给成员四 B 侧接口的对象。
            不包含 selected_cluster。

        debug:
            只留在 A 侧本地。
            可以用于测试、日志、排错。
            不得传给 B 侧。

    协议语义：
        encrypted_sim_scores + PartyALocalState
        -> selected_cluster 仅 A 侧 debug
        -> SecondRoundRequest(encrypted_query_50, encrypted_selector)
    """

    if not hasattr(party_a_state, "secret_context"):
        raise AttributeError(
            "party_a_state must have attribute 'secret_context'"
        )

    if not hasattr(party_a_state, "encrypted_query_50"):
        raise AttributeError(
            "party_a_state must have attribute 'encrypted_query_50'"
        )

    # Step 4: A 侧解密相似度
    sim_scores = decrypt_sim_scores(
        encrypted_sim_scores=encrypted_sim_scores,
        secret_context=party_a_state.secret_context,
    )

    # Step 4: 选择最可能的 cluster，构造 one-hot
    selected_cluster, selector = build_selector(
        sim_scores=sim_scores,
        k=k,
    )

    # Step 5: 加密 one-hot selector
    encrypted_selector = encrypt_selector(
        selector=selector,
        secret_context=party_a_state.secret_context,
    )

    # Step 5: 构造第二轮请求
    # 注意：这里绝不能放 selected_cluster。
    second_round_request = SecondRoundRequest(
        encrypted_query_50=party_a_state.encrypted_query_50,
        encrypted_selector=encrypted_selector,
    )

    # Debug 只留在 A 侧
    debug = ClusterSelectionDebug(
        selected_cluster=selected_cluster,
    )

    return second_round_request, debug
