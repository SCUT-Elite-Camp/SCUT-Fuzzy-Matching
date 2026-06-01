# party_b/online_responder.py
"""
成员一 - Step 3：B 侧第一轮在线质心匹配。
成员四 - Step 6/7/8：B 侧列式匹配（由成员四实现，此处仅留桩）。

本文件只实现成员一负责的部分（compare_to_centroids）。
Step 6/7/8 的列式匹配函数由成员四在此文件中继续实现。

严格约束：
  - B 侧函数不得接收或访问 secret_context（私钥）
  - B 侧函数不得接收 encrypted_query_50（第一轮不发送）
  - 函数参数不得出现 selected_cluster（明文 cluster 索引）
"""

from __future__ import annotations
from typing import Any, List

import numpy as np

from protocol.protocol_types import FirstRoundRequest
from ckks.operations import dot_ct_pt, serialize_ciphertext, deserialize_ciphertext
from ckks.context import load_public_context


# ─── Step 3：第一轮质心匹配 ──────────────────────────────────────────────────

def compare_to_centroids(
    first_round_request: FirstRoundRequest,
    centroids: np.ndarray,
    serialize_output: bool = False,
) -> List[Any]:
    """
    B 侧接收 A 的第一轮请求，计算加密查询向量与所有质心的余弦相似度。

    对应论文 Algorithm 1 CompareToCentroids 函数。

    参数：
        first_round_request: FirstRoundRequest，包含 public_context_bytes 和 encrypted_query_200
        centroids:           np.ndarray，形状 (k, 200)，L2 normalized 质心
        serialize_output:    若为 True，返回 list[bytes]（模拟网络传输）；
                             否则返回 list[CKKSVector]（同进程测试）

    返回：
        encrypted_sim_scores: list，长度 k
          - serialize_output=False: list of CKKSVector
          - serialize_output=True:  list of bytes

    约束：
        - 不得解密任何密文
        - 不得访问 encrypted_query_50
        - 不得接收 secret_context
    """
    # 1. 加载 public context（不含 secret key）
    public_context = load_public_context(first_round_request.public_context_bytes)

    # 2. 反序列化 encrypted_query_200
    encrypted_q = _deserialize_encrypted_query(
        first_round_request.encrypted_query_200, public_context
    )

    # 3. 遍历每个质心，计算 CT-PT 点积（= cosine similarity，因为向量已归一化）
    k = centroids.shape[0]
    encrypted_sim_scores = []

    for c in range(k):
        centroid_vec = centroids[c]  # (200,)
        # CT-PT dot product: E(query_200_std) · centroid_c
        enc_score = dot_ct_pt(encrypted_q, centroid_vec)
        encrypted_sim_scores.append(enc_score)

    # 4. 按需序列化（模拟网络传输）
    if serialize_output:
        return [serialize_ciphertext(s) for s in encrypted_sim_scores]

    return encrypted_sim_scores


# ─── 内部辅助函数 ─────────────────────────────────────────────────────────────

def _deserialize_encrypted_query(
    encrypted_query_200: Any,
    public_context: Any,
) -> Any:
    """
    反序列化 encrypted_query_200。

    支持：
      - bytes（网络传输模式）→ 调用 deserialize
      - TenSEAL 对象（同进程模式）→ 直接使用，但 link context
    """
    try:
        import tenseal as ts
    except ImportError:
        raise ImportError("TenSEAL 未安装")

    if isinstance(encrypted_query_200, bytes):
        ct = ts.lazy_ckks_vector_from(encrypted_query_200)
        ct.link_context(public_context)
        return ct
    else:
        # 同进程模式：直接使用 TenSEAL 对象
        # 注意：同进程模式下对象已经绑定了 context
        return encrypted_query_200


# ─── Step 6/7/8 桩（由成员四实现） ───────────────────────────────────────────
# 成员四将在此函数中实现列式匹配逻辑。
# 此处仅定义函数签名，保证接口一致性。

def column_wise_matching(
    cluster_matrix: np.ndarray,
    second_round_request: Any,  # SecondRoundRequest
    public_context: Any,
    tau: float,
):
    """
    B 侧列式匹配（Step 6/7/8）。
    由成员四实现，此处为接口桩。

    严格约束：
        - 参数中不得出现 secret_context
        - 参数中不得出现 selected_cluster（明文 cluster 索引）
        - 只接收 SecondRoundRequest（含两个密文）
    """
    raise NotImplementedError(
        "column_wise_matching 由成员四在 party_b/online_responder.py 中实现"
    )
