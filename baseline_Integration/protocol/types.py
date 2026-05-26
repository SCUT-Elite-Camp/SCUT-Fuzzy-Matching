"""类型别名与核心数据类定义。

所有成员均从此处导入密文类型别名与协议数据结构，禁止自行发明类型名。
"""

from dataclasses import dataclass
from typing import Any, Union

import numpy as np

# ---------------------------------------------------------------------------
# 密文类型别名（协议统一术语）
# ---------------------------------------------------------------------------

# CipherBytes: 序列化后的密文字节流，用于网络传输与通信量评估。
CipherBytes = bytes

# CipherObject: TenSEAL CKKSVector 等原生密文对象，用于同进程端到端测试。
CipherObject = Any

# CipherLike: 密文联合类型：既可以是字节流，也可以是原生对象。
CipherLike = Union[CipherBytes, CipherObject]

# 不同长度的加密向量
EncryptedVector200 = CipherLike   # EL=200，用于聚类/质心匹配
EncryptedVector50 = CipherLike    # EL=50，用于列式名字匹配
EncryptedVectorK = CipherLike     # EL=K，K 为聚类块大小（最大块）
EncryptedScalar = CipherLike      # 加密标量，例如加密的余弦相似度值


# ---------------------------------------------------------------------------
# 协议数据结构
# ---------------------------------------------------------------------------

@dataclass
class OfflineArtifacts:
    """B 侧离线阶段产生的公开数据。"""
    centroids: np.ndarray
    cluster_matrix: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    cluster_assignments: np.ndarray
    max_size: int


@dataclass
class FirstRoundRequest:
    """A 侧发送给 B 的第一轮请求 payload。

    仅包含公开上下文字节流和 EL=200 的加密查询向量。
    第一轮禁止包含 encrypted_query_50。
    """
    public_context_bytes: bytes
    encrypted_query_200: EncryptedVector200


@dataclass
class PartyALocalState:
    """A 侧本地持有的状态，包含私钥上下文和预准备的 EL=50 加密向量。

    此结构留在 A 侧，不发送给 B。
    """
    secret_context: object
    encrypted_query_50: EncryptedVector50
