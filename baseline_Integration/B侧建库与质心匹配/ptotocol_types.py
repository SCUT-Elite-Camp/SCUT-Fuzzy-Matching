# protocol/types.py
"""
协议公共类型定义。
所有成员必须使用此处定义的类型别名，不得自行发明密文类型名。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterator
import numpy as np

# ─── 密文类型别名 ────────────────────────────────────────────────────────────
try:
    from typing import TypeAlias
except ImportError:
    TypeAlias = Any  # Python < 3.10 兼容

CipherBytes: TypeAlias = bytes
CipherObject: TypeAlias = Any
CipherLike: TypeAlias = Any  # CipherBytes | CipherObject

EncryptedVector200: TypeAlias = CipherLike
EncryptedVector50: TypeAlias = CipherLike
EncryptedVectorK: TypeAlias = CipherLike
EncryptedScalar: TypeAlias = CipherLike


# ─── 数据结构 ─────────────────────────────────────────────────────────────────

@dataclass
class OfflineArtifacts:
    """B 侧离线建库产物（Step 1 输出）。"""
    centroids: np.ndarray          # (k, 200)
    cluster_matrix: np.ndarray     # (k, max_size, 50)
    scaler_mean: np.ndarray        # (200,)
    scaler_scale: np.ndarray       # (200,)
    cluster_assignments: np.ndarray  # (n,)
    max_size: int


@dataclass
class FirstRoundRequest:
    """
    第一轮网络请求，唯一允许发给 B 侧的内容。
    只包含 public context 和 encrypted_query_200，
    不得包含 encrypted_query_50。
    """
    public_context_bytes: bytes
    encrypted_query_200: EncryptedVector200


@dataclass
class PartyALocalState:
    """
    A 侧本地状态，不得交给 B 侧。
    保存 secret_context 和第二轮使用的 encrypted_query_50。
    """
    secret_context: Any
    encrypted_query_50: EncryptedVector50


@dataclass
class SecondRoundRequest:
    """
    第二轮网络请求，唯一允许交给 B 侧的第二轮内容。
    不含 selected_cluster（明文 cluster 索引）。
    """
    encrypted_query_50: EncryptedVector50
    encrypted_selector: EncryptedVectorK


@dataclass
class ClusterSelectionDebug:
    """仅用于 A 侧调试、测试、日志，不得进入 B 侧接口。"""
    selected_cluster: int


@dataclass
class MatchResult:
    """对外暴露的最终匹配结果，只包含 catch。"""
    catch: bool


@dataclass
class MatchDebug:
    """
    A 侧 debug 信息，只能用于本地 benchmark/debug/测试。
    不得放进 production MatchResult，避免泄露 early stop 列位置。
    """
    checked_columns: int
    first_positive_column: int | None = None
