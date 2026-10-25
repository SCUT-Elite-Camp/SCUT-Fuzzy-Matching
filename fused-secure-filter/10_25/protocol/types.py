"""类型别名与核心数据类定义。

所有成员均从此处导入密文类型别名与协议数据结构，禁止自行发明类型名。
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from config.params import POLY_MODULUS_DEGREE

# ---------------------------------------------------------------------------
# 密文类型别名（协议统一术语）
# ---------------------------------------------------------------------------

# CipherBytes: 序列化后的密文字节流，用于网络传输与通信量评估。
CipherBytes = bytes

# CipherObject: TenSEAL CKKSVector 等原生密文对象，用于同进程端到端测试。
CipherObject = Any

# CipherLike: 密文联合类型：既可以是字节流，也可以是原生对象。
CipherLike = CipherBytes | CipherObject

# 不同长度的加密向量
EncryptedVector200 = CipherLike  # EL=200，用于聚类/质心匹配
EncryptedVector50 = CipherLike  # EL=50，用于列式名字匹配
EncryptedVectorK = CipherLike  # EL=K，K 为聚类块大小（最大块）
EncryptedScalar = CipherLike  # 加密标量，例如加密的余弦相似度值


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


@dataclass
class SecondRoundRequest:
    """A 侧发送给 B 的第二轮请求 payload。"""

    encrypted_query_50: EncryptedVector50
    encrypted_selector: EncryptedVectorK


@dataclass
class ClusterSelectionDebug:
    """A 侧本地保留的 cluster 选择调试信息。"""

    selected_cluster: int


@dataclass
class MatchResult:
    """生产接口对外只暴露是否命中。"""

    catch: bool


@dataclass
class MatchDebug:
    """A 侧本地调试信息，不得对外暴露。"""

    checked_columns: int
    first_positive_column: int | None


# ---------------------------------------------------------------------------
# 批量协议数据结构 (Batch Protocol Data Structures)
# ---------------------------------------------------------------------------


@dataclass
class BatchFirstRoundRequest:
    public_context_bytes: bytes
    encrypted_query_200: list[CipherLike]  # 长度 200，每个密文 m slots
    batch_size: int


@dataclass
class BatchPartyALocalState:
    secret_context: object
    encrypted_query_50: list[CipherLike]  # 长度 50，每个密文 m slots
    batch_size: int


@dataclass
class BatchSecondRoundRequest:
    encrypted_query_50: list[CipherLike]  # 长度 50
    encrypted_selectors: list[CipherLike]  # 长度 k
    batch_size: int


@dataclass
class BatchClusterSelectionDebug:
    selected_clusters: np.ndarray  # (m,)


@dataclass
class BatchMatchResult:
    catches: np.ndarray  # bool, (m,)


@dataclass
class BatchMatchDebug:
    checked_columns: int
    first_positive_columns: np.ndarray  # int, (m,), 未命中为 -1


# ---------------------------------------------------------------------------
# 二维 Slot Tiling 批量协议数据结构 (V2 Tiled Batch Protocol Data Structures)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotTileLayout:
    """query-major 2-D slot tiling layout metadata."""

    batch_size: int
    tile_width: int
    active_slots: int
    slot_capacity: int = POLY_MODULUS_DEGREE // 2

    def __post_init__(self) -> None:
        fields = {
            "batch_size": self.batch_size,
            "tile_width": self.tile_width,
            "active_slots": self.active_slots,
            "slot_capacity": self.slot_capacity,
        }
        for name, value in fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an int, got {type(value)}")

        expected_capacity = POLY_MODULUS_DEGREE // 2
        if self.slot_capacity != expected_capacity:
            raise ValueError(
                f"slot_capacity must be {expected_capacity}, got {self.slot_capacity}"
            )
        if self.batch_size <= 0 or self.batch_size > self.slot_capacity:
            raise ValueError(
                f"batch_size must be in [1, {self.slot_capacity}], "
                f"got {self.batch_size}"
            )
        expected_width = self.slot_capacity // self.batch_size
        if self.tile_width != expected_width:
            raise ValueError(
                f"tile_width {self.tile_width} inconsistent with batch_size "
                f"{self.batch_size}; expected {expected_width}"
            )
        expected_active = self.batch_size * self.tile_width
        if self.active_slots != expected_active:
            raise ValueError(
                f"active_slots {self.active_slots} inconsistent with layout; "
                f"expected {expected_active}"
            )


@dataclass
class TiledFirstRoundRequest:
    """V2 第一轮请求：query-major tiling 的 EncQ200 + 公开 context。"""

    public_context_bytes: bytes
    encrypted_query_200: list[CipherLike]  # 长度 200，每个密文 active_slots
    layout: SlotTileLayout


@dataclass
class TiledPartyALocalState:
    """V2 A 侧本地状态：EncQ50 保留在 A，不发给 B。"""

    secret_context: object
    encrypted_query_50: list[CipherLike]  # 长度 50，每个密文 active_slots
    layout: SlotTileLayout


@dataclass
class TiledSecondRoundRequest:
    """V2 第二轮请求：EncQ50 + k 个 encrypted selector（每个 active_slots）。"""

    encrypted_query_50: list[CipherLike]  # 长度 50
    encrypted_selectors: list[CipherLike]  # 长度 k
    layout: SlotTileLayout


@dataclass
class TiledClusterSelectionDebug:
    """V2 A 侧本地 cluster 选择调试信息。"""

    selected_clusters: np.ndarray  # (m,)


@dataclass
class TiledMatchDebug:
    """V2 A 侧本地命中调试信息。"""

    checked_tiles: int
    logical_columns_checked: int
    first_positive_columns: np.ndarray  # int, (m,), 未命中为 -1


@dataclass
class TiledMatchResult:
    """V2 批量命中生产结果。"""

    catches: np.ndarray  # bool, (m,)


@dataclass
class TiledProtocolRun:
    """V2 完整两轮协议运行的对外结果。"""

    match_result: TiledMatchResult
    match_debug: TiledMatchDebug
    cluster_debug: TiledClusterSelectionDebug
    round1_ciphertext_count: int
    round2_ciphertext_count: int
