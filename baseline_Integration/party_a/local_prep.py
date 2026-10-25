"""成员二：A 侧查询预处理与加密模块。

负责：
1. 对查询姓名做 MinHash 编码、L2 归一化和标准化
2. 创建 CKKS 上下文并加密查询向量
3. 构造第一轮请求和本地状态
"""

# Imports below the path bootstrap are intentional for direct script execution.
# ruff: noqa: E402

import sys
from collections.abc import Sequence
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径设置：确保可以导入 baseline_Integration 内部模块
# ---------------------------------------------------------------------------
_INTEGRATION_DIR = Path(__file__).resolve().parent.parent
if str(_INTEGRATION_DIR) not in sys.path:
    sys.path.insert(0, str(_INTEGRATION_DIR))

# ---------------------------------------------------------------------------
# 第三方与项目导入
# ---------------------------------------------------------------------------
import numpy as np
import tenseal as ts

from ckks.batching import encrypt_feature_batch
from ckks.context import create_ckks_context as _create_ckks_context
from ckks.keys import encrypt, serialize_public_context
from ckks.tiling import encrypt_tiled_feature_batch, make_slot_tile_layout
from config.params import (
    NUM_PERMUTATIONS_CLUSTER,
    NUM_PERMUTATIONS_MATCH,
    POLY_MODULUS_DEGREE,
)
from clustering.projection import project_normalized_signatures
from minhash.encoder import batch_encode
from preprocessing.normalizer import l2_normalize
from protocol.types import (
    BatchFirstRoundRequest,
    BatchPartyALocalState,
    FirstRoundRequest,
    PartyALocalState,
    TiledFirstRoundRequest,
    TiledPartyALocalState,
)


def _transform_cluster_query(
    query_200_norm: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    projection_dim: int | None,
    projection_seed: int,
) -> np.ndarray:
    if projection_dim is not None:
        return project_normalized_signatures(
            query_200_norm, projection_dim, projection_seed
        )

    mean = np.asarray(scaler_mean, dtype=np.float64)
    scale = np.asarray(scaler_scale, dtype=np.float64)
    if mean.shape != (NUM_PERMUTATIONS_CLUSTER,) or scale.shape != (
        NUM_PERMUTATIONS_CLUSTER,
    ):
        raise ValueError(
            f"scaler 参数形状不匹配: 期望 ({NUM_PERMUTATIONS_CLUSTER},)，"
            f"实际 mean {mean.shape}, scale {scale.shape}"
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError("scaler 参数包含 NaN 或 Inf 非法数值")
    if np.any(scale <= 0):
        raise ValueError("scaler_scale must contain only positive values")
    return (query_200_norm - mean) / scale


def encode_query_vectors(
    query_name: str,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    *,
    projection_dim: int | None = None,
    projection_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """生成聚类查询向量与原 50 维精判向量。"""
    try:
        query_200_raw = batch_encode([query_name], NUM_PERMUTATIONS_CLUSTER)
        query_50_raw = query_200_raw[:, :NUM_PERMUTATIONS_MATCH].copy()
        query_200_norm_2d = l2_normalize(query_200_raw)
        query_50_norm_2d = l2_normalize(query_50_raw)
        query_cluster = _transform_cluster_query(
            query_200_norm_2d,
            scaler_mean,
            scaler_scale,
            projection_dim,
            projection_seed,
        )
        return query_cluster[0], query_50_norm_2d[0]
    except Exception as e:
        raise RuntimeError(
            f"Failed to encode query vectors for name '{query_name}': {e}"
        ) from e


def create_ckks_context() -> ts.Context:
    """创建 CKKS 上下文（从 ckks.context 模块重新导出）。

    Returns:
        ts.Context: 已配置 Galois 密钥的 CKKS 上下文。
    """
    return _create_ckks_context()


def encrypt_query_vectors(
    query_200_std: np.ndarray,
    query_50_norm: np.ndarray,
    context: ts.Context,
) -> tuple[FirstRoundRequest, PartyALocalState]:
    """加密两个查询向量并构造协议数据结构。

    步骤：
        1. 生成 relinearization keys（同态乘法所需）
        2. 加密 query_200_std 和 query_50_norm
        3. 序列化公开上下文（不含私钥）
        4. 构造 FirstRoundRequest（第一轮网络 payload）和 PartyALocalState（A 侧本地持有）

    Args:
        query_200_std: 聚类查询向量；默认形状 (200,)，投影时为指定维度。
        query_50_norm: L2 归一化后的 EL=50 查询向量，形状 (50,)。
        context: 含私钥的 CKKS 上下文。

    Returns:
        tuple[FirstRoundRequest, PartyALocalState]:
            - FirstRoundRequest: 第一轮发送给 B 的请求，只含 public_context_bytes 和 encrypted_query_200。
            - PartyALocalState: A 侧本地状态，含 secret_context 和 encrypted_query_50。
    """
    try:
        # 生成重线性化密钥：同态乘法后密文维度升高，需此密钥降维
        context.generate_relin_keys()

        # 加密两个查询向量（返回 TenSEAL CKKSVector 对象）
        encrypted_query_200 = encrypt(query_200_std, context)
        encrypted_query_50 = encrypt(query_50_norm, context)

        # 序列化公开上下文：复制后移除私钥再序列化，确保不含 secret key
        public_context_bytes = serialize_public_context(context)

        first_round_req = FirstRoundRequest(
            public_context_bytes=public_context_bytes,
            encrypted_query_200=encrypted_query_200,
        )

        local_state = PartyALocalState(
            secret_context=context,
            encrypted_query_50=encrypted_query_50,
        )

        return first_round_req, local_state
    except Exception as e:
        raise RuntimeError(f"Failed to encrypt query vectors: {e}") from e


def prepare_encrypted_query(
    query_name: str,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    *,
    projection_dim: int | None = None,
    projection_seed: int = 42,
) -> tuple[FirstRoundRequest, PartyALocalState]:
    """A 侧查询准备主入口：编码、创建上下文、加密。"""
    query_cluster, query_50_norm = encode_query_vectors(
        query_name,
        scaler_mean,
        scaler_scale,
        projection_dim=projection_dim,
        projection_seed=projection_seed,
    )
    context = create_ckks_context()
    return encrypt_query_vectors(query_cluster, query_50_norm, context)


def prepare_encrypted_query_with_context(
    query_name: str,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    context: ts.Context,
    public_context_bytes: bytes | None = None,
    *,
    projection_dim: int | None = None,
    projection_seed: int = 42,
) -> tuple[FirstRoundRequest, PartyALocalState, bytes]:
    """Prepare a query using an existing A-side CKKS context."""
    query_cluster, query_50_norm = encode_query_vectors(
        query_name,
        scaler_mean,
        scaler_scale,
        projection_dim=projection_dim,
        projection_seed=projection_seed,
    )
    encrypted_query_200 = encrypt(query_cluster, context)
    encrypted_query_50 = encrypt(query_50_norm, context)
    if public_context_bytes is None:
        public_context_bytes = serialize_public_context(context)

    first_round_req = FirstRoundRequest(
        public_context_bytes=public_context_bytes,
        encrypted_query_200=encrypted_query_200,
    )
    local_state = PartyALocalState(
        secret_context=context,
        encrypted_query_50=encrypted_query_50,
    )
    return first_round_req, local_state, public_context_bytes


def prepare_tiled_query_batch(
    query_names: Sequence[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    *,
    context: ts.Context | None = None,
    public_context_bytes: bytes | None = None,
    projection_dim: int | None = None,
    projection_seed: int = 42,
) -> tuple[TiledFirstRoundRequest, TiledPartyALocalState]:
    """V2 二维 tiling 批量查询准备主入口。"""
    if context is None and public_context_bytes is not None:
        raise ValueError(
            "public_context_bytes cannot be supplied without its private context"
        )

    query_cluster, query_50_norm = encode_query_batch(
        query_names,
        scaler_mean,
        scaler_scale,
        projection_dim=projection_dim,
        projection_seed=projection_seed,
    )
    m = len(query_names)
    layout = make_slot_tile_layout(m)

    if context is None:
        context = create_ckks_context()
    if not isinstance(context, ts.Context) or not context.is_private():
        raise ValueError("context must be a private TenSEAL context owned by Party A")

    if not context.has_relin_keys():
        context.generate_relin_keys()
    expected_public_context = serialize_public_context(context)
    if public_context_bytes is None:
        public_context_bytes = expected_public_context
    else:
        if not isinstance(public_context_bytes, bytes):
            raise ValueError("public_context_bytes must be bytes")
        if public_context_bytes != expected_public_context:
            raise ValueError(
                "public_context_bytes does not match the supplied private context"
            )

    encrypted_query_200 = encrypt_tiled_feature_batch(query_cluster, context, layout)
    encrypted_query_50 = encrypt_tiled_feature_batch(query_50_norm, context, layout)

    first_round_req = TiledFirstRoundRequest(
        public_context_bytes=public_context_bytes,
        encrypted_query_200=encrypted_query_200,
        layout=layout,
    )

    local_state = TiledPartyALocalState(
        secret_context=context,
        encrypted_query_50=encrypted_query_50,
        layout=layout,
        query_50_norm=query_50_norm,
    )

    return first_round_req, local_state


def encode_query_batch(
    query_names: Sequence[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    *,
    projection_dim: int | None = None,
    projection_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """批量生成聚类查询矩阵与原 50 维精判矩阵。"""
    if len(query_names) == 0:
        raise ValueError("query_names 列表不能为空")

    m = len(query_names)
    max_batch = POLY_MODULUS_DEGREE // 2
    if m > max_batch:
        raise ValueError(f"batch_size {m} 超过最大允许值 {max_batch}")

    try:
        query_200_raw = batch_encode(list(query_names), NUM_PERMUTATIONS_CLUSTER)
        query_50_raw = query_200_raw[:, :NUM_PERMUTATIONS_MATCH].copy()
        query_200_norm_2d = l2_normalize(query_200_raw)
        query_50_norm_2d = l2_normalize(query_50_raw)
        query_cluster = _transform_cluster_query(
            query_200_norm_2d,
            scaler_mean,
            scaler_scale,
            projection_dim,
            projection_seed,
        )
        return query_cluster, query_50_norm_2d
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise RuntimeError(f"批量编码查询向量失败: {e}") from e


def prepare_encrypted_query_batch(
    query_names: Sequence[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    *,
    context: ts.Context | None = None,
    public_context_bytes: bytes | None = None,
    projection_dim: int | None = None,
    projection_seed: int = 42,
) -> tuple[BatchFirstRoundRequest, BatchPartyALocalState]:
    """A 侧批量查询准备主入口：编码 → 创建/共享上下文 → 纵向 SIMD 加密。

    Args:
        query_names: 查询名字序列 (m 条)。
        scaler_mean: B 侧 StandardScaler mean_ (200,)。
        scaler_scale: B 侧 StandardScaler scale_ (200,)。
        context: 可选现有 CKKS 上下文（含私钥）。
        public_context_bytes: 可选已有公开上下文字节。

    Returns:
        tuple[BatchFirstRoundRequest, BatchPartyALocalState]:
            - BatchFirstRoundRequest: 第一轮请求 payload。
            - BatchPartyALocalState: A 侧本地状态。
    """
    query_200_std, query_50_norm = encode_query_batch(
        query_names,
        scaler_mean,
        scaler_scale,
        projection_dim=projection_dim,
        projection_seed=projection_seed,
    )
    m = len(query_names)

    if context is None:
        context = create_ckks_context()

    if public_context_bytes is None:
        # 只在需要新建公开 context 时生成一次 relin keys。
        # 分块评估传入 context + public_context_bytes 时不重复生成。
        context.generate_relin_keys()
        public_context_bytes = serialize_public_context(context)

    encrypted_query_200 = encrypt_feature_batch(query_200_std, context)
    encrypted_query_50 = encrypt_feature_batch(query_50_norm, context)

    first_round_req = BatchFirstRoundRequest(
        public_context_bytes=public_context_bytes,
        encrypted_query_200=encrypted_query_200,
        batch_size=m,
    )

    local_state = BatchPartyALocalState(
        secret_context=context,
        encrypted_query_50=encrypted_query_50,
        batch_size=m,
    )

    return first_round_req, local_state
