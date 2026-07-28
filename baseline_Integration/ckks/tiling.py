"""CKKS 二维 query-major slot tiling 原语（V2 batching）。

布局：slot(q, t) = q * T + t
  q = query index in [0, m)
  t = logical item index in [0, T)
  S = POLY_MODULUS_DEGREE // 2
  T = floor(S / m)
  active_slots = m * T <= S

一个逻辑宽度 W 被切成 tile_count = ceil(W / T) 个 tile。
最后一个 tile 用零填充到 T；unpack 时按 valid_width 丢弃 padding。
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import tenseal as ts

from config.params import POLY_MODULUS_DEGREE
from protocol.types import SlotTileLayout

SLOT_CAPACITY = POLY_MODULUS_DEGREE // 2


def _require_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be int, got {type(value)}")
    return value


def make_slot_tile_layout(batch_size: int) -> SlotTileLayout:
    """由 batch_size 计算 V2 tiling 布局。"""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise ValueError(f"batch_size must be int, got {type(batch_size)}")
    if batch_size <= 0 or batch_size > SLOT_CAPACITY:
        raise ValueError(
            f"batch_size must be in [1, {SLOT_CAPACITY}], got {batch_size}"
        )
    tile_width = SLOT_CAPACITY // batch_size
    active_slots = batch_size * tile_width
    return SlotTileLayout(
        batch_size=batch_size,
        tile_width=tile_width,
        active_slots=active_slots,
        slot_capacity=SLOT_CAPACITY,
    )


def tile_count(logical_width: int, layout: SlotTileLayout) -> int:
    """计算逻辑宽度 W 需要多少个 tile。"""
    _require_int("logical_width", logical_width)
    if logical_width < 0:
        raise ValueError(f"logical_width must be non-negative, got {logical_width}")
    if logical_width == 0:
        return 0
    return (logical_width + layout.tile_width - 1) // layout.tile_width


def iter_tile_slices(
    logical_width: int, layout: SlotTileLayout
) -> Iterator[tuple[int, int, int]]:
    """遍历每个 tile，返回 (tile_index, start, valid_width)。

    start: 该 tile 覆盖的逻辑起始位置。
    valid_width: 该 tile 中真实逻辑项的个数，<= layout.tile_width。
    """
    _require_int("logical_width", logical_width)
    if logical_width < 0:
        raise ValueError(f"logical_width must be non-negative, got {logical_width}")
    T = layout.tile_width
    for tile_idx in range(tile_count(logical_width, layout)):
        start = tile_idx * T
        valid_width = min(T, logical_width - start)
        yield tile_idx, start, valid_width


def repeat_query_rows(matrix: np.ndarray, layout: SlotTileLayout) -> np.ndarray:
    """把 (m, d) query 矩阵按 q-major 重复 T 次 -> (m*T, d)。

    输出行顺序：
      [q0, q0, ..., q0, q1, q1, ..., q1, ...]
    其中每个 query 重复 T 次。
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {arr.shape}")
    m, _ = arr.shape
    if m != layout.batch_size:
        raise ValueError(
            f"matrix row count {m} != layout.batch_size {layout.batch_size}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("matrix contains NaN or Inf")
    return np.repeat(arr, layout.tile_width, axis=0)


def expand_plain_tile(values: np.ndarray, layout: SlotTileLayout) -> np.ndarray:
    """把一维 logical tile (valid_width,) 或 (T,) 展开到 active_slots。

    要求 values 长度为 tile_width；将其按 tile-major 顺序重复 m 次：
      [v0, v1, ..., v(T-1), v0, v1, ..., v(T-1), ...]  共 m 组
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {arr.shape}")
    if arr.shape[0] != layout.tile_width:
        raise ValueError(
            f"values length {arr.shape[0]} != tile_width {layout.tile_width}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("values contains NaN or Inf")
    return np.tile(arr, layout.batch_size)


def unpack_score_tile(
    values: np.ndarray, layout: SlotTileLayout, valid_width: int
) -> np.ndarray:
    """把解密后的 active_slots 长度一维数组 reshape 为 (m, valid_width)。

    输入顺序为 q-major：先 q0 的 T 个 slots，再 q1 的 T 个 slots...
    只保留前 valid_width 列，丢弃尾 padding。
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape[0] != layout.active_slots:
        raise ValueError(
            f"values length {arr.shape[0]} != active_slots {layout.active_slots}"
        )
    _require_int("valid_width", valid_width)
    if valid_width <= 0 or valid_width > layout.tile_width:
        raise ValueError(
            f"valid_width {valid_width} out of range (1, {layout.tile_width}]"
        )
    full = arr.reshape(layout.batch_size, layout.tile_width)
    return full[:, :valid_width].copy()


def pack_centroid_tile(
    centroids: np.ndarray, layout: SlotTileLayout, start: int, valid_width: int
) -> np.ndarray:
    """构造 centroid tile 的明文矩阵，用于 ct-pt 乘法。

    centroids shape = (k, d)。取 centroids[start:start+valid_width, :] 零填充到 (T, d)。
    返回 shape (T, d)，每一列 f 将经过 expand_plain_tile 变成 active_slots 长度
    后与 EncQ200[f] 做点积。
    """
    arr = np.asarray(centroids, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"centroids must be 2-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("centroids contains NaN or Inf")
    _validate_tile_bounds(arr.shape[0], layout, start, valid_width)
    T = layout.tile_width
    tile = np.zeros((T, arr.shape[1]), dtype=np.float64)
    tile[:valid_width, :] = arr[start : start + valid_width, :]
    return tile


def pack_selector_tile(
    selector_matrix: np.ndarray, layout: SlotTileLayout
) -> np.ndarray:
    """把 (m, k) selector 按 q-major 重复 T 次 -> (m*T, k)。

    与 repeat_query_rows 相同，但语义上是 selector 而非 query。
    """
    return repeat_query_rows(selector_matrix, layout)


def encrypt_tiled_feature_batch(
    matrix: np.ndarray, context: ts.Context, layout: SlotTileLayout
) -> list[ts.CKKSVector]:
    """把 (m, d) query/feature 矩阵加密为 d 个 active_slots 长度的密文。

    内部先把 matrix 按 q-major 重复 T 次得到 (m*T, d)，再逐列加密。
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("matrix contains NaN or Inf")

    tiled = repeat_query_rows(arr, layout)  # (m*T, d)
    d = tiled.shape[1]
    ciphertexts = []
    for f in range(d):
        vec = ts.ckks_vector(context, tiled[:, f].tolist())
        ciphertexts.append(vec)
    return ciphertexts


def decrypt_tiled_score_tile(
    ciphertext: ts.CKKSVector,
    secret_context: ts.Context,
    layout: SlotTileLayout,
    valid_width: int,
) -> np.ndarray:
    """解密一个 tiled 分数密文并 unpack 成 (m, valid_width)。"""
    if isinstance(ciphertext, bytes):
        vec = ts.ckks_vector_from(secret_context, ciphertext)
    else:
        vec = ciphertext
    if vec.size() != layout.active_slots:
        raise ValueError(
            f"ciphertext size {vec.size()} != active_slots {layout.active_slots}"
        )
    raw = np.asarray(vec.decrypt(secret_context.secret_key()), dtype=np.float64)
    return unpack_score_tile(raw, layout, valid_width)


def pack_candidate_tile(
    cluster_matrix: np.ndarray,
    layout: SlotTileLayout,
    start: int,
    valid_width: int,
) -> np.ndarray:
    """构造 candidate column tile 的明文矩阵。

    cluster_matrix shape = (k, L, d)。
    取 cluster_matrix[:, start:start+valid_width, :] 零填充到 (k, T, d)。
    """
    arr = np.asarray(cluster_matrix, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"cluster_matrix must be 3-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("cluster_matrix contains NaN or Inf")
    k, L, d = arr.shape
    _validate_tile_bounds(L, layout, start, valid_width)
    T = layout.tile_width
    tile = np.zeros((k, T, d), dtype=np.float64)
    tile[:, :valid_width, :] = arr[:, start : start + valid_width, :]
    return tile


def _validate_tile_bounds(
    logical_width: int,
    layout: SlotTileLayout,
    start: int,
    valid_width: int,
) -> None:
    _require_int("start", start)
    _require_int("valid_width", valid_width)
    if start < 0 or start >= logical_width:
        raise ValueError(
            f"start {start} out of range for logical_width {logical_width}"
        )
    if valid_width <= 0 or valid_width > layout.tile_width:
        raise ValueError(
            f"valid_width {valid_width} out of range (1, {layout.tile_width}]"
        )
    if start + valid_width > logical_width:
        raise ValueError(
            f"tile [{start}, {start + valid_width}) exceeds logical_width "
            f"{logical_width}"
        )
