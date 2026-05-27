"""MinHash 签名生成模块（Algorithm 2）。

基于论文 Privacy-Preserving Fuzzy Name Matching 的规范实现：
  - trigram shingle（含边界空格）
  - SHA-256 mod 2^20
  - 200 个线性置换取最小值
  - 支持批量编码
"""

import hashlib
import struct
from typing import Iterable

import numpy as np

from config.params import HASH_SEED, MAX_HASH, NUM_PERMUTATIONS_CLUSTER
from preprocessing.text_cleaner import clean_name

# ---------------------------------------------------------------------------
# 置换函数系数（全局单例，延迟初始化）
# ---------------------------------------------------------------------------
_permutation_params: "np.ndarray | None" = None  # shape (200, 2), dtype=uint64


def _get_permutation_params(num_perms: int = NUM_PERMUTATIONS_CLUSTER) -> np.ndarray:
    """获取线性置换系数 (a_j, b_j)。

    使用固定种子生成，保证 Party A 与 Party B 的置换完全一致。
    π_j(x) = (a_j * x + b_j) mod MAX_HASH

    Args:
        num_perms: 置换函数数量，默认 200。

    Returns:
        形状 (num_perms, 2) 的 uint64 数组，[:,0] 为 a，[:,1] 为 b。
    """
    # Why: 始终按最大数量 NUM_PERMUTATIONS_CLUSTER 一次生成并切片前缀返回，
    # 否则 b_vals 在 rng 中的起点会随 num_perms 变化，破坏
    # signature(name, 50) == signature(name, 200)[:50] 这一协议不变量。
    global _permutation_params
    if _permutation_params is None:
        rng = np.random.default_rng(HASH_SEED)
        full = NUM_PERMUTATIONS_CLUSTER
        a_vals = rng.integers(1, MAX_HASH, size=full, dtype=np.uint64)
        a_vals |= 1  # a 必须为奇数，与 MAX_HASH 互质保证置换为双射
        b_vals = rng.integers(0, MAX_HASH, size=full, dtype=np.uint64)
        _permutation_params = np.column_stack((a_vals, b_vals))
    return _permutation_params[:num_perms]


# ---------------------------------------------------------------------------
# 公共接口
# ---------------------------------------------------------------------------

def build_shingles(name: str, size: int = 3) -> set[str]:
    """为单个名字生成 n-gram（shingle）集合，含边界空格。

    根据规范，名字前后各加一个空格后再滑窗切分。
    例："John" → " John " → {" Jo", "Joh", "ohn", "hn "}

    Args:
        name: 已清洗或原始名字字符串。
        size: shingle 长度，默认 3（trigram）。

    Returns:
        不重复的 shingle 字符串集合。
    """
    cleaned = clean_name(name)
    # 添加边界空格
    padded = f" {cleaned} "
    if len(padded) < size:
        return set()
    return {padded[i : i + size] for i in range(len(padded) - size + 1)}


def _hash_shingle(shingle: str) -> int:
    """对单个 shingle 计算 SHA-256 并取模 MAX_HASH。

    编码使用 UTF-8，与 Python 默认一致。
    """
    digest = hashlib.sha256(shingle.encode("utf-8")).digest()
    # 取前 4 字节转 uint32，再对 MAX_HASH 取模
    val = struct.unpack("<I", digest[:4])[0]
    return int(val % MAX_HASH)


def generate_signature(name: str, num_perms: int) -> np.ndarray:
    """为单个名字生成 MinHash 签名向量。

    步骤：
      1. 生成 shingle 集合
      2. 每个 shingle → SHA-256 mod MAX_HASH
      3. 对每个置换 π_j，取所有哈希值经 π_j 变换后的最小值

    Args:
        name: 名字字符串。
        num_perms: 签名长度（置换函数数量）。

    Returns:
        形状 (num_perms,) 的 float32 numpy 数组。
        若名字过短无法生成 shingle，返回全零向量。
    """
    shingles = build_shingles(name)
    if not shingles:
        return np.zeros(num_perms, dtype=np.float32)

    # 计算所有 shingle 的哈希值
    hash_vals = np.fromiter(
        (_hash_shingle(s) for s in shingles), dtype=np.uint64, count=len(shingles)
    )

    # 获取置换系数（前 num_perms 个）
    params = _get_permutation_params(num_perms)
    a = params[:, 0].reshape(-1, 1)  # (num_perms, 1)
    b = params[:, 1].reshape(-1, 1)  # (num_perms, 1)

    # 广播计算所有置换值: (num_perms, n_shingles)
    permuted = (a * hash_vals + b) % MAX_HASH

    # 每行取最小值得到 MinHash 签名
    sig = permuted.min(axis=1).astype(np.float32)
    return sig


def batch_encode(names: Iterable[str], num_perms: int) -> np.ndarray:
    """批量生成 MinHash 签名矩阵。

    Args:
        names: 名字列表或可迭代对象。
        num_perms: 签名长度。

    Returns:
        形状 (N, num_perms) 的 float32 numpy 数组，N 为名字数量。
    """
    names_list = list(names)
    if not names_list:
        return np.empty((0, num_perms), dtype=np.float32)

    signatures = [generate_signature(name, num_perms) for name in names_list]
    return np.vstack(signatures)
