# minhash/encoder.py
"""
MinHash 签名生成模块。

约束：
- EL=50 签名必须是 EL=200 签名的前 50 维，不允许用不同种子单独生成。
- 使用 HASH_SEED 保证可复现性。
- shingle 基于字符级 n-gram（含空格）。
"""

import hashlib
import numpy as np
from typing import Iterable

from config.params import SHINGLE_SIZE, MAX_HASH, HASH_SEED, NUM_PERMUTATIONS_CLUSTER
from preprocessing.text_cleaner import clean_name


def _generate_shingles(name: str, shingle_size: int) -> set:
    """生成字符级 n-gram 集合。"""
    name = clean_name(name)
    if len(name) < shingle_size:
        return {name} if name else {"<empty>"}
    return {name[i: i + shingle_size] for i in range(len(name) - shingle_size + 1)}


def _hash_shingle(shingle: str) -> int:
    """将 shingle 哈希为整数，截断到 MAX_HASH 范围。"""
    h = hashlib.sha256(shingle.encode("utf-8")).hexdigest()
    return int(h, 16) % MAX_HASH


def generate_signature(name: str, num_permutations: int) -> np.ndarray:
    """
    为单个姓名生成 MinHash 签名向量。

    实现方式：使用随机线性哈希族模拟置换，
    每个置换函数为 (a * x + b) % MAX_HASH，
    其中 a, b 始终从 NUM_PERMUTATIONS_CLUSTER 个参数中取前 num_permutations 个，
    保证 EL=50 是 EL=200 的前缀。

    返回形状: (num_permutations,)，类型 float64
    """
    shingles = _generate_shingles(name, SHINGLE_SIZE)
    hashes = np.array([_hash_shingle(s) for s in shingles], dtype=np.int64)

    # 始终生成 NUM_PERMUTATIONS_CLUSTER 个参数，保证任意截断的一致性
    rng = np.random.RandomState(HASH_SEED)
    a_full = rng.randint(1, MAX_HASH, size=NUM_PERMUTATIONS_CLUSTER).astype(np.int64)
    b_full = rng.randint(0, MAX_HASH, size=NUM_PERMUTATIONS_CLUSTER).astype(np.int64)
    a = a_full[:num_permutations]
    b = b_full[:num_permutations]

    permuted = (a[:, None] * hashes[None, :] + b[:, None]) % MAX_HASH
    signature = permuted.min(axis=1).astype(np.float64)
    return signature


def batch_encode(names: Iterable[str], num_permutations: int) -> np.ndarray:
    """
    批量生成 MinHash 签名矩阵。

    返回形状: (n, num_permutations)

    关键约束：参数始终从 NUM_PERMUTATIONS_CLUSTER 个中截取，
    保证 batch_encode(names, 50) == batch_encode(names, 200)[:, :50]。
    """
    names = list(names)
    if not names:
        return np.empty((0, num_permutations), dtype=np.float64)

    # 始终生成 NUM_PERMUTATIONS_CLUSTER 个参数，取前 num_permutations 个
    rng = np.random.RandomState(HASH_SEED)
    a_full = rng.randint(1, MAX_HASH, size=NUM_PERMUTATIONS_CLUSTER).astype(np.int64)
    b_full = rng.randint(0, MAX_HASH, size=NUM_PERMUTATIONS_CLUSTER).astype(np.int64)
    a = a_full[:num_permutations]
    b = b_full[:num_permutations]

    signatures = []
    for name in names:
        shingles = _generate_shingles(name, SHINGLE_SIZE)
        hashes = np.array([_hash_shingle(s) for s in shingles], dtype=np.int64)
        if len(hashes) == 0:
            sig = np.zeros(num_permutations, dtype=np.float64)
        else:
            permuted = (a[:, None] * hashes[None, :] + b[:, None]) % MAX_HASH
            sig = permuted.min(axis=1).astype(np.float64)
        signatures.append(sig)

    return np.stack(signatures, axis=0)
