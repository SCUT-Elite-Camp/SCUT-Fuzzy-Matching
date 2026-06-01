# party_b/offline_prep.py
"""
成员一 - Step 1：B 侧离线建库。

职责：
  - 对 B 的原始姓名数据库生成 MinHash 签名（EL=200）
  - 归一化、标准化
  - 运行 cosine/spherical K-Means 聚类
  - 构建列式名称矩阵 cluster_matrix（EL=50）
  - 输出 OfflineArtifacts 供后续在线阶段使用

严格约束：
  - EL=50 签名必须来自 EL=200 签名的前 50 维
  - 聚类必须使用 cosine K-Means，不得用欧氏 KMeans 结果替代
  - cluster_matrix padding 必须使用全零向量
  - empty cluster 不得产生 NaN 质心
"""

from __future__ import annotations

import pathlib
from typing import Iterable

import numpy as np
from sklearn.preprocessing import StandardScaler

from config.params import (
    NUM_PERMUTATIONS_CLUSTER,
    NUM_PERMUTATIONS_MATCH,
    KMEANS_ITERATIONS,
    choose_k,
)
from minhash.encoder import batch_encode
from preprocessing.normalizer import l2_normalize
from clustering.kmeans_cosine import run_cosine_kmeans
from protocol.protocal_types import OfflineArtifacts


# ─── 公开接口 ─────────────────────────────────────────────────────────────────

def prepare_party_b_offline(
    names_b: Iterable[str],
    k_mode: str = "sqrt",
    random_state: int = 42,
) -> OfflineArtifacts:
    """
    B 侧离线建库主入口。

    参数：
        names_b:      B 侧原始姓名列表
        k_mode:       聚类数选择模式，'sqrt'（默认）或 'paper'
        random_state: 聚类随机种子

    返回：
        OfflineArtifacts，包含：
          centroids         (k, 200)  L2 normalized 质心
          cluster_matrix    (k, max_size, 50)  列式名称矩阵，padding 为全零
          scaler_mean       (200,)    StandardScaler 均值
          scaler_scale      (200,)    StandardScaler 标准差
          cluster_assignments (n,)   每个样本所属 cluster 索引
          max_size          int       最大 cluster 大小
    """
    names_b = list(names_b)
    n = len(names_b)
    if n == 0:
        raise ValueError("names_b 不能为空")

    # ── Step 1a: 生成 EL=200 MinHash 签名 ────────────────────────────────────
    signatures_200 = batch_encode(names_b, NUM_PERMUTATIONS_CLUSTER)
    # shape: (n, 200)

    # ── Step 1b: EL=50 签名是前 50 维，不得另行生成 ──────────────────────────
    signatures_50 = signatures_200[:, :NUM_PERMUTATIONS_MATCH]
    # shape: (n, 50)

    # ── Step 1c: 分别对两个矩阵做逐行 L2 归一化 ─────────────────────────────
    normalized_200 = l2_normalize(signatures_200)   # (n, 200)
    normalized_50 = l2_normalize(signatures_50)     # (n, 50)

    # ── Step 1d: 用 StandardScaler fit B 侧 EL=200 归一化向量 ────────────────
    scaler, standardized_200, scaler_mean, scaler_scale = fit_scaler(normalized_200)

    # ── Step 1e: 选择 k，运行 cosine K-Means ─────────────────────────────────
    k = choose_k(n, mode=k_mode)
    centroids, cluster_assignments = run_cosine_kmeans(
        standardized_200, k=k, iterations=KMEANS_ITERATIONS,
        random_state=random_state,
    )
    # centroids: (k, 200), L2 normalized
    # cluster_assignments: (n,)

    # ── Step 1f: 构建列式名称矩阵（EL=50，padding 全零） ─────────────────────
    cluster_matrix, max_size = build_cluster_matrix(
        normalized_50, cluster_assignments, k
    )
    # cluster_matrix: (k, max_size, 50)

    return OfflineArtifacts(
        centroids=centroids,
        cluster_matrix=cluster_matrix,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        cluster_assignments=cluster_assignments,
        max_size=max_size,
    )


def fit_scaler(
    normalized_200: np.ndarray,
) -> tuple[StandardScaler, np.ndarray, np.ndarray, np.ndarray]:
    """
    拟合 B 侧 StandardScaler，返回 (scaler, standardized, mean, scale)。

    参数：
        normalized_200: (n, 200)，已 L2 归一化

    返回：
        scaler:         fitted StandardScaler 对象
        standardized:   (n, 200) 标准化后的矩阵
        mean:           (200,) scaler.mean_
        scale:          (200,) scaler.scale_
    """
    scaler = StandardScaler()
    standardized = scaler.fit_transform(normalized_200)
    return scaler, standardized, scaler.mean_, scaler.scale_


def build_cluster_matrix(
    normalized_50: np.ndarray,
    cluster_assignments: np.ndarray,
    k: int,
) -> tuple[np.ndarray, int]:
    """
    将 EL=50 归一化签名按 cluster 分组，构建列式名称矩阵。

    padding 规则：每个 cluster 不足 max_size 的位置填全零向量。
    全零向量不会产生正匹配（cos sim = 0 < threshold）。

    参数：
        normalized_50:       (n, 50)
        cluster_assignments: (n,)
        k:                   聚类数

    返回：
        cluster_matrix: (k, max_size, 50)
        max_size:        int，最大 cluster 大小
    """
    # 计算每个 cluster 的大小
    cluster_sizes = np.bincount(cluster_assignments, minlength=k)
    max_size = int(cluster_sizes.max()) if len(cluster_sizes) > 0 else 0

    if max_size == 0:
        return np.zeros((k, 0, normalized_50.shape[1]), dtype=np.float64), 0

    d = normalized_50.shape[1]  # 50
    cluster_matrix = np.zeros((k, max_size, d), dtype=np.float64)

    for c in range(k):
        mask = cluster_assignments == c
        members = normalized_50[mask]  # (size_c, 50)
        cluster_matrix[c, : len(members), :] = members
        # 剩余位置已由 zeros 初始化为全零向量

    return cluster_matrix, max_size


# ─── 持久化工具 ───────────────────────────────────────────────────────────────

def save_offline_artifacts(artifacts: OfflineArtifacts, save_dir: str | pathlib.Path) -> None:
    """
    将 OfflineArtifacts 保存为 .npy 文件。
    字段名与 OfflineArtifacts 保持一致。
    """
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / "centroids.npy", artifacts.centroids)
    np.save(save_dir / "cluster_matrix.npy", artifacts.cluster_matrix)
    np.save(save_dir / "scaler_mean.npy", artifacts.scaler_mean)
    np.save(save_dir / "scaler_scale.npy", artifacts.scaler_scale)
    np.save(save_dir / "cluster_assignments.npy", artifacts.cluster_assignments)
    np.save(save_dir / "max_size.npy", np.array([artifacts.max_size]))


def load_offline_artifacts(save_dir: str | pathlib.Path) -> OfflineArtifacts:
    """
    从 .npy 文件加载 OfflineArtifacts。
    """
    save_dir = pathlib.Path(save_dir)

    return OfflineArtifacts(
        centroids=np.load(save_dir / "centroids.npy"),
        cluster_matrix=np.load(save_dir / "cluster_matrix.npy"),
        scaler_mean=np.load(save_dir / "scaler_mean.npy"),
        scaler_scale=np.load(save_dir / "scaler_scale.npy"),
        cluster_assignments=np.load(save_dir / "cluster_assignments.npy"),
        max_size=int(np.load(save_dir / "max_size.npy")[0]),
    )
