# clustering/kmeans_cosine.py
"""
Cosine/Spherical K-Means 实现。

约束（来自接口规范）：
1. 不得直接用 sklearn KMeans 的欧氏距离结果替代最终聚类语义。
2. 每轮分配必须按 cosine similarity 选择最近质心。
3. 每轮更新质心时先取 cluster 内向量均值，再做 L2 normalize。
4. empty cluster 处理：保留上一轮质心；若无上一轮，用当前误差最大的样本重置。
5. 返回的 centroids 必须是 L2 normalized 的。
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing.normalizer import l2_normalize
from config.params import KMEANS_ITERATIONS


def _init_centroids(X: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    """
    随机初始化 k 个质心（从 X 中随机采样，并 L2 normalize）。
    """
    indices = rng.choice(len(X), size=min(k, len(X)), replace=False)
    centroids = X[indices].copy()
    return l2_normalize(centroids)


def run_cosine_kmeans(
    standardized_200: np.ndarray,
    k: int,
    iterations: int = KMEANS_ITERATIONS,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    在标准化后的 EL=200 向量上运行 cosine/spherical K-Means。

    参数：
        standardized_200: 形状 (n, 200)，已经过 StandardScaler 处理
        k: 聚类数
        iterations: 最大迭代次数
        random_state: 随机种子

    返回：
        centroids: 形状 (k, 200)，L2 normalized
        cluster_assignments: 形状 (n,)，每个样本所属 cluster 的索引
    """
    n = len(standardized_200)
    if k > n:
        k = n

    rng = np.random.RandomState(random_state)

    # 初始化质心
    centroids = _init_centroids(standardized_200, k, rng)
    prev_centroids = None
    cluster_assignments = np.zeros(n, dtype=np.int32)

    for iteration in range(iterations):
        prev_centroids = centroids.copy()

        # 步骤1：按 cosine similarity 分配每个样本到最近质心
        # sim_matrix: (n, k)
        sim_matrix = cosine_similarity(standardized_200, centroids)
        cluster_assignments = np.argmax(sim_matrix, axis=1).astype(np.int32)

        # 步骤2：更新质心
        new_centroids = np.zeros_like(centroids)
        empty_clusters = []

        for c in range(k):
            mask = cluster_assignments == c
            if mask.sum() == 0:
                empty_clusters.append(c)
                continue
            # 取 cluster 内向量均值，再 L2 normalize
            mean_vec = standardized_200[mask].mean(axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm < 1e-10:
                # 均值为零向量，保留上一轮质心
                new_centroids[c] = prev_centroids[c]
            else:
                new_centroids[c] = mean_vec / norm

        # 处理 empty clusters
        for c in empty_clusters:
            # 计算每个样本到其当前质心的余弦距离（1 - cosine_sim）
            assigned_centroid_sims = sim_matrix[np.arange(n), cluster_assignments]
            cos_distances = 1.0 - assigned_centroid_sims
            # 找误差最大的样本
            farthest_idx = np.argmax(cos_distances)
            new_centroids[c] = standardized_200[farthest_idx].copy()
            norm = np.linalg.norm(new_centroids[c])
            if norm > 1e-10:
                new_centroids[c] /= norm
            else:
                # fallback：随机从 X 选一个
                idx = rng.randint(0, n)
                new_centroids[c] = standardized_200[idx].copy()
                norm2 = np.linalg.norm(new_centroids[c])
                if norm2 > 1e-10:
                    new_centroids[c] /= norm2

        centroids = new_centroids

        # 收敛检测
        centroid_shift = np.max(np.linalg.norm(centroids - prev_centroids, axis=1))
        if centroid_shift < 1e-8:
            break

    # 最终重新分配，保证 cluster_assignments 与 centroids 一致
    sim_matrix = cosine_similarity(standardized_200, centroids)
    cluster_assignments = np.argmax(sim_matrix, axis=1).astype(np.int32)

    return centroids, cluster_assignments
