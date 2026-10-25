"""离线聚类索引（复用生产同款参数，numpy 复刻）。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.evaluate_cluster_conditional_filter import cosine_kmeans


@dataclass
class ClusterIndex:
    assignments: np.ndarray   # (n,) 每条库记录所属簇
    centroids: np.ndarray     # (k, 200)
    sizes: np.ndarray         # (k,)
    max_size: int


def build_cluster_index(db_std: np.ndarray, k: int) -> ClusterIndex:
    centroids, assignments = cosine_kmeans(db_std, k)
    sizes = np.bincount(assignments, minlength=k)
    return ClusterIndex(assignments=assignments, centroids=centroids,
                        sizes=sizes, max_size=int(sizes.max()))
