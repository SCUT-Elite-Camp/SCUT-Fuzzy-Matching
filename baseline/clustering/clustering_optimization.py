"""
Party B 离线：
  1. EL=200 MinHash signature
  2. L2 normalize
  3. StandardScaler
  4. KMeans 聚类
  5. EL=50 signature 构建列矩阵

Party A 在线：
  1. 加密 query200，发给 B
  2. B 密文比较 query 与所有 centroids
  3. A 解密质心分数，选择最相似 cluster
  4. A 加密 one-hot sign vector 和 query50
  5. B 执行密文列匹配
  6. A 解密每列 score，只根据正负判断 catch
"""

import os
import json
import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config.params import (
    NUM_PERMUTATIONS_CLUSTER,
    NUM_PERMUTATIONS_MATCH,
    K_CLUSTERS_FUNC,
    KMEANS_ITERATIONS,
    SIMILARITY_THRESHOLD,
    ARTIFACTS_DIR,
)

from preprocessing.normalization import l2_normalize
from baseline.minhash_encoding.minhash.encoder import batch_encode

# 按你的真实路径修改
from crypto.ckks import encrypt, decrypt


@dataclass
class ClusteringArtifacts:
    centroids: np.ndarray
    assignments: np.ndarray
    scaler: StandardScaler
    col_matrix: np.ndarray
    mask: np.ndarray
    names: list[str]


def fit_scaler(matrix: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(matrix)
    return scaler


def transform(matrix: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(matrix)


def run_kmeans(
    matrix: np.ndarray,
    k: int | None = None,
    max_iter: int = KMEANS_ITERATIONS,
    seed: int = 42,
):
    matrix = np.asarray(matrix, dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError("matrix 必须是二维数组，shape=(N, d)")

    n_samples = matrix.shape[0]

    if n_samples == 0:
        raise ValueError("matrix 不能为空")

    if k is None:
        k = K_CLUSTERS_FUNC(n_samples)

    k = max(1, min(int(k), n_samples))

    matrix_norm = l2_normalize(matrix)

    scaler = fit_scaler(matrix_norm)
    matrix_scaled = transform(matrix_norm, scaler)

    kmeans = KMeans(
        n_clusters=k,
        max_iter=max_iter,
        random_state=seed,
        n_init=10,
    )

    assignments = kmeans.fit_predict(matrix_scaled)
    centroids = kmeans.cluster_centers_.astype(np.float32)

    return centroids, assignments, scaler


def build_column_matrix(
    sig50: np.ndarray,
    assignments: np.ndarray,
):
    sig50 = np.asarray(sig50, dtype=np.float32)
    assignments = np.asarray(assignments)

    if sig50.ndim != 2:
        raise ValueError("sig50 必须是二维数组，shape=(N, 50)")

    if len(sig50) != len(assignments):
        raise ValueError("sig50 和 assignments 的样本数不一致")

    sig50_norm = l2_normalize(sig50)

    k = int(assignments.max()) + 1

    clusters = [
        sig50_norm[assignments == cluster_id]
        for cluster_id in range(k)
    ]

    max_cluster_size = max(len(cluster) for cluster in clusters)
    dim = sig50_norm.shape[1]

    col_matrix = np.zeros(
        (k, max_cluster_size, dim),
        dtype=np.float32,
    )

    mask = np.zeros(
        (k, max_cluster_size),
        dtype=bool,
    )

    for cluster_id, cluster in enumerate(clusters):
        size = len(cluster)

        if size == 0:
            continue

        col_matrix[cluster_id, :size, :] = cluster
        mask[cluster_id, :size] = True

    return col_matrix, mask


def compare_to_centroids_plain(
    query200: np.ndarray,
    centroids: np.ndarray,
    scaler: StandardScaler,
):
    query200 = np.asarray(query200, dtype=np.float32).reshape(1, -1)

    query_norm = l2_normalize(query200)
    query_scaled = scaler.transform(query_norm)

    scores = query_scaled @ centroids.T
    scores = scores.reshape(-1)

    best_cluster_id = int(np.argmax(scores))

    return best_cluster_id, scores


def column_wise_match_plain(
    query50: np.ndarray,
    col_matrix: np.ndarray,
    mask: np.ndarray,
    cluster_id: int,
    tau: float = SIMILARITY_THRESHOLD,
):
    query50 = np.asarray(query50, dtype=np.float32).reshape(1, -1)
    query50_norm = l2_normalize(query50)[0]

    candidates = col_matrix[cluster_id]
    valid_mask = mask[cluster_id]

    scores = candidates @ query50_norm
    scores = scores[valid_mask]

    matched = scores >= tau
    matched_positions = np.where(matched)[0]

    return {
        "catch": bool(np.any(matched)),
        "scores": scores,
        "matched_positions": matched_positions,
    }


def encrypted_compare_to_centroids(
    ct_query200,
    centroids: np.ndarray,
):
    """
    Party B 执行：
    E(query200) 与明文 centroids 做 CT-PT dot product。
    """
    encrypted_scores = []

    for centroid in centroids:
        centroid = np.asarray(centroid, dtype=np.float64)
        ct_score = ct_query200.dot(centroid.tolist())
        encrypted_scores.append(ct_score)

    return encrypted_scores


def decrypt_encrypted_scores(encrypted_scores, sk) -> np.ndarray:
    """
    Party A 执行：
    解密 B 返回的加密质心分数。
    """
    scores = []

    for ct_score in encrypted_scores:
        score = decrypt(ct_score, sk)[0]
        scores.append(score)

    return np.array(scores, dtype=np.float64)


def encrypted_column_wise_match(
    ct_query50_parts: list,
    ct_sign,
    col_matrix: np.ndarray,
    tau: float = SIMILARITY_THRESHOLD,
    random_low: int = 1,
    random_high: int = 100,
):
    """
    Party B 执行密文列式匹配。

    ct_query50_parts:
        query50 的每一维单独加密，长度为 50。

    ct_sign:
        加密 one-hot cluster indicator，长度为 k。

    col_matrix:
        shape=(k, max_cluster_size, 50)

    返回：
        每一列一个加密 score。
        score = random_positive * (cosine_similarity - tau)
    """
    k, max_cluster_size, dim = col_matrix.shape

    if len(ct_query50_parts) != dim:
        raise ValueError("ct_query50_parts 长度必须等于 col_matrix 的最后一维")

    encrypted_column_scores = []
    rng = np.random.default_rng()

    for col_id in range(max_cluster_size):

        encrypted_products = []

        for d in range(dim):
            column_dim_values = col_matrix[:, col_id, d].astype(np.float64)

            # E(S) dot column_dim_values
            # 选出目标 cluster 在当前列、当前维度的值
            ct_selected_dim = ct_sign.dot(column_dim_values.tolist())

            # CT-CT 乘法
            ct_product = ct_selected_dim * ct_query50_parts[d]

            encrypted_products.append(ct_product)

        ct_cos_score = encrypted_products[0]

        for ct_product in encrypted_products[1:]:
            ct_cos_score = ct_cos_score + ct_product

        ct_score = ct_cos_score - float(tau)

        r = int(rng.integers(random_low, random_high))
        ct_score = ct_score * float(r)

        encrypted_column_scores.append(ct_score)

    return encrypted_column_scores


class ClusteringOptimizer:
    def __init__(
        self,
        k: int | None = None,
        tau: float = SIMILARITY_THRESHOLD,
        max_iter: int = KMEANS_ITERATIONS,
        seed: int = 42,
    ):
        self.k = k
        self.tau = tau
        self.max_iter = max_iter
        self.seed = seed

        self.centroids = None
        self.assignments = None
        self.scaler = None
        self.col_matrix = None
        self.mask = None
        self.names = None

    def fit(self, names_B: list[str]):
        self.names = list(names_B)

        if len(self.names) == 0:
            raise ValueError("names_B 不能为空")

        sig200 = batch_encode(
            self.names,
            NUM_PERMUTATIONS_CLUSTER,
        )

        sig50 = batch_encode(
            self.names,
            NUM_PERMUTATIONS_MATCH,
        )

        self.centroids, self.assignments, self.scaler = run_kmeans(
            matrix=sig200,
            k=self.k,
            max_iter=self.max_iter,
            seed=self.seed,
        )

        self.col_matrix, self.mask = build_column_matrix(
            sig50=sig50,
            assignments=self.assignments,
        )

        return self

    def search_plain(self, query_name: str):
        """
        明文版本，方便对照测试。
        """
        self._check_fitted()

        query200 = batch_encode(
            [query_name],
            NUM_PERMUTATIONS_CLUSTER,
        )[0]

        query50 = batch_encode(
            [query_name],
            NUM_PERMUTATIONS_MATCH,
        )[0]

        cluster_id, centroid_scores = compare_to_centroids_plain(
            query200=query200,
            centroids=self.centroids,
            scaler=self.scaler,
        )

        result = column_wise_match_plain(
            query50=query50,
            col_matrix=self.col_matrix,
            mask=self.mask,
            cluster_id=cluster_id,
            tau=self.tau,
        )

        cluster_global_indices = np.where(
            self.assignments == cluster_id
        )[0]

        matched_global_indices = cluster_global_indices[
            result["matched_positions"]
        ]

        matched_names = [
            self.names[i]
            for i in matched_global_indices
        ]

        result.update({
            "query_name": query_name,
            "cluster_id": cluster_id,
            "centroid_scores": centroid_scores,
            "matched_global_indices": matched_global_indices,
            "matched_names": matched_names,
        })

        return result

    def encrypted_search(self, query_name: str, ctx, sk):
        """
        加密版本，更贴近论文流程。
        """
        self._check_fitted()

        # =========================
        # Party A: encode query
        # =========================
        query200 = batch_encode(
            [query_name],
            NUM_PERMUTATIONS_CLUSTER,
        )

        query50 = batch_encode(
            [query_name],
            NUM_PERMUTATIONS_MATCH,
        )

        # query200 用于质心比较：L2 + scaler
        query200_norm = l2_normalize(query200)
        query200_scaled = self.scaler.transform(query200_norm)[0]

        # query50 用于列式匹配：只做 L2
        query50_norm = l2_normalize(query50)[0]

        # =========================
        # Party A: encrypt query200
        # =========================
        ct_query200 = encrypt(
            query200_scaled.astype(np.float64),
            ctx,
        )

        # =========================
        # Party B: encrypted centroid comparison
        # =========================
        encrypted_centroid_scores = encrypted_compare_to_centroids(
            ct_query200=ct_query200,
            centroids=self.centroids,
        )

        # =========================
        # Party A: decrypt centroid scores
        # =========================
        centroid_scores = decrypt_encrypted_scores(
            encrypted_centroid_scores,
            sk,
        )

        cluster_id = int(np.argmax(centroid_scores))

        # =========================
        # Party A: encrypt one-hot sign vector
        # =========================
        k = self.centroids.shape[0]

        sign_vector = np.zeros(k, dtype=np.float64)
        sign_vector[cluster_id] = 1.0

        ct_sign = encrypt(
            sign_vector,
            ctx,
        )

        # =========================
        # Party A: encrypt query50 by dimension
        # =========================
        ct_query50_parts = [
            encrypt(np.array([float(v)], dtype=np.float64), ctx)
            for v in query50_norm
        ]

        # =========================
        # Party B: encrypted column-wise matching
        # =========================
        encrypted_column_scores = encrypted_column_wise_match(
            ct_query50_parts=ct_query50_parts,
            ct_sign=ct_sign,
            col_matrix=self.col_matrix,
            tau=self.tau,
        )

        # =========================
        # Party A: decrypt column scores
        # =========================
        decrypted_scores = []

        for ct_score in encrypted_column_scores:
            score = decrypt(ct_score, sk)[0]
            decrypted_scores.append(score)

        decrypted_scores = np.array(decrypted_scores, dtype=np.float64)

        matched_positions = np.where(decrypted_scores > 0)[0]
        catch = bool(len(matched_positions) > 0)

        cluster_global_indices = np.where(
            self.assignments == cluster_id
        )[0]

        matched_positions = matched_positions[
            matched_positions < len(cluster_global_indices)
        ]

        matched_global_indices = cluster_global_indices[
            matched_positions
        ]

        matched_names = [
            self.names[i]
            for i in matched_global_indices
        ]

        return {
            "query_name": query_name,
            "catch": catch,
            "cluster_id": cluster_id,
            "centroid_scores": centroid_scores,
            "encrypted_column_scores": encrypted_column_scores,
            "decrypted_column_scores": decrypted_scores,
            "matched_positions": matched_positions,
            "matched_global_indices": matched_global_indices,
            "matched_names": matched_names,
        }

    def save(self, save_dir: str = ARTIFACTS_DIR):
        self._check_fitted()

        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, "centroids.npy"), self.centroids)
        np.save(os.path.join(save_dir, "assignments.npy"), self.assignments)
        np.save(os.path.join(save_dir, "col_matrix.npy"), self.col_matrix)
        np.save(os.path.join(save_dir, "mask.npy"), self.mask)

        with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

        with open(os.path.join(save_dir, "names.json"), "w", encoding="utf-8") as f:
            json.dump(self.names, f, ensure_ascii=False, indent=2)

        metadata = {
            "k": int(self.centroids.shape[0]),
            "tau": float(self.tau),
            "num_permutations_cluster": NUM_PERMUTATIONS_CLUSTER,
            "num_permutations_match": NUM_PERMUTATIONS_MATCH,
            "max_cluster_size": int(self.col_matrix.shape[1]),
            "num_names": int(len(self.names)),
        }

        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, save_dir: str = ARTIFACTS_DIR):
        optimizer = cls()

        optimizer.centroids = np.load(os.path.join(save_dir, "centroids.npy"))
        optimizer.assignments = np.load(os.path.join(save_dir, "assignments.npy"))
        optimizer.col_matrix = np.load(os.path.join(save_dir, "col_matrix.npy"))
        optimizer.mask = np.load(os.path.join(save_dir, "mask.npy"))

        with open(os.path.join(save_dir, "scaler.pkl"), "rb") as f:
            optimizer.scaler = pickle.load(f)

        with open(os.path.join(save_dir, "names.json"), "r", encoding="utf-8") as f:
            optimizer.names = json.load(f)

        return optimizer

    def get_cluster_sizes(self):
        self._check_fitted()
        return np.bincount(self.assignments)

    def get_coverage_curve(self):
        self._check_fitted()

        valid_per_column = self.mask.sum(axis=0)
        covered_names = np.cumsum(valid_per_column)
        columns = np.arange(1, len(covered_names) + 1)

        return columns, covered_names

    def _check_fitted(self):
        if self.centroids is None:
            raise RuntimeError("ClusteringOptimizer 尚未 fit 或 load")
