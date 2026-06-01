# tests/test_member1.py
"""
成员一模块测试。
覆盖：
  - MinHash EL50 是 EL200 的前缀
  - L2 归一化
  - StandardScaler 拟合与参数形状
  - cosine K-Means 质心归一化
  - empty cluster 不产生 NaN 质心
  - cluster_matrix 形状与零 padding
  - Step 3 质心匹配：解密后分数与明文点积近似一致
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pytest

from config.params import NUM_PERMUTATIONS_CLUSTER, NUM_PERMUTATIONS_MATCH
from minhash.encoder import batch_encode
from preprocessing.normalizer import l2_normalize
from preprocessing.text_cleaner import clean_name
from party_b.offline_prep import (
    prepare_party_b_offline,
    fit_scaler,
    build_cluster_matrix,
)
from clustering.kmeans_cosine import run_cosine_kmeans
from protocol.protocol_types import OfflineArtifacts


# ─── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_NAMES = [
    "Alice Johnson", "alice johnson", "Alicia Jonson",
    "Bob Smith", "Robert Smith", "Bobby Smith",
    "Charlie Brown", "charles brown",
    "Diana Prince", "diana prins",
    "Edward Norton", "edward nortn",
    "Fiona Green", "fiona greene",
    "George Wilson", "george willson",
    "Hannah Lee", "hana lee",
]


# ─── MinHash 测试 ──────────────────────────────────────────────────────────────

class TestMinHash:
    def test_el50_is_prefix_of_el200(self):
        """EL=50 签名必须是 EL=200 签名的前 50 维。"""
        names = SAMPLE_NAMES[:5]
        sigs_200 = batch_encode(names, NUM_PERMUTATIONS_CLUSTER)
        sigs_50_prefix = sigs_200[:, :NUM_PERMUTATIONS_MATCH]
        sigs_50_direct = batch_encode(names, NUM_PERMUTATIONS_MATCH)

        # 因为使用相同种子和相同参数，前缀必须一致
        np.testing.assert_array_equal(
            sigs_50_prefix, sigs_50_direct,
            err_msg="EL=50 签名不是 EL=200 的前 50 维，MinHash 种子不一致"
        )

    def test_output_shape(self):
        """签名矩阵形状正确。"""
        names = SAMPLE_NAMES[:6]
        sigs = batch_encode(names, NUM_PERMUTATIONS_CLUSTER)
        assert sigs.shape == (6, NUM_PERMUTATIONS_CLUSTER)

    def test_single_name_deterministic(self):
        """相同姓名生成相同签名。"""
        sigs1 = batch_encode(["Alice Johnson"], NUM_PERMUTATIONS_CLUSTER)
        sigs2 = batch_encode(["Alice Johnson"], NUM_PERMUTATIONS_CLUSTER)
        np.testing.assert_array_equal(sigs1, sigs2)

    def test_similar_names_have_high_similarity(self):
        """相似姓名的 L2 归一化签名余弦相似度应较高。"""
        sigs = batch_encode(["Alice Johnson", "alice johnson"], NUM_PERMUTATIONS_CLUSTER)
        norms = l2_normalize(sigs)
        cos_sim = float(norms[0] @ norms[1])
        assert cos_sim > 0.5, f"相似姓名相似度过低: {cos_sim}"


# ─── 归一化测试 ────────────────────────────────────────────────────────────────

class TestNormalizer:
    def test_l2_normalize_unit_norm(self):
        """L2 归一化后每行范数为 1。"""
        X = np.random.randn(10, 50).astype(np.float64)
        normed = l2_normalize(X)
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, np.ones(10), atol=1e-6)

    def test_l2_normalize_zero_vector_stays_zero(self):
        """零向量归一化后仍为零向量。"""
        X = np.zeros((3, 50))
        X[1] = np.random.randn(50)
        normed = l2_normalize(X)
        np.testing.assert_array_equal(normed[0], np.zeros(50))
        np.testing.assert_array_equal(normed[2], np.zeros(50))

    def test_2d_input_required_for_batch(self):
        """l2_normalize 在二维矩阵上工作正常。"""
        X = np.random.randn(5, 200)
        result = l2_normalize(X)
        assert result.shape == (5, 200)


# ─── StandardScaler 测试 ──────────────────────────────────────────────────────

class TestScaler:
    def test_scaler_output_shapes(self):
        """scaler_mean 和 scaler_scale 长度为 200。"""
        sigs = batch_encode(SAMPLE_NAMES, NUM_PERMUTATIONS_CLUSTER)
        normed = l2_normalize(sigs)
        _, _, mean, scale = fit_scaler(normed)
        assert mean.shape == (NUM_PERMUTATIONS_CLUSTER,)
        assert scale.shape == (NUM_PERMUTATIONS_CLUSTER,)

    def test_scaler_standardized_mean_near_zero(self):
        """标准化后每维均值近似为 0。"""
        sigs = batch_encode(SAMPLE_NAMES * 3, NUM_PERMUTATIONS_CLUSTER)
        normed = l2_normalize(sigs)
        _, standardized, _, _ = fit_scaler(normed)
        col_means = standardized.mean(axis=0)
        np.testing.assert_allclose(col_means, np.zeros(200), atol=1e-6)

    def test_scaler_standardized_std_near_one(self):
        """标准化后每维标准差近似为 1。"""
        sigs = batch_encode(SAMPLE_NAMES * 3, NUM_PERMUTATIONS_CLUSTER)
        normed = l2_normalize(sigs)
        _, standardized, _, _ = fit_scaler(normed)
        col_stds = standardized.std(axis=0)
        # 允许某些维度方差为 0（常数列），跳过
        nonzero = col_stds > 1e-8
        if nonzero.sum() > 0:
            np.testing.assert_allclose(col_stds[nonzero], np.ones(nonzero.sum()), atol=0.1)


# ─── Cosine K-Means 测试 ──────────────────────────────────────────────────────

class TestCosineKMeans:
    def _get_standardized(self, names):
        sigs = batch_encode(names, NUM_PERMUTATIONS_CLUSTER)
        normed = l2_normalize(sigs)
        _, standardized, _, _ = fit_scaler(normed)
        return standardized

    def test_centroids_are_l2_normalized(self):
        """cosine K-Means 质心应完成 L2 normalize。"""
        standardized = self._get_standardized(SAMPLE_NAMES)
        centroids, _ = run_cosine_kmeans(standardized, k=4)
        norms = np.linalg.norm(centroids, axis=1)
        np.testing.assert_allclose(norms, np.ones(4), atol=1e-5,
                                   err_msg="质心未 L2 归一化")

    def test_cluster_assignments_range(self):
        """cluster_assignments 值在 [0, k) 范围内。"""
        standardized = self._get_standardized(SAMPLE_NAMES)
        k = 4
        _, assignments = run_cosine_kmeans(standardized, k=k)
        assert assignments.min() >= 0
        assert assignments.max() < k

    def test_cluster_assignments_length(self):
        """cluster_assignments 长度等于样本数。"""
        standardized = self._get_standardized(SAMPLE_NAMES)
        _, assignments = run_cosine_kmeans(standardized, k=4)
        assert len(assignments) == len(SAMPLE_NAMES)

    def test_empty_cluster_does_not_create_nan_centroid(self):
        """empty cluster 不得产生 NaN 质心。"""
        # 用极小数据集强制产生 empty cluster
        standardized = self._get_standardized(SAMPLE_NAMES[:4])
        # k 比样本数大，必然有 empty cluster
        centroids, _ = run_cosine_kmeans(standardized, k=8)
        assert not np.any(np.isnan(centroids)), "存在 NaN 质心"
        assert not np.any(np.isinf(centroids)), "存在 Inf 质心"

    def test_centroids_shape(self):
        """centroids 形状为 (k, 200)。"""
        standardized = self._get_standardized(SAMPLE_NAMES)
        k = 4
        centroids, _ = run_cosine_kmeans(standardized, k=k)
        assert centroids.shape == (k, NUM_PERMUTATIONS_CLUSTER)


# ─── Cluster Matrix 测试 ──────────────────────────────────────────────────────

class TestClusterMatrix:
    def test_cluster_matrix_shape_and_zero_padding(self):
        """cluster_matrix 形状为 (k, max_size, 50)，padding 为全零。"""
        sigs = batch_encode(SAMPLE_NAMES, NUM_PERMUTATIONS_CLUSTER)
        normalized_50 = l2_normalize(sigs[:, :NUM_PERMUTATIONS_MATCH])

        k = 4
        n = len(SAMPLE_NAMES)
        # 人工构造 assignments
        assignments = np.arange(n) % k

        matrix, max_size = build_cluster_matrix(normalized_50, assignments, k)

        assert matrix.shape == (k, max_size, NUM_PERMUTATIONS_MATCH), \
            f"形状错误: {matrix.shape}"
        assert max_size > 0

        # 检查 padding：每个 cluster 实际大小之后的行应为全零
        for c in range(k):
            actual_size = int((assignments == c).sum())
            if actual_size < max_size:
                padding_rows = matrix[c, actual_size:, :]
                assert np.allclose(padding_rows, 0), \
                    f"Cluster {c} padding 不是全零向量"

    def test_cluster_matrix_non_padding_rows_not_all_zero(self):
        """cluster_matrix 中非 padding 行不应全为零（除非原始向量为零）。"""
        sigs = batch_encode(SAMPLE_NAMES, NUM_PERMUTATIONS_CLUSTER)
        normalized_50 = l2_normalize(sigs[:, :NUM_PERMUTATIONS_MATCH])
        k = 3
        assignments = np.arange(len(SAMPLE_NAMES)) % k
        matrix, max_size = build_cluster_matrix(normalized_50, assignments, k)

        for c in range(k):
            actual_size = int((assignments == c).sum())
            for row in range(actual_size):
                row_norm = np.linalg.norm(matrix[c, row, :])
                assert row_norm > 0.5, \
                    f"Cluster {c} row {row} 为零向量，但原始向量应有效"


# ─── 端到端建库测试 ────────────────────────────────────────────────────────────

class TestOfflinePrep:
    def test_prepare_party_b_offline_output_shapes(self):
        """prepare_party_b_offline 输出所有字段形状正确。"""
        artifacts = prepare_party_b_offline(SAMPLE_NAMES, k_mode="sqrt")

        n = len(SAMPLE_NAMES)
        k = artifacts.centroids.shape[0]

        assert artifacts.centroids.shape == (k, 200), \
            f"centroids 形状错误: {artifacts.centroids.shape}"
        assert artifacts.cluster_matrix.shape == (k, artifacts.max_size, 50), \
            f"cluster_matrix 形状错误: {artifacts.cluster_matrix.shape}"
        assert artifacts.scaler_mean.shape == (200,)
        assert artifacts.scaler_scale.shape == (200,)
        assert artifacts.cluster_assignments.shape == (n,)
        assert isinstance(artifacts.max_size, int)

    def test_centroids_are_normalized(self):
        """所有质心应完成 L2 归一化。"""
        artifacts = prepare_party_b_offline(SAMPLE_NAMES)
        norms = np.linalg.norm(artifacts.centroids, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-5)

    def test_cluster_matrix_padding_is_zero(self):
        """cluster_matrix 的 padding 位置全为零向量。"""
        artifacts = prepare_party_b_offline(SAMPLE_NAMES)
        k = artifacts.centroids.shape[0]
        assignments = artifacts.cluster_assignments
        matrix = artifacts.cluster_matrix

        for c in range(k):
            actual_size = int((assignments == c).sum())
            if actual_size < artifacts.max_size:
                padding = matrix[c, actual_size:, :]
                assert np.allclose(padding, 0), \
                    f"Cluster {c} padding 不是全零向量"

    def test_scaler_params_not_nan(self):
        """scaler_mean 和 scaler_scale 不含 NaN 或 Inf。"""
        artifacts = prepare_party_b_offline(SAMPLE_NAMES)
        assert not np.any(np.isnan(artifacts.scaler_mean))
        assert not np.any(np.isinf(artifacts.scaler_mean))
        assert not np.any(np.isnan(artifacts.scaler_scale))
        assert not np.any(np.isinf(artifacts.scaler_scale))


# ─── Step 3 质心匹配测试（需要 TenSEAL）────────────────────────────────────────

class TestStep3:
    """
    测试 compare_to_centroids 的正确性：
    解密后的相似度分数应与明文点积近似一致。
    """

    @pytest.fixture(autouse=True)
    def check_tenseal(self):
        pytest.importorskip("tenseal", reason="TenSEAL 未安装，跳过 Step 3 测试")

    def _make_first_round_request(self, query_200_std, artifacts):
        import tenseal as ts
        from ckks.context import create_ckks_context, get_public_context_bytes
        from protocol.types import FirstRoundRequest
        from ckks.operations import encrypt_vector_packed

        ctx = create_ckks_context()
        pub_bytes = get_public_context_bytes(ctx)
        enc_q = encrypt_vector_packed(ctx, query_200_std)

        return (
            FirstRoundRequest(
                public_context_bytes=pub_bytes,
                encrypted_query_200=enc_q,
            ),
            ctx,
        )

    def test_ct_pt_dot_matches_plain_dot_after_decrypt(self):
        """解密后质心相似度应与明文点积近似一致。"""
        import tenseal as ts
        from party_b.online_responder import compare_to_centroids
        from ckks.context import create_ckks_context, get_public_context_bytes
        from protocol.types import FirstRoundRequest
        from ckks.operations import encrypt_vector_packed, decrypt_scalar

        artifacts = prepare_party_b_offline(SAMPLE_NAMES)

        # 构造查询向量（模拟成员二的输出）
        query_sigs = batch_encode(["Alice Johnson"], NUM_PERMUTATIONS_CLUSTER)
        query_norm = l2_normalize(query_sigs)[0]  # (200,)
        # 用 B 侧 scaler 标准化
        query_std = (query_norm - artifacts.scaler_mean) / artifacts.scaler_scale

        ctx = create_ckks_context()
        pub_bytes = get_public_context_bytes(ctx)
        enc_q = encrypt_vector_packed(ctx, query_std)

        req = FirstRoundRequest(
            public_context_bytes=pub_bytes,
            encrypted_query_200=enc_q,
        )

        # B 侧计算加密相似度
        enc_scores = compare_to_centroids(req, artifacts.centroids, serialize_output=False)

        # 明文点积
        plain_scores = artifacts.centroids @ query_std  # (k,)

        # 解密并比较
        k = artifacts.centroids.shape[0]
        assert len(enc_scores) == k

        for i, enc_score in enumerate(enc_scores):
            decrypted = decrypt_scalar(ctx, enc_score)
            plain = float(plain_scores[i])
            assert abs(decrypted - plain) < 0.01, \
                f"Centroid {i}: decrypted={decrypted:.6f}, plain={plain:.6f}, diff={abs(decrypted-plain):.6f}"

    def test_first_round_request_does_not_include_query_50(self):
        """FirstRoundRequest 不包含 encrypted_query_50。"""
        from protocol.types import FirstRoundRequest
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(FirstRoundRequest)}
        assert "encrypted_query_50" not in field_names, \
            "FirstRoundRequest 不应包含 encrypted_query_50"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
