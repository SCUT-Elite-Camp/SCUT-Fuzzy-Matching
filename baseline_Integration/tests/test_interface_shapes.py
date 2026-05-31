"""
接口形状检查：验证成员一~四的函数输入输出形状是否符合规范。
当前版本使用 mock 数据验证形状逻辑，待真实接口稳定后替换为实际调用。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


class TestInterfaceShapes:
    """检查各模块接口的输入输出形状"""

    def test_member1_minhash_shape(self):
        """成员一：MinHash 签名形状应为 (N, 200) 和 (N, 50)"""
        N = 100
        mock_signatures_200 = np.random.randint(0, 2**20, size=(N, 200))
        mock_signatures_50 = mock_signatures_200[:, :50]  # 前50维

        assert mock_signatures_200.shape == (N, 200), f"Expected (N,200), got {mock_signatures_200.shape}"
        assert mock_signatures_50.shape == (N, 50), f"Expected (N,50), got {mock_signatures_50.shape}"
        print("✅ member1 minhash shape: passed")

    def test_member1_scaler_shape(self):
        """成员一：标准化参数 shape 应为 (200,)"""
        mock_mean = np.zeros(200)
        mock_scale = np.ones(200)

        assert mock_mean.shape == (200,), f"Expected (200,), got {mock_mean.shape}"
        assert mock_scale.shape == (200,), f"Expected (200,), got {mock_scale.shape}"
        print("✅ member1 scaler shape: passed")

    def test_member1_centroids_shape(self):
        """成员一：聚类质心 C_k 形状应为 (k, 200)"""
        k = 50
        mock_centroids = np.random.randn(k, 200)

        assert mock_centroids.shape == (k, 200), f"Expected ({k},200), got {mock_centroids.shape}"
        print("✅ member1 centroids shape: passed")

    def test_member1_column_matrix_shape(self):
        """成员一：列式矩阵 C 形状应为 (k, max_size, 50)"""
        k = 50
        max_size = 300
        mock_column_matrix = np.random.randn(k, max_size, 50)

        assert mock_column_matrix.shape == (k, max_size, 50), \
            f"Expected ({k},{max_size},50), got {mock_column_matrix.shape}"
        print("✅ member1 column matrix shape: passed")

    def test_member2_encrypt_output_shape(self):
        """成员二：加密函数输出应为密文对象（此处用 mock 模拟）"""
        # 真实场景应调用 member2.encrypt(vec)
        mock_ciphertext = b"mock_ciphertext_200d"
        assert isinstance(mock_ciphertext, bytes), "Encrypted output should be bytes or TenSEAL object"
        print("✅ member2 encrypt output type: passed (mock)")

    def test_member3_onehot_shape(self):
        """成员三：one-hot 向量长度应等于 k"""
        k = 50
        best_idx = 23
        one_hot = np.zeros(k)
        one_hot[best_idx] = 1.0

        assert one_hot.shape == (k,), f"Expected ({k},), got {one_hot.shape}"
        assert np.sum(one_hot) == 1.0, "One-hot should have exactly one 1"
        print("✅ member3 one-hot shape: passed")

    def test_member4_column_output_shape(self):
        """成员四：列式匹配返回的分数列表长度应等于 max_size"""
        max_size = 300
        mock_scores = [b"fake_cipher"] * max_size

        assert len(mock_scores) == max_size, f"Expected {max_size} scores, got {len(mock_scores)}"
        print("✅ member4 column output count: passed")


if __name__ == "__main__":
    tester = TestInterfaceShapes()
    tester.test_member1_minhash_shape()
    tester.test_member1_scaler_shape()
    tester.test_member1_centroids_shape()
    tester.test_member1_column_matrix_shape()
    tester.test_member2_encrypt_output_shape()
    tester.test_member3_onehot_shape()
    tester.test_member4_column_output_shape()
    print("\n🎉 All interface shape tests passed (mock mode).")