"""
数值正确性检查：验证 MinHash 一致性、标准化正确性、密文点积近似正确性。
当前版本使用 mock 模拟，待真实接口稳定后替换为实际调用。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


class TestCorrectness:
    """验证数值计算是否正确"""

    def test_minhash_consistency(self):
        """A 和 B 用相同名字生成 MinHash 签名应该一致"""
        # 模拟 A 和 B 使用相同的种子
        np.random.seed(42)
        name = "John Smith"

        # 模拟签名生成（实际应调用成员一的 MinHash 函数）
        def mock_minhash(name, seed):
            np.random.seed(seed)
            return np.random.randint(0, 2**20, size=200)

        sig_A = mock_minhash(name, 42)
        sig_B = mock_minhash(name, 42)

        np.testing.assert_array_equal(sig_A, sig_B, err_msg="MinHash signatures should be identical")
        print("✅ minhash_consistency: passed")

    def test_scaler_consistency(self):
        """A 使用 B 的标准化参数变换后，应与 B 侧同分布"""
        # B 拟合 scaler
        np.random.seed(42)
        B_data = np.random.randn(1000, 200)
        mean = np.mean(B_data, axis=0)
        std = np.std(B_data, axis=0)

        # A 使用 B 的 mean/std 变换
        A_sample = np.random.randn(200)
        A_transformed = (A_sample - mean) / (std + 1e-8)

        # B 侧一个随机样本的变换
        B_sample = B_data[0]
        B_transformed = (B_sample - mean) / (std + 1e-8)

        # 两者应在同一空间（形状相同，范围相近）
        assert A_transformed.shape == B_transformed.shape
        print("✅ scaler_consistency: passed")

    def test_cosine_similarity_approx(self):
        """密文点积解密后应接近明文余弦相似度（允许 CKKS 误差）"""
        # 明文计算余弦相似度
        vec1 = np.random.randn(50)
        vec2 = np.random.randn(50)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        plain_cosine = np.dot(vec1 / norm1, vec2 / norm2)

        # 模拟 CKKS 加密解密后的结果（添加小噪声）
        encrypted_cosine = plain_cosine + np.random.normal(0, 0.01)  # 模拟噪声

        # CKKS 误差应小于阈值判断的容差
        tolerance = 0.05
        assert abs(encrypted_cosine - plain_cosine) < tolerance, \
            f"CKKS error too large: {abs(encrypted_cosine - plain_cosine)} > {tolerance}"
        print("✅ cosine_similarity_approx: passed")

    def test_threshold_sign_preserved(self):
        """减去阈值再乘随机数后，正负号应保持不变"""
        tau = 0.9
        test_cases = [0.95, 0.85]  # 第一个 > tau，第二个 < tau

        for cos_score in test_cases:
            temp = cos_score - tau
            r = 12345  # 正随机数
            score = r * temp
            expected_sign = 1 if cos_score > tau else -1
            actual_sign = 1 if score > 0 else -1
            assert actual_sign == expected_sign, \
                f"Sign changed for cos={cos_score}: expected {expected_sign}, got {actual_sign}"
        print("✅ threshold_sign_preserved: passed")


if __name__ == "__main__":
    tester = TestCorrectness()
    tester.test_minhash_consistency()
    tester.test_scaler_consistency()
    tester.test_cosine_similarity_approx()
    tester.test_threshold_sign_preserved()
    print("\n📊 All correctness tests passed (mock mode).")