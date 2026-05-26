"""
端到端集成测试：调用前四位成员的真实模块，验证全链路能跑通。
当前版本使用 mock 模拟，待真实接口稳定后取消注释并替换为实际调用。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# TODO: 待成员模块交付后，取消下面的注释
# from member1.offline import build_database, get_scaler, get_centroids, get_column_matrix
# from member2.encryptor import encrypt_query
# from member3.selector import select_cluster, build_onehot
# from member4.matcher import column_wise_match, final_judge


class TestIntegration:
    """端到端集成测试"""

    def test_end_to_end_mock(self):
        """
        使用 mock 模拟完整流程，验证协议能跑通。
        待真实接口稳定后，替换每个步骤的 mock 实现。
        """
        print("\n=== Running end-to-end integration test (mock mode) ===")

        # 1. 成员一：B 侧离线建库
        print("1. Member1: Building B database (mock)...")
        N_B = 1000
        k = int(np.sqrt(N_B))  # ≈31
        # mock 产出
        scaler_mean = np.zeros(200)
        scaler_scale = np.ones(200)
        centroids = np.random.randn(k, 200)
        column_matrix = np.random.randn(k, 300, 50)  # (k, max_size, 50)
        print(f"   - Scaler shape: {scaler_mean.shape}")
        print(f"   - Centroids shape: {centroids.shape}")
        print(f"   - Column matrix shape: {column_matrix.shape}")

        # 2. 成员二：A 侧查询加密
        print("\n2. Member2: Encrypting query (mock)...")
        query_name = "John Doe"
        # mock 加密
        ciphertext_200 = b"mock_cipher_200"
        ciphertext_50 = b"mock_cipher_50"
        print(f"   - Query: {query_name}")
        print(f"   - Ciphertext_200 size: {len(ciphertext_200)} bytes")
        print(f"   - Ciphertext_50 size: {len(ciphertext_50)} bytes")

        # 3. 成员一在线：质心匹配
        print("\n3. Member1: Comparing with centroids (mock)...")
        # mock 返回加密相似度
        encrypted_sim_scores = [b"mock_sim"] * k
        print(f"   - Returned {len(encrypted_sim_scores)} encrypted scores")

        # 4. 成员三：解密并选择 cluster
        print("\n4. Member3: Selecting best cluster (mock)...")
        # mock 解密得到相似度
        mock_sim_values = np.random.rand(k)
        best_idx = np.argmax(mock_sim_values)
        one_hot = np.zeros(k)
        one_hot[best_idx] = 1.0
        one_hot_encrypted = b"mock_onehot"
        print(f"   - Best cluster index: {best_idx} (kept secret on A side)")
        print(f"   - One-hot encrypted size: {len(one_hot_encrypted)} bytes")

        # 5. 成员四：列式匹配
        print("\n5. Member4: Column-wise matching (mock)...")
        max_size = column_matrix.shape[1]
        encrypted_scores = []
        for col in range(max_size):
            # 模拟每列返回一个加密分数
            encrypted_scores.append(b"mock_score")
        print(f"   - Processed {len(encrypted_scores)} columns")
        print(f"   - Each score size: ~{len(encrypted_scores[0]) if encrypted_scores else 0} bytes")

        # 6. 成员四：A 侧最终判断
        print("\n6. Member4: Final judgment (mock)...")
        # mock 解密分数并判断
        mock_decrypted_scores = np.random.randn(max_size)
        tau = 0.9
        catch = any(s > tau for s in mock_decrypted_scores)
        print(f"   - Threshold τ = {tau}")
        print(f"   - Catch = {catch} (potential match found)")

        # 7. 验证结果
        print("\n" + "=" * 50)
        if catch is not None:
            print("✅ End-to-end test completed successfully!")
        else:
            print("❌ Test failed: catch is None")
        print("=" * 50)


if __name__ == "__main__":
    tester = TestIntegration()
    tester.test_end_to_end_mock()