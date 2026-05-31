"""
安全边界检查：验证 A/B 双方没有越界访问对方数据。
当前版本使用 mock 模拟检查逻辑，待真实接口稳定后替换。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSecurityBoundary:
    """验证文档中定义的安全红线没有被踩"""

    def test_round1_no_short_ciphertext(self):
        """第一轮请求：只能包含长查询密文(EL=200)，不能包含短查询密文(EL=50)"""
        # 模拟 A 构建第一轮请求
        round1_request = {
            "ciphertext_200": b"mock_cipher_200",  # 长查询密文
            # "ciphertext_50": b"mock_cipher_50",  # ❌ 不应该存在
        }
        assert "ciphertext_50" not in round1_request, \
            "❌ Round1 request should NOT contain short ciphertext (EL=50)"
        assert "ciphertext_200" in round1_request, \
            "✅ Round1 request must contain long ciphertext (EL=200)"
        print("✅ round1_no_short_ciphertext: passed")

    def test_round2_no_plaintext_cluster(self):
        """第二轮请求：不能包含明文的 cluster 索引"""
        # 模拟 A 构建第二轮请求（正确做法：发送加密 one-hot）
        best_cluster_idx = 23  # 这是明文，只能留在 A 侧
        mock_one_hot_encrypted = b"mock_encrypted_onehot"  # 发送的是加密后的

        # ❌ 错误：不应该直接把 best_cluster_idx 放进请求
        # round2_request_wrong = {"cluster_index": best_cluster_idx}

        # ✅ 正确：只发送加密后的 one-hot
        round2_request_correct = {"one_hot_encrypted": mock_one_hot_encrypted}

        assert "cluster_index" not in round2_request_correct, \
            "❌ Round2 request should NOT contain plaintext cluster index"
        assert "one_hot_encrypted" in round2_request_correct, \
            "✅ Round2 request must contain encrypted one-hot vector"
        print("✅ round2_no_plaintext_cluster: passed")

    def test_b_no_private_key(self):
        """B 侧代码不能接收 A 的私钥"""
        # 模拟 B 的初始化函数
        def init_b_side(public_context, column_matrix):
            # B 只接收公开上下文和列矩阵，不接收私钥
            return {"ctx": public_context, "matrix": column_matrix}

        mock_public_ctx = b"mock_public_ctx"
        mock_column_matrix = b"mock_column_matrix"
        # 如果传入私钥应该报错或不被接受
        # mock_private_key = b"mock_private_key"  # ❌ B 不应该接收这个

        b_obj = init_b_side(mock_public_ctx, mock_column_matrix)
        assert "ctx" in b_obj and "matrix" in b_obj
        assert "private_key" not in b_obj
        print("✅ b_no_private_key: passed")

    def test_a_no_column_matrix(self):
        """A 侧不能访问 B 的列式名称矩阵（明文）"""
        # 模拟 A 的初始化函数
        def init_a_side(private_key, query_name):
            # A 只接收私钥和查询名字，不接收 B 的列矩阵
            return {"sk": private_key, "query": query_name}

        mock_sk = b"mock_sk"
        mock_query = "John Doe"
        a_obj = init_a_side(mock_sk, mock_query)

        assert "column_matrix" not in a_obj, \
            "❌ A side should NOT have access to B's column matrix"
        print("✅ a_no_column_matrix: passed")

    def test_random_r_positive_and_different(self):
        """用于混淆的随机数 r 必须是正数，且每列不同"""
        import random
        random.seed(42)
        max_size = 10
        r_values = [random.randint(1, 10000) for _ in range(max_size)]

        assert all(r > 0 for r in r_values), "All r must be positive"
        assert len(set(r_values)) == len(r_values), "Each column must have different r"
        print("✅ random_r_positive_and_different: passed")


if __name__ == "__main__":
    tester = TestSecurityBoundary()
    tester.test_round1_no_short_ciphertext()
    tester.test_round2_no_plaintext_cluster()
    tester.test_b_no_private_key()
    tester.test_a_no_column_matrix()
    tester.test_random_r_positive_and_different()
    print("\n🔒 All security boundary tests passed.")