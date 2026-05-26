"""
实验流程测试骨架：用 mock 数据模拟整个实验过程，验证指标计算、通信统计等。
后续替换为真实模块调用。
"""

# 添加项目根目录到路径
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  

from evaluation.metrics import compute_precision, compute_recall, compute_f1
from evaluation.communication_cost import CommStats, measure_ct_size as measure_size
from evaluation.benchmark import timer, StageTimer
from tests.mock_data import (
    mock_minhash_signatures,
    mock_normalized_signatures,
    mock_cluster_matrix,
    mock_centroids,
    mock_query_signatures
)

# 模拟查询方 A 和响应方 B 的交互流程（用 mock 数据）
class MockResponder:
    """模拟 B 方的在线响应（不涉及真正的加密）"""
    def __init__(self, centroids, cluster_matrix):
        self.centroids = centroids          # (k, el_cluster)
        self.cluster_matrix = cluster_matrix  # (k, max_size, el_match)
        self.comm = CommStats()

    def compare_to_centroids(self, query_normalized_scaled):
        """模拟第一轮：计算与各质心的余弦相似度（明文）"""
        # query_normalized_scaled: (el_cluster,)
        scores = []
        for c in self.centroids:
            sim = np.dot(query_normalized_scaled, c) / (np.linalg.norm(query_normalized_scaled) * np.linalg.norm(c) + 1e-8)
            scores.append(sim)
        # 模拟通信：发送加密后的相似度列表（用 pickled 大小近似）
        self.comm.add_sent(scores)   # 假设发送给 A
        return np.array(scores)

    def column_wise_matching(self, query_normalized, one_hot_encrypted):
        """
        模拟第二轮：逐列计算相似度。
        实际中 one_hot_encrypted 是加密的，这里用明文 one-hot 模拟。
        """
        scores = []
        # one_hot_encrypted 是明文 one-hot 向量 (k,)
        best_cluster_idx = np.argmax(one_hot_encrypted)  # 找出最佳簇
        max_cols = self.cluster_matrix.shape[1]
        for col in range(max_cols):
            # 取出该列中最佳簇对应的签名
            selected_name = self.cluster_matrix[best_cluster_idx, col, :]  # (el_match,)
            if np.all(selected_name == 0):  # 填充的零向量
                sim = -1.0   # 产生负分数
            else:
                sim = np.dot(query_normalized, selected_name) / (np.linalg.norm(query_normalized) * np.linalg.norm(selected_name) + 1e-8)
            scores.append(sim)
        # 模拟通信：每列返回一个加密分数（此处简化为全部返回）
        for s in scores:
            self.comm.add_recv(s)
        return scores

def run_mock_experiment():
    """运行一次完整的 mock 实验，输出指标和统计"""
    # 参数设置（示例）
    num_names_B = 10000
    el_cluster = 200
    el_match = 50
    k = int(np.sqrt(num_names_B))  # ≈100
    max_cols = 300   # 模拟最大簇大小

    # 生成 mock 数据
    print("生成 mock 数据...")
    centroids = mock_centroids(k, el_cluster)
    cluster_matrix = mock_cluster_matrix(k, max_cols, el_match)
    responder = MockResponder(centroids, cluster_matrix)

    # 模拟查询（假设 A 有一个查询名字）
    query_raw = mock_query_signatures(el_cluster)
    # 这里需要标准化（假设已经通过 B 的 scaler 处理，用 mock 直接模拟）
    query_norm_scaled = query_raw   # 简化：直接用原始

    # 第一轮：质心匹配
    print("第一轮：质心匹配...")
    sim_scores = responder.compare_to_centroids(query_norm_scaled)
    best_idx = np.argmax(sim_scores)
    print(f"  -> 最佳质心索引: {best_idx}, 最高相似度: {sim_scores[best_idx]:.4f}")

    # 模拟 A 构造 one-hot 向量并加密（此处直接传递 one-hot 明文）
    one_hot = np.zeros(k)
    one_hot[best_idx] = 1.0

    # 第二轮：列式匹配
    print("第二轮：列式匹配...")
    query_match = mock_query_signatures(el_match)   # 使用 el=50 的查询签名
    col_scores = responder.column_wise_matching(query_match, one_hot)

    # 判断匹配：阈值 τ = 0.9
    tau = 0.9
    catch = any(s > tau for s in col_scores)
    print(f"匹配结果: {'有潜在匹配' if catch else '无匹配'}")
    print(f"通信统计: {responder.comm}")

    # 这里因为没有真实 ground truth，仅演示流程。真实实验时需要标注 TP/FP 等。

if __name__ == "__main__":
    run_mock_experiment()