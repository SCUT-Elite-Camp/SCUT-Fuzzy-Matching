"""
生成模拟数据，用于提前测试实验流程（不依赖真实模块）
"""
import numpy as np

def mock_minhash_signatures(num_samples: int, el: int = 200, random_seed: int = 42):
    """
    模拟 MinHash 签名输出（任务1）
    返回形状 (num_samples, el) 的 numpy 数组，值在 [0, 2^20) 范围内
    """
    np.random.seed(random_seed)
    return np.random.randint(0, 2**20, size=(num_samples, el)).astype(np.float64)

def mock_normalized_signatures(num_samples: int, el: int = 200, random_seed: int = 42):
    """
    模拟 L2 归一化后的签名（输出范围为 [-1, 1] 左右）
    """
    np.random.seed(random_seed)
    raw = np.random.randn(num_samples, el).astype(np.float64)
    # L2 归一化
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    normalized = raw / (norms + 1e-8)
    return normalized

def mock_cluster_matrix(k: int, max_size: int, el: int = 50, random_seed: int = 42):
    """
    模拟列式聚类矩阵 C (任务4)
    形状: (k, max_size, el)
    """
    np.random.seed(random_seed)
    # 用随机向量填充
    mat = np.random.randn(k, max_size, el).astype(np.float64)
    # L2 归一化每个向量
    norms = np.linalg.norm(mat, axis=2, keepdims=True)
    mat = mat / (norms + 1e-8)
    return mat

def mock_centroids(k: int, el: int = 200, random_seed: int = 42):
    """模拟质心矩阵"""
    np.random.seed(random_seed)
    centroids = np.random.randn(k, el).astype(np.float64)
    # 标准化（模拟 StandardScaler 后的结果）
    centroids = (centroids - np.mean(centroids, axis=0)) / (np.std(centroids, axis=0) + 1e-8)
    return centroids

def mock_query_signatures(el: int = 200, random_seed: int = 42):
    """模拟单个查询的签名"""
    np.random.seed(random_seed)
    raw = np.random.randn(el).astype(np.float64)
    # L2 归一化
    norm = np.linalg.norm(raw)
    normalized = raw / (norm + 1e-8)
    return normalized