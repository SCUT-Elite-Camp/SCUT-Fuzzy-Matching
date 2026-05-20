"""
数据集加载：NCVR、图书馆目录、US Census
若文件不存在，则生成模拟数据用于测试。
"""

import numpy as np
import os

def load_dataset(name, path):
    """
    Args:
        name: str, 数据集名称，可选 "ncvr", "libcat", "census"
        path: str, 数据集文件所在目录或文件路径

    Returns:
        tuple: (names_A, names_B, labels)
            names_A: list of str, 查询方名单
            names_B: list of str, 响应方名单
            labels: list of bool/ int, 对于每个查询，是否存在真实匹配（长度等于names_A）
    """
    if name.lower() == "ncvr":
        return _load_ncvr(path)
    elif name.lower() == "libcat":
        return _load_libcat(path)
    elif name.lower() == "census":
        return _load_census(path)
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose from 'ncvr', 'libcat', 'census'.")

def _load_ncvr(path):
    """
    模拟 NCVR 数据加载。
    实际使用时根据 NCVR CSV 文件格式修改。
    """
    # 如果真实文件存在，则读取；否则生成模拟数据
    if os.path.exists(path):
        # TODO: 根据实际 CSV 格式读取
        # 这里提供占位代码
        import pandas as pd
        df = pd.read_csv(path)
        # 假设有列 'name_A', 'name_B', 'match_label'
        names_A = df['name_A'].tolist()
        names_B = df['name_B'].tolist()
        labels = df['match_label'].astype(bool).tolist()
        return names_A, names_B, labels
    else:
        print(f"NCVR data not found at {path}, generating mock data for testing.")
        # 生成模拟数据
        np.random.seed(42)
        n_queries = 100
        n_db = 1000
        names_A = [f"Person_{i}" for i in range(n_queries)]
        names_B = [f"Person_{j}" for j in range(n_db)]
        # 随机生成一些匹配标签（前10个为匹配）
        labels = [i < 10 for i in range(n_queries)]
        return names_A, names_B, labels

def _load_libcat(path):
    """模拟图书馆目录数据加载"""
    if os.path.exists(path):
        # TODO: 根据实际格式读取
        import pandas as pd
        df = pd.read_csv(path)
        names_A = df['title_A'].tolist()
        names_B = df['title_B'].tolist()
        labels = df['match_label'].astype(bool).tolist()
        return names_A, names_B, labels
    else:
        print(f"LibCat data not found at {path}, generating mock data.")
        np.random.seed(123)
        n_queries = 80
        n_db = 2000
        names_A = [f"BookTitle_{i}" for i in range(n_queries)]
        names_B = [f"BookTitle_{j}" for j in range(n_db)]
        labels = [i < 8 for i in range(n_queries)]
        return names_A, names_B, labels

def _load_census(path):
    """模拟美国普查数据加载"""
    if os.path.exists(path):
        # TODO: 根据实际格式读取
        import pandas as pd
        df = pd.read_csv(path)
        names_A = df['name_A'].tolist()
        names_B = df['name_B'].tolist()
        labels = df['match_label'].astype(bool).tolist()
        return names_A, names_B, labels
    else:
        print(f"Census data not found at {path}, generating mock data with typos.")
        np.random.seed(456)
        # 生成常见姓名的变体（编辑距离模拟）
        common_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
        n_queries = 50
        n_db = 500
        names_A = []
        names_B = []
        # 确保前20个查询在 B 中有匹配（可能带有轻微拼写错误）
        for i in range(n_queries):
            if i < 20:
                orig = common_names[i % len(common_names)]
                names_A.append(orig)
                # 模拟 B 中相同名字
                names_B.append(orig)
                labels.append(True)
            else:
                names_A.append(f"Random_{i}")
                names_B.append(f"Random_{i+100}")
                labels.append(False)
        # 将 B 补齐到 n_db
        while len(names_B) < n_db:
            names_B.append("Extra_" + str(len(names_B)))
        return names_A, names_B, labels