"""
数据集加载：NCVR、图书馆目录、US Census、以及从常见名字合成数据
若文件不存在则生成模拟数据。
"""

import numpy as np
import os

def load_dataset(name, path):
    """
    Args:
        name: str, 数据集名称，可选 "ncvr", "libcat", "census", "forenames"
        path: str, 数据集文件所在目录或文件路径（对于 forenames，直接指向 CSV 文件路径）

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
    elif name.lower() == "forenames":
        return _load_forenames(path)
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose from 'ncvr', 'libcat', 'census', 'forenames'.")

def _load_ncvr(path):
    """模拟 NCVR 数据加载。实际使用时根据 NCVR CSV 文件格式修改。"""
    if os.path.exists(path):
        # TODO: 根据实际 CSV 格式读取
        import pandas as pd
        df = pd.read_csv(path)
        names_A = df['name_A'].tolist()
        names_B = df['name_B'].tolist()
        labels = df['match_label'].astype(bool).tolist()
        return names_A, names_B, labels
    else:
        print(f"NCVR data not found at {path}, generating mock data for testing.")
        np.random.seed(42)
        n_queries = 100
        n_db = 1000
        names_A = [f"Person_{i}" for i in range(n_queries)]
        names_B = [f"Person_{j}" for j in range(n_db)]
        labels = [i < 10 for i in range(n_queries)]
        return names_A, names_B, labels

def _load_libcat(path):
    """模拟图书馆目录数据加载"""
    if os.path.exists(path):
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
        import pandas as pd
        df = pd.read_csv(path)
        names_A = df['name_A'].tolist()
        names_B = df['name_B'].tolist()
        labels = df['match_label'].astype(bool).tolist()
        return names_A, names_B, labels
    else:
        print(f"Census data not found at {path}, generating mock data with typos.")
        np.random.seed(456)
        common_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
        n_queries = 50
        n_db = 500
        names_A = []
        names_B = []
        labels = []
        for i in range(n_queries):
            if i < 20:
                orig = common_names[i % len(common_names)]
                names_A.append(orig)
                names_B.append(orig)
                labels.append(True)
            else:
                names_A.append(f"Random_{i}")
                names_B.append(f"Random_{i+100}")
                labels.append(False)
        while len(names_B) < n_db:
            names_B.append("Extra_" + str(len(names_B)))
        return names_A, names_B, labels

def _load_forenames(csv_path, n_B=1000, n_match=100, n_nonmatch=100, fuzzy_ratio=0.3, random_seed=42):
    """
    从 common-forenames-by-country.csv 合成模糊姓名匹配数据集。
    
    Args:
        csv_path: CSV 文件路径
        n_B: 响应方 B 的数据库大小
        n_match: 匹配查询的数量（标签为 True）
        n_nonmatch: 不匹配查询的数量（标签为 False）
        fuzzy_ratio: 匹配查询中被故意引入拼写错误的比例
        random_seed: 随机种子，保证可复现
    
    Returns:
        (names_A, names_B, labels)
    """
    import pandas as pd
    import random

    # 读取 CSV
    if not os.path.exists(csv_path):
        print(f"Forenames CSV not found at {csv_path}, falling back to mock data.")
        return _load_ncvr(csv_path)  # 回退到模拟数据

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    # 提取罗马化名字，去空，去重
    name_pool = df['Romanized Name'].dropna().unique().tolist()
    name_pool = [n for n in name_pool if isinstance(n, str) and n.strip() != '']

    if len(name_pool) < n_B + n_match + n_nonmatch:
        print(f"Forenames pool too small ({len(name_pool)}), increasing by allowing duplicates or reducing size.")
        # 如果名字不足，则允许重复使用（但尽量保持 B 中唯一）
        pass

    np.random.seed(random_seed)
    random.seed(random_seed)

    # 1. 构建 B 的数据库（尽量不重复）
    if len(name_pool) >= n_B:
        names_B = list(np.random.choice(name_pool, n_B, replace=False))
    else:
        names_B = list(np.random.choice(name_pool, n_B, replace=True))

    # 2. 构建匹配查询（从 B 中选，且可以引入模糊变体）
    matched_originals = list(np.random.choice(names_B, n_match, replace=False))
    names_A_matched = []
    for name in matched_originals:
        # 是否引入模糊错误？
        if random.random() < fuzzy_ratio and len(name) > 2:
            # 简单：随机替换一个字母
            pos = random.randint(0, len(name)-1)
            new_char = chr(random.randint(97, 122))
            fuzzy_name = name[:pos] + new_char + name[pos+1:]
            names_A_matched.append(fuzzy_name)
        else:
            names_A_matched.append(name)

    # 3. 构建不匹配查询（从不在 B 中的名字里选，若不够则允许重复）
    candidate_nonmatch = [n for n in name_pool if n not in names_B]
    if len(candidate_nonmatch) < n_nonmatch:
        # 不足则从全部 name_pool 中选（可能与 B 重复，但概率低）
        candidate_nonmatch = name_pool
    names_A_nonmatch = list(np.random.choice(candidate_nonmatch, n_nonmatch, replace=False))

    # 合并查询集
    names_A = names_A_matched + names_A_nonmatch
    labels = [True] * n_match + [False] * n_nonmatch

    return names_A, names_B, labels