"""归一化与标准化模块

提供 L2 逐行归一化，供 MinHash 签名矩阵使用
StandardScaler 相关逻辑由 clustering 模块负责,因其需 fit 后共享参数
"""

import numpy as np


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """对输入矩阵逐行做 L2 归一化。

    归一化后，两个向量的内积即等于余弦相似度：
        ⟨N̂[i], N̂[j]⟩ = cos(N̂[i], N̂[j])

    Args:
        matrix: 形状 (N, d) 的 numpy 数组。

    Returns:
        形状 (N, d) 的 L2 归一化后的数组。零向量保持为零向量（避免除零）。
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    # 计算每行的 L2 范数，保持二维便于广播
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # 避免除零：范数为 0 的行保持原样（零向量）
    norms[norms == 0] = 1.0
    return matrix / norms
