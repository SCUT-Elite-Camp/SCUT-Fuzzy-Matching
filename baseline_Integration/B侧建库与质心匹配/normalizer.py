# preprocessing/normalizer.py
import numpy as np
from sklearn.preprocessing import normalize


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """
    对二维矩阵每行做 L2 归一化。
    输入形状: (n, d)
    输出形状: (n, d)，每行范数为 1（零向量保持为零）
    """
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
        return normalize(matrix, norm="l2")[0]
    return normalize(matrix, norm="l2")
