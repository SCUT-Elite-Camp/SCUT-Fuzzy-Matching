# config/params.py
# 全局参数配置，所有模块必须从此处导入，不允许在业务代码中硬编码

SHINGLE_SIZE = 3
NUM_PERMUTATIONS_CLUSTER = 200
NUM_PERMUTATIONS_MATCH = 50
MAX_HASH = 2 ** 20
HASH_SEED = 42

K_CLUSTERS_FUNC = lambda n: int(n ** 0.5)
KMEANS_ITERATIONS = 20

POLY_MODULUS_DEGREE = 8192
COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
SCALE = 2 ** 40

SIMILARITY_THRESHOLD = 0.9
DECRYPT_EPS = 1e-6

RANDOM_MASK_MIN = 1.0
RANDOM_MASK_MAX = 10.0


def choose_k(n: int, mode: str = "sqrt") -> int:
    """
    选择聚类数 k。
    mode='sqrt': 默认工程策略，k = int(sqrt(n))
    mode='paper': 按论文实验映射固定 k（需根据实验表扩展）
    """
    if mode == "sqrt":
        return max(1, int(n ** 0.5))
    elif mode == "paper":
        # 按论文 Table 2 的实验设置映射
        paper_map = {
            10_000: 50,
            100_000: 100,
            1_000_000: 500,
        }
        # 找最近的键
        closest = min(paper_map.keys(), key=lambda x: abs(x - n))
        return paper_map[closest]
    else:
        raise ValueError(f"未知 mode: {mode}，支持 'sqrt' 或 'paper'")
