"""全局参数配置 —— 单一真相来源。

所有模块均从此处 import 参数，禁止硬编码。
"""

# ==================== MinHash 参数 ====================
SHINGLE_SIZE = 3                  # trigram（含空格）
NUM_PERMUTATIONS_CLUSTER = 200    # EL=200，用于聚类 / 质心匹配
NUM_PERMUTATIONS_MATCH = 50       # EL=50，用于列式名字匹配
MAX_HASH = 2 ** 20                # 20-bit hash space
HASH_SEED = 42                    # 固定种子，保证 Party A / B 置换完全一致

# ==================== 聚类参数 ====================
K_CLUSTERS_FUNC = lambda n: int(n ** 0.5)   # k ≈ √|N_B|
KMEANS_ITERATIONS = 20

# ==================== CKKS 参数 ====================
POLY_MODULUS_DEGREE = 8192        # 多项式模度，决定槽位数量（最多 4096）
COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]  # 系数模位宽链，共 200 位安全强度
SCALE = 2 ** 40             # 缩放因子（定点数小数点位置），控制数值精度

# ==================== 协议参数 ====================
SIMILARITY_THRESHOLD = 0.9        # 余弦相似度阈值 τ

# ==================== 路径 ====================
ARTIFACTS_DIR = "./artifacts"
DATA_DIR = "./data"

# ==================== 其他参数 ====================
DECRYPT_EPS = 1e-6  # 解密误差容忍度，CKKS 解密结果与原始值的最大允许差距
RANDOM_MASK_MIN = 1.0  # 随机掩码最小值（用于协议安全性增强，防止泄露原始相似度）
RANDOM_MASK_MAX = 10.0  # 随机掩码最大值（用于协议安全性增强，防止泄露原始相似度）