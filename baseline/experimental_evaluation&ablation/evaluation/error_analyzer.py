"""
误差分析组件：评估 CKKS 加密对余弦相似度计算的影响
依赖 TenSEAL 库（需要先安装：pip install tenseal）
"""

import numpy as np
import tenseal as ts
from typing import Tuple, List, Dict
import time

class CKKSApproximationError:
    def __init__(self, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60], scale=2**40):
        """
        初始化 CKKS 上下文（与项目规范一致）
        """
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        self.context.generate_galois_keys()
        self.context.global_scale = scale
        self.scale = scale

    def encrypt_decrypt(self, plain_vector: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        加密并立即解密一个向量，返回 (原始向量, 解密向量)
        """
        secret_key = self.context.secret_key()
        public_key = self.context.public_key()
        # 加密
        cipher = ts.ckks_vector(self.context, plain_vector)
        # 解密
        decrypted = cipher.decrypt(secret_key)
        return np.array(plain_vector), np.array(decrypted)

    def ct_pt_dot_error(self, vec_a: np.ndarray, vec_b: np.ndarray) -> Dict:
        """
        计算 CT-PT 点积的明文与密文近似误差
        vec_a: 将被加密的向量
        vec_b: 明文字典向量
        """
        # 明文点积
        plain_dot = np.dot(vec_a, vec_b)
        # 加密 vec_a
        cipher_a = ts.ckks_vector(self.context, vec_a.tolist())
        # 计算 CT-PT 点积
        cipher_dot = cipher_a.dot(vec_b.tolist())
        # 解密
        decrypted_dot = cipher_dot.decrypt()[0]
        error = abs(plain_dot - decrypted_dot)
        rel_error = error / (abs(plain_dot) + 1e-12)
        return {
            "plain_dot": plain_dot,
            "decrypted_dot": decrypted_dot,
            "absolute_error": error,
            "relative_error": rel_error
        }

    def ct_ct_dot_error(self, vec_a: np.ndarray, vec_b: np.ndarray) -> Dict:
        """
        计算 CT-CT 点积的明文与密文近似误差
        """
        plain_dot = np.dot(vec_a, vec_b)
        cipher_a = ts.ckks_vector(self.context, vec_a.tolist())
        cipher_b = ts.ckks_vector(self.context, vec_b.tolist())
        cipher_dot = cipher_a.dot(cipher_b)
        decrypted_dot = cipher_dot.decrypt()[0]
        error = abs(plain_dot - decrypted_dot)
        rel_error = error / (abs(plain_dot) + 1e-12)
        return {
            "plain_dot": plain_dot,
            "decrypted_dot": decrypted_dot,
            "absolute_error": error,
            "relative_error": rel_error
        }

    def threshold_sign_stability(self, cos_sim: float, tau: float, num_trials=100) -> Dict:
        """
        测试阈值判断的符号稳定性：cos_sim - tau 再乘以随机正数后，正负号是否保持。
        返回失败比例。
        """
        failures = 0
        for _ in range(num_trials):
            # 构造一个包含 cos_sim 的单元素向量
            plain = [cos_sim]
            cipher = ts.ckks_vector(self.context, plain)
            # 减去 tau（常数）
            cipher_sub = cipher - tau
            # 乘以随机正数 r（在密文上）
            r = np.random.uniform(1, 1000)
            cipher_mul = cipher_sub * r
            decrypted = cipher_mul.decrypt()[0]
            expected_sign = 1 if cos_sim > tau else -1
            actual_sign = 1 if decrypted > 0 else -1
            if actual_sign != expected_sign:
                failures += 1
        return {
            "cos_sim": cos_sim,
            "tau": tau,
            "num_trials": num_trials,
            "failures": failures,
            "failure_rate": failures / num_trials
        }

    def batch_analysis(self, num_samples=1000, vec_dim=50, tau=0.9):
        """
        批量分析：生成随机归一化向量，计算明文余弦相似度与密文点积的误差，
        并统计阈值判断稳定性。
        """
        ct_pt_errors = []
        ct_ct_errors = []
        sign_stable_tests = []

        for _ in range(num_samples):
            # 生成随机向量并归一化
            a = np.random.randn(vec_dim)
            b = np.random.randn(vec_dim)
            a = a / np.linalg.norm(a)
            b = b / np.linalg.norm(b)
            cos_sim = np.dot(a, b)

            # CT-PT 点积误差
            res_pt = self.ct_pt_dot_error(a, b)
            ct_pt_errors.append(res_pt["absolute_error"])

            # CT-CT 点积误差
            res_ct = self.ct_ct_dot_error(a, b)
            ct_ct_errors.append(res_ct["absolute_error"])

            # 符号稳定性（只测试边界附近：cos_sim 接近 tau）
            if abs(cos_sim - tau) < 0.05:
                stab = self.threshold_sign_stability(cos_sim, tau, num_trials=20)
                sign_stable_tests.append(stab["failure_rate"])

        # 统计
        def stats(arr):
            return {
                "mean": np.mean(arr),
                "std": np.std(arr),
                "max": np.max(arr),
                "median": np.median(arr)
            }

        return {
            "ct_pt_abs_error": stats(ct_pt_errors),
            "ct_ct_abs_error": stats(ct_ct_errors),
            "threshold_failure_rate": stats(sign_stable_tests) if sign_stable_tests else {"mean": 0, "std": 0, "max": 0, "median": 0}
        }


def generate_report():
    """生成误差分析报告"""
    print("初始化 CKKS 上下文...")
    analyzer = CKKSApproximationError()
    print("运行批量误差分析（1000 个随机向量对，维度 50）...")
    results = analyzer.batch_analysis(num_samples=1000, vec_dim=50, tau=0.9)

    print("\n" + "="*60)
    print("误差分析报告")
    print("="*60)
    print(f"CT-PT 点积绝对误差 (plain - decrypted):")
    print(f"  均值: {results['ct_pt_abs_error']['mean']:.6e}")
    print(f"  标准差: {results['ct_pt_abs_error']['std']:.6e}")
    print(f"  最大值: {results['ct_pt_abs_error']['max']:.6e}")
    print(f"  中位数: {results['ct_pt_abs_error']['median']:.6e}")
    print()
    print(f"CT-CT 点积绝对误差:")
    print(f"  均值: {results['ct_ct_abs_error']['mean']:.6e}")
    print(f"  标准差: {results['ct_ct_abs_error']['std']:.6e}")
    print(f"  最大值: {results['ct_ct_abs_error']['max']:.6e}")
    print(f"  中位数: {results['ct_ct_abs_error']['median']:.6e}")
    print()
    print("阈值判断符号稳定性（在 τ=0.9 附近）:")
    if results['threshold_failure_rate']['mean'] == 0:
        print("  ✅ 所有测试中正负号均未翻转，阈值判断可靠。")
    else:
        print(f"  ⚠️ 失败率均值: {results['threshold_failure_rate']['mean']:.4f}")
        print(f"  建议增大 CKKS 缩放因子或降低乘法深度。")
    print("="*60)

    # 额外：展示一个具体例子
    print("\n示例：单个向量对 (CT-PT 点积)")
    a = np.random.randn(50)
    b = np.random.randn(50)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    res = analyzer.ct_pt_dot_error(a, b)
    print(f"明文点积: {res['plain_dot']:.6f}")
    print(f"解密点积: {res['decrypted_dot']:.6f}")
    print(f"绝对误差: {res['absolute_error']:.6e}")
    print(f"相对误差: {res['relative_error']:.2e}")

if __name__ == "__main__":
    generate_report()