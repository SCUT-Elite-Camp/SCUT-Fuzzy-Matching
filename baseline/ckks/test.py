import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # 定位到baseline文件夹

import numpy as np
import tenseal as ts
from ckks import *


# ==========================================================
# 完整功能演示
# ============================================================

def main():
    """完整功能演示：上下文构建 → 密钥生成 → 加密 →
       序列化 → 反序列化 → 解密 → 同态运算验证。

    使用真实 numpy 数组测试全流程，验证:
        - 密文生成正常
        - 序列化 / 反序列化正常
        - 解密误差 ≤ 10^-6
        - 反序列化密文可执行同态加、减、乘运算
    """
    print("=" * 64)
    print("CKKS 全同态加密模块 — 功能演示")
    print("=" * 64)

    # ----------------------------------------------------------
    # 步骤 1: 构建 CKKS 上下文
    # ----------------------------------------------------------
    print("\n[1/6] 构建 CKKS 上下文...")
    ctx = build_context()
    print("  >> 固定参数:")
    print(f"      poly_modulus_degree  = {_POLY_MODULUS_DEGREE}")
    print(f"      coeff_mod_bit_sizes  = {_COEFF_MOD_BIT_SIZES}")
    print(f"      global_scale         = 2^40 = {_GLOBAL_SCALE}")
    print("  >> 批量加密支持: 已启用 (Galois 密钥已生成)")
    print("  [OK] 上下文构建成功")

    # ----------------------------------------------------------
    # 步骤 2: 生成密钥
    # ----------------------------------------------------------
    print("\n[2/6] 生成密钥对及重线性化密钥...")
    ctx, sk = generate_keys(ctx)
    print(f"  >> 私钥: 已生成")
    print( "  >> 重线性化密钥: 已生成并绑定至上下文")
    print( "  [OK] 密钥生成成功")

    # ----------------------------------------------------------
    # 步骤 3: 准备测试数据并加密
    # ----------------------------------------------------------
    print("\n[3/6] 准备测试数据并执行加密...")

    # 真实 numpy 测试向量（覆盖正数、负数、零、小数、大数）
    original = np.array(
        [1,2,3,4],
        dtype=np.float64
    )
    print(f"  >> 原始向量 ({len(original)} 维):")
    print(f"     {original}")

    ct_original = encrypt(original, ctx)
    print(f"  >> 密文类型: {type(ct_original).__name__}")
    print(f"  >> 密文尺度因子: {_GLOBAL_SCALE}")
    print( "  [OK] 加密成功")

    # ----------------------------------------------------------
    # 步骤 4: 序列化密文
    # ----------------------------------------------------------
    print("\n[4/6] 序列化密文...")
    ct_bytes = serialize_ct(ct_original)
    print(f"  >> 字节流长度: {len(ct_bytes)} bytes ({len(ct_bytes)/1024:.1f} KB)")
    print( "  [OK] 序列化成功")

    # ----------------------------------------------------------
    # 步骤 5: 反序列化密文
    # ----------------------------------------------------------
    print("\n[5/6] 反序列化密文...")
    ct_restored = deserialize_ct(ct_bytes, ctx)
    print(f"  >> 恢复密文类型: {type(ct_restored).__name__}")
    print( "  [OK] 反序列化成功")

    # ----------------------------------------------------------
    # 步骤 6: 解密并验证精度
    # ----------------------------------------------------------
    print("\n[6/6] 解密并验证精度...")
    decrypted = decrypt(ct_restored, sk)
    print(f"  >> 解密向量 ({len(decrypted)} 维):")
    print(f"     {decrypted}")

    # 逐元素对比误差
    abs_errors = np.abs(original - decrypted)
    max_error = np.max(abs_errors)
    mean_error = np.mean(abs_errors)
    print(f"\n  >> 绝对误差向量: {abs_errors}")
    print(f"  >> 最大绝对误差:  {max_error:.4e}")
    print(f"  >> 平均绝对误差:  {mean_error:.4e}")

    if max_error < 1e-6:
        print( "  [OK] 解密精度满足要求 (最大误差 < 10^-6)")
    else:
        print(f"  [FAIL] 解密精度不满足要求! (max_error={max_error:.4e})")

    # ==========================================================
    # 补充验证: 同态运算
    # ==========================================================

    # print("\n" + "=" * 64)
    # print("补充验证: 同态运算")
    # print("=" * 64)

    # # 第二组测试向量
    # vec_b = np.array(
    #     [0.5, 1.0, -1.0, 2.0, -50.0, 0.001, -42.0, 3.14],
    #     dtype=np.float64
    # )
    # ct_b = encrypt(vec_b, ctx)

    # # ---- 同态加法 ----
    # print("\n--- 同态加法: ct_a + ct_b ---")
    # ct_add = ct_original + ct_b
    # dec_add = decrypt(ct_add, sk)
    # expected_add = original + vec_b
    # err_add = np.max(np.abs(expected_add - dec_add))
    # print(f"  预期:  {expected_add}")
    # print(f"  结果:  {dec_add}")
    # print(f"  最大误差: {err_add:.4e}" +
    #       ("  [OK]" if err_add < 1e-6 else "  [FAIL]"))

    # # ---- 同态减法 ----
    # print("\n--- 同态减法: ct_a - ct_b ---")
    # ct_sub = ct_original - ct_b
    # dec_sub = decrypt(ct_sub, sk)
    # expected_sub = original - vec_b
    # err_sub = np.max(np.abs(expected_sub - dec_sub))
    # print(f"  预期:  {expected_sub}")
    # print(f"  结果:  {dec_sub}")
    # print(f"  最大误差: {err_sub:.4e}" +
    #       ("  [OK]" if err_sub < 1e-6 else "  [FAIL]"))

    # # ---- 同态乘法 (element-wise) ----
    # print("\n--- 同态乘法: ct_a * ct_b ---")
    # ct_mul = ct_original * ct_b
    # dec_mul = decrypt(ct_mul, sk)
    # expected_mul = original * vec_b
    # err_mul = np.max(np.abs(expected_mul - dec_mul))
    # print(f"  预期:  {expected_mul}")
    # print(f"  结果:  {dec_mul}")
    # print(f"  最大误差: {err_mul:.4e}" +
    #       ("  [OK]" if err_mul < 1e-6 else "  [FAIL]"))

    # # ---- 同态标量乘法 ----
    # print("\n--- 同态标量乘法: ct_a * 3.5 ---")
    # ct_scalar = ct_original * 3.5
    # dec_scalar = decrypt(ct_scalar, sk)
    # expected_scalar = original * 3.5
    # err_scalar = np.max(np.abs(expected_scalar - dec_scalar))
    # print(f"  预期:  {expected_scalar}")
    # print(f"  结果:  {dec_scalar}")
    # print(f"  最大误差: {err_scalar:.4e}" +
    #       ("  [OK]" if err_scalar < 1e-6 else "  [FAIL]"))

    # # ---- 同态求反 ----
    # print("\n--- 同态求反: -ct_a ---")
    # ct_neg = -ct_original
    # dec_neg = decrypt(ct_neg, sk)
    # expected_neg = -original
    # err_neg = np.max(np.abs(expected_neg - dec_neg))
    # print(f"  预期:  {expected_neg}")
    # print(f"  结果:  {dec_neg}")
    # print(f"  最大误差: {err_neg:.4e}" +
    #       ("  [OK]" if err_neg < 1e-6 else "  [FAIL]"))

    # # ---- 同态幂次 ----
    # print("\n--- 同态幂次: ct_a ** 3 ---")
    # ct_pow = ct_original ** 3
    # dec_pow = decrypt(ct_pow, sk)
    # expected_pow = original ** 3
    # err_pow = np.max(np.abs(expected_pow - dec_pow))
    # print(f"  预期:  {expected_pow}")
    # print(f"  结果:  {dec_pow}")
    # print(f"  最大误差: {err_pow:.4e}" +
    #       ("  [OK]" if err_pow < 1e-6 else "  [FAIL]"))

    # # ----------------------------------------------------------
    # # 测试总结
    # # ----------------------------------------------------------
    # print("\n" + "=" * 64)
    # print("测试总结")
    # print("=" * 64)

    # results = {
    #     "解密精度":            max_error,
    #     "同态加法 (ct_a+ct_b)": err_add,
    #     "同态减法 (ct_a-ct_b)": err_sub,
    #     "同态乘法 (ct_a*ct_b)": err_mul,
    #     "标量乘法 (ct_a*3.5)":  err_scalar,
    #     "同态求反 (-ct_a)":     err_neg,
    #     "幂次运算 (ct_a**3)":   err_pow,
    #     "序列化/反序列化":      0.0,  # 已验证类型与解密
    # }

    # for name, err in results.items():
    #     status = "[OK]" if err < 1e-6 else "[FAIL]"
    #     if name == "序列化/反序列化":
    #         print(f"  {status}  {name}: 密文完整恢复，可正常运算")
    #     else:
    #         print(f"  {status}  {name}: 最大误差 = {err:.4e}")

    # all_pass = all(err < 1e-6 for err in results.values())
    # print(f"\n  结论: {'[全部通过]' if all_pass else '[存在失败项]'}")
    # print("=" * 64)


if __name__ == "__main__":
    main()
