"""CKKS 上下文创建模块。

所有加密参数从 config.params 导入，禁止硬编码。
"""

import tenseal as ts

from config.params import COEFF_MOD_BIT_SIZES, POLY_MODULUS_DEGREE, SCALE


def create_ckks_context() -> ts.Context:
    """创建并返回已配置 Galois 密钥的 TenSEAL CKKS 上下文。

    使用 config.params 中的全局加密参数：
        - poly_modulus_degree = POLY_MODULUS_DEGREE
        - coeff_mod_bit_sizes = COEFF_MOD_BIT_SIZES
        - global_scale = SCALE

    自动生成 Galois 密钥以支持槽位旋转与批量编码操作。

    Returns:
        ts.Context: 已配置的 CKKS 上下文对象，包含 Galois 密钥。
    """
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
    )
    # 显式设置缩放因子，控制定点数精度
    ctx.global_scale = SCALE
    # Galois 密钥是 slot 旋转与批量打包的必要条件
    ctx.generate_galois_keys()
    return ctx
