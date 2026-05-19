import numpy as np
import tenseal as ts


# ============================================================
# 固定加密参数
# ============================================================
_POLY_MODULUS_DEGREE = 8192          # 多项式模度，决定槽位数量（最多 4096）
_COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]  # 系数模位宽链，共 200 位安全强度
_GLOBAL_SCALE = 2 ** 40              # 缩放因子（定点数小数点位置），控制数值精度


# ============================================================
# 核心功能接口
# ============================================================

def build_context():
    """构建符合规范的 CKKS 上下文，自动开启批量加密支持。

    使用固定加密参数初始化 TenSEAL CKKS 上下文:
        - 多项式模度 poly_modulus_degree = 8192
        - 缩放因子 global_scale = 2^40
        - 系数模位宽 coeff_mod_bit_sizes = [60, 40, 40, 60]

    功能:
        生成 CKKS 方案上下文并自动生成 Galois 密钥，
        以支持批量 slot 打包与旋转操作。

    参数:
        无

    返回值:
        tenseal.Context: 已配置完毕的 CKKS 上下文对象，
        包含 Galois 密钥，可直接用于批量加密。

    示例:
        >>> ctx = build_context()
        >>> isinstance(ctx, ts.Context)
        True

    注意事项:
        - 参数已固定，不可修改，否则与其他函数不兼容。
        - Galois 密钥是 slot 旋转与批量编码的前提。
        - global_scale 必须显式设置为 2^40。
    """
    # 构建 CKKS 上下文，显式指定所有固定参数
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=_POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=_COEFF_MOD_BIT_SIZES
    )
    ctx.global_scale = _GLOBAL_SCALE

    # 生成 Galois 密钥：支持槽位旋转，是批量 pack/unpack 的必要条件
    ctx.generate_galois_keys()

    return ctx