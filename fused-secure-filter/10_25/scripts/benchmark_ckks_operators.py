#!/usr/bin/env python3
"""CKKS 算子单价微基准（修正版）。

上一版的问题（均已修正）：

  1. `fresh() * [0.5] * n` 在 Python 里是从左到右结合，实际测的是"密文乘标量再乘 n"，
     不是"乘长度为 n 的明文向量"。现在显式构造 `plain = [0.5] * n` 再使用。
  2. 计时区间内调用 `fresh()`，把加密成本算进了算子成本；ct-ct 还构造两份。
     现在所有输入在计时区间外准备好，循环体内只执行目标算子。
  3. 模数链写死成 400 bit，超出 degree=8192 的安全上限，且与生产不符。
     现在直接复用 `config.params` 与 `ckks.context.create_ckks_context`。
  4. 原先想用"带/不带 Galois 密钥"做敏感性——生产上下文**已经无条件生成 Galois 密钥**
     (`ckks/context.py`)，所以这一项对基线与 compaction 是同一个常数，不应作为变量。

## 能测什么、不能测什么

TenSEAL 暴露的接口只有复合算子：`ct*ct` 内部包含 relinearize + rescale，无法单独计时。
因此本脚本输出的是**复合算子单价**，并明确标注：

    primitives_measured      : ct_pt_mul, ct_pt_add, ct_ct_mul, ct_ct_add, rotate
    primitives_not_measurable: rescale, relin（需 C++ 层插桩，或用差值法并接受误差）

成本模型必须二选一：用复合单价且不再叠加 rescale/relin，或用原语单价。
两者混用即为重复计价——上一版模型犯的正是这个错。

## 用法

    python scripts/benchmark_ckks_operators.py --repeats 30 \
        --out artifacts/ckks_ops/ops.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.params import COEFF_MOD_BIT_SIZES, POLY_MODULUS_DEGREE, SCALE  # noqa: E402

# degree=8192 的 128-bit 安全系数模数总位数上限（SEAL 默认表）
MAX_COEFF_MODULUS_BITS = {8192: 218, 16384: 438, 32768: 881}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--vector-size", type=int, default=4000)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "ckks_ops" / "ops.json")
    args = parser.parse_args()

    # ---- 先做参数合法性检查，不合法就不要浪费时间测 ----
    total_bits = sum(COEFF_MOD_BIT_SIZES)
    limit = MAX_COEFF_MODULUS_BITS.get(POLY_MODULUS_DEGREE)
    depth = max(0, len(COEFF_MOD_BIT_SIZES) - 2)  # 最后一个素数用于 key-switch
    print(f"参数: degree={POLY_MODULUS_DEGREE}, chain={COEFF_MOD_BIT_SIZES}, "
          f"总位数={total_bits}, 安全上限={limit}, 可用乘法深度≈{depth}")
    if limit is not None and total_bits > limit:
        print(f"模数链总位数 {total_bits} > 上限 {limit}：SEAL 会拒绝加载，先修参数再测。")
        return 2

    try:
        import tenseal as ts
    except ImportError:
        print("tenseal 未安装：需在生产同款环境运行。")
        return 3

    from ckks.context import create_ckks_context
    from ckks.keys import serialize_public_context

    ctx = create_ckks_context()  # 与生产完全一致，已含 Galois 密钥
    n = args.vector_size
    plain = [0.5] * n
    values = [0.01 * (index % 7) for index in range(n)]

    # ---- 所有输入在计时区间外准备好 ----
    ct_a = ts.ckks_vector(ctx, values)
    ct_b = ts.ckks_vector(ctx, values)
    ct_small = ts.ckks_vector(ctx, [0.01] * n)

    def timeit(fn, repeats: int) -> float:
        for _ in range(2):  # 预热
            fn()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - start)
        return statistics.median(samples)

    ops = {
        "ct_pt_mul": timeit(lambda: ct_a * plain, args.repeats),
        "ct_pt_add": timeit(lambda: ct_a + plain, args.repeats),
        "ct_ct_mul": timeit(lambda: ct_a * ct_b, args.repeats),
        "ct_ct_add": timeit(lambda: ct_a + ct_b, args.repeats),
        "rotate": timeit(lambda: ct_small.rotate(1), args.repeats),
    }

    # 正确性抽查：乘明文向量应逐槽等于乘积
    prod = (ct_a * plain).decrypt()
    expected = [value * 0.5 for value in values]
    max_abs_err = max(abs(a - b) for a, b in zip(prod[:100], expected[:100]))

    base = ops["ct_pt_mul"]
    payload = {
        "config": {
            "poly_modulus_degree": POLY_MODULUS_DEGREE,
            "coeff_mod_bit_sizes": COEFF_MOD_BIT_SIZES,
            "total_modulus_bits": total_bits,
            "secure_limit_bits": limit,
            "available_multiplicative_depth": depth,
            "scale_bits": int(round(SCALE.bit_length() - 1)),
            "vector_size": n,
            "repeats": args.repeats,
        },
        "seconds": ops,
        "relative_to_ct_pt_mul": {key: value / base for key, value in ops.items()},
        "ciphertext_bytes": len(ct_a.serialize()),
        "public_context_bytes": len(serialize_public_context(
            create_ckks_context()
        )) if callable(serialize_public_context) else None,
        "max_abs_error_ct_pt_mul": max_abs_err,
        "primitives_measured": ["ct_pt_mul", "ct_pt_add", "ct_ct_mul", "ct_ct_add", "rotate"],
        "primitives_not_measurable": ["rescale", "relin"],
        "usage_note": "成本模型必须二选一：复合单价（不再叠加 rescale/relin）或原语单价。混用=重复计价。",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'op':<14}{'median s':>12}{'relative':>10}")
    print("-" * 36)
    for key, value in ops.items():
        print(f"{key:<14}{value:>12.6f}{value / base:>10.2f}")
    print(f"\n密文 {len(ct_a.serialize())/1024:.0f} KB, 明文乘最大误差 {max_abs_err:.2e}")
    print(f"outputs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
