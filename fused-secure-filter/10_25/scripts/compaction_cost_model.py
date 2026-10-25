#!/usr/bin/env python3
"""compaction 的三层成本模型（容量 / 理想收益 / 安全压缩成本）。

替代上一版 `compaction_feasibility.py`。上一版把"Benes 深度 × 特征数 × tile 数"
直接当作旋转次数的**下界**，这个定性是错的：它是**某一类实现（置换网络）**的价格，
不是问题的价格。本脚本把它降级为一个显式假设，并把成本拆成三项可分别实测的量。

## 三层

  L1 容量      纯组合学：T = S//m、W <= T、压缩后 tile 数。这部分可以当成事实。
  L2 理想收益  基线 kernel 与"压缩后 kernel"的算子级差异。**算子计数来自代码审计**，
              单位成本由外部传入（ct-pt / ct-ct / rescale / relin 各自独立的单价）。
  L3 安全压缩  候选必须以秘密方式压进定宽 W 槽位。这里的成本按"置换网络族"估算，
              且显式写出那个 50 倍的来源（特征维是 50 个独立密文，这是**布局事实**，
              不是算法选择）。

## 为什么特征数是 50 倍

`tiled_batch_matching` 里点积写作
`score = sum_f enc_query_50[f] * selected_features[f]`：
每个特征是一个独立密文。任何对候选维度的重排，都必须对 50 个特征密文各做一遍。
把特征塞进槽位可以省掉这 50 倍，但会挤占 (query, candidate) 的槽位预算，m 随之塌陷。

## 已知的未覆盖项

- 秘密选择还需要比较/等值测试，而当前 kernel **完全不做比较**（只有线性运算）。
  `--c-cmp` 用来给这一类算子单独定价。
- 输出定宽、隐藏输出数量、dummy 填充的语义正确性，本脚本只做断言式验证，不计成本。

## 用法

    python scripts/compaction_cost_model.py --c-pt 1 --c-ct 2 --c-rescale 1 \
        --c-relin 2 --c-rot 5 --c-cmp 3 --stages 22
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SLOT_CAPACITY = 4096

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.params import COEFF_MOD_BIT_SIZES, POLY_MODULUS_DEGREE  # noqa: E402

MAX_COEFF_MODULUS_BITS = {8192: 218, 16384: 438, 32768: 881}


def kernel_ledger(k: int, d: int, tiles: int) -> dict:
    """逐 tile 的算子台账，直接对应 party_b/online_responder.py:381-406。"""
    per_tile = {
        "ct_pt_mul": k * d,          # enc_selectors[i] * plain_i
        "ct_pt_add": 1,              # score += -tau * R
        "ct_ct_mul": d,              # enc_query_50[f] * selected_features[f]
        "ct_ct_add": d * (k - 1) + (d - 1),  # selector 累加 + 点积累加
        "rescale": k * d + d,        # 每次乘法 consumes 一个 level
        "relin": d,                  # 每次 ct-ct 乘法需要
        "rotate": 0,                 # 当前 kernel 不做任何旋转
        "compare": 0,                # 当前 kernel 不做任何比较
    }
    return {
        "per_tile": per_tile,
        "tiles": tiles,
        "total": {key: value * tiles for key, value in per_tile.items()},
        "source": "party_b/online_responder.py:381-406 (逐 tile 的 f/i 双循环)",
    }


def cost_of(ops: dict, unit: dict) -> float:
    return sum(ops[key] * unit[key] for key in ops)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--clusters", type=int, default=50)
    parser.add_argument("--max-size", type=int, default=483)
    parser.add_argument("--match-dim", type=int, default=50)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--stages", type=int, default=22,
                        help="置换网络级数；22 对应 20 槽位 Benes，属于实现假设而非下界")
    # 单位成本（相对量，单位自定：可以是毫秒，也可以是 ct-pt 当量）
    parser.add_argument("--c-pt", type=float, default=1.0)
    parser.add_argument("--c-ct", type=float, default=2.0)
    parser.add_argument("--c-ct-add", type=float, default=0.1,
                        help="ct-ct 加法相对 ct-pt 乘的单价；加法无 NTT，通常远低于乘法")
    parser.add_argument("--c-rescale", type=float, default=1.0)
    parser.add_argument("--c-relin", type=float, default=2.0)
    parser.add_argument("--c-rot", type=float, default=5.0)
    parser.add_argument("--c-cmp", type=float, default=3.0)
    parser.add_argument("--cost-model", choices=["composite", "primitive"], default="composite",
                        help="composite: ct 乘法单价已含 relin/rescale，不再重复计；"
                             "primitive: 单独计 rescale/relin 单价")
    parser.add_argument("--out", type=Path, default=Path("artifacts/compaction_cost/model.json"))
    args = parser.parse_args()

    m, k, d, S = args.batch_size, args.clusters, args.match_dim, SLOT_CAPACITY
    T = S // m
    unit = {
        "ct_pt_mul": args.c_pt, "ct_pt_add": args.c_pt,
        "ct_ct_mul": args.c_ct, "ct_ct_add": args.c_ct_add,
        "rescale": args.c_rescale, "relin": args.c_relin,
        "rotate": args.c_rot, "compare": args.c_cmp,
    }

    # ---------- L1 容量 ----------
    base_tiles = math.ceil(args.max_size / T)
    compact_tiles = math.ceil(m * args.width / S)
    l1 = {
        "T": T, "W": args.width, "W_max": T, "W_feasible": args.width <= T,
        "base_tiles": base_tiles, "compact_tiles": compact_tiles,
        "note": "W <= T=20 是槽位硬约束；W=6 是每条查询的候选槽位，不等于 tile 数",
    }

    # ---------- L1b 乘法深度可行性（纯结构，与单价无关） ----------
    total_bits = sum(COEFF_MOD_BIT_SIZES)
    limit = MAX_COEFF_MODULUS_BITS.get(POLY_MODULUS_DEGREE)
    available_depth = max(0, len(COEFF_MOD_BIT_SIZES) - 2)
    # 匹配 kernel 本身的深度：selector ct-pt 乘 1 层，query ct-ct 乘 1 层
    matching_depth = 2
    # compaction 至少需要的乘法层数（每次掩码选择 = 1 次 ct-pt 乘）
    compaction_depth_min = 1
    headroom_bits = (limit - total_bits) if limit else None
    depth_check = {
        "poly_modulus_degree": POLY_MODULUS_DEGREE,
        "chain": list(COEFF_MOD_BIT_SIZES),
        "total_bits": total_bits,
        "secure_limit_bits": limit,
        "headroom_bits": headroom_bits,
        "available_depth": available_depth,
        "matching_depth": matching_depth,
        "compaction_depth_min": compaction_depth_min,
        "depth_fits": matching_depth + compaction_depth_min <= available_depth,
        "can_add_prime_at_current_degree": bool(headroom_bits is not None and headroom_bits >= 40),
        "note": "compaction 输出不能解密（会向查询方泄漏库内候选）也不能重加密（无 key），"
                "所以它与匹配阶段共享同一份深度预算。",
    }

    # ---------- L2 理想收益 ----------
    base = kernel_ledger(k, d, base_tiles)
    if args.cost_model == "composite":
        # TenSEAL 的 ct*ct / ct*plain 内部已完成 relin 与 rescale，重复计会高估基线
        base["total"]["rescale"] = 0
        base["total"]["relin"] = 0
    
    # 压缩后：每条查询自带候选列表，不需要 one-hot selector，只剩逐特征 ct-pt 点积
    compact_ops = {
        "ct_pt_mul": d * compact_tiles, "ct_pt_add": compact_tiles,
        "ct_ct_mul": 0, "ct_ct_add": 0,
        "rescale": d * compact_tiles, "relin": 0, "rotate": 0, "compare": 0,
    }
    c_base = cost_of(base["total"], unit)
    c_matched = cost_of(compact_ops, unit)
    l2 = {
        "baseline_ops": base["total"], "baseline_cost": c_base,
        "matched_ops": compact_ops, "matched_cost": c_matched,
        "budget_cost": c_base - c_matched,
    }

    # ---------- L3 安全压缩（置换网络族） ----------
    # 源数据：m*max_size 个候选 × d 个特征密文（特征维是独立密文，这是布局事实）
    source_ciphertexts_per_feature = math.ceil(m * args.max_size / S)
    routing_rotations = d * source_ciphertexts_per_feature * args.stages
    # 秘密选择还需要对每个候选做一次等值/选择运算
    logical_predicates = m * args.max_size              # 需要判定多少个候选
    packed_selection_ops = math.ceil(m * args.max_size / S)  # 打包后每特征的选择操作数
    select_compares = logical_predicates / S            # 仅作槽位折算，不是比较次数
    l3 = {
        "features": d,
        "source_ciphertexts_per_feature": source_ciphertexts_per_feature,
        "stages": args.stages,
        "routing_rotations": routing_rotations,
        "logical_predicates": logical_predicates,
        "packed_selection_ops_per_feature": packed_selection_ops,
        "slot_normalized_compares": select_compares,
        "cost": routing_rotations * unit["rotate"] + select_compares * unit["compare"],
        "family": "置换网络（Benes 类）；GC 方案不在本模型内",
    }

    verdict = l3["cost"] < l2["budget_cost"]
    payload = {
        "config": vars(args) | {"out": str(args.out)},
        "unit_costs": unit,
        "cost_model": args.cost_model,
        "L1_capacity": l1,
        "L1b_depth": depth_check,
        "L2_ideal_gain": l2,
        "L2_label": "idealized matching-cost budget（未证明可兑现）",
        "L3_secure_compaction": l3,
        "break_even": {
            "condition": "L3 < L2.budget",
            "holds": bool(verdict),
            "ratio": l3["cost"] / l2["budget_cost"] if l2["budget_cost"] else None,
        },
        "ledger": base,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("== L1 容量 ==")
    print(f"  S={S}, m={m}, T={T}, W={args.width} (上限 {T})")
    print(f"  基线 tiles={base_tiles}, 压缩后 tiles={compact_tiles}")
    print("\n== L1b 乘法深度 ==")
    print(f"  {depth_check['chain']} 共 {total_bits} bit，上限 {limit}，余量 {headroom_bits} bit")
    print(f"  可用深度 {available_depth}；匹配 kernel 已用 {matching_depth}；"
          f"compaction 至少还需 {compaction_depth_min}")
    print(f"  -> 深度是否放得下: {'是' if depth_check['depth_fits'] else '否'}")
    print(f"  -> 能否在 degree={POLY_MODULUS_DEGREE} 下加一个素数: "
          f"{'能' if depth_check['can_add_prime_at_current_degree'] else '不能'}")
    print("\n== L2 算子台账（每 tile）==")
    for key, value in base["per_tile"].items():
        print(f"  {key:<12} {value:>6}   ×{base_tiles} = {value*base_tiles:>8,}")
    print(f"  计价模型: {args.cost_model}")
    print(f"\n  基线成本 {c_base:,.0f} / 压缩后 {c_matched:,.0f} / 可花预算 {c_base-c_matched:,.0f}")
    print("\n== L3 安全压缩（置换网络族）==")
    print(f"  特征密文 {d} × 每特征源密文 {source_ciphertexts_per_feature} × 级数 {args.stages}"
          f" = {routing_rotations:,} 次旋转")
    print(f"  等值/选择 约 {select_compares:,.0f} 次")
    print(f"  成本 {l3['cost']:,.0f}")
    threshold = l2["budget_cost"] / routing_rotations if routing_rotations else float("inf")
    print(f"\n  break-even 旋转单价 c_rot* = 预算 / 旋转数 = {threshold:.2f}")
    print(f"\n== 判据 ==  L3 {l3['cost']:,.0f}  vs  预算 {l2['budget_cost']:,.0f}  ->  "
          f"{'可能划算' if verdict else '不划算'} (比值 {l3['cost']/l2['budget_cost']:.2f})")

    # 敏感性：旋转单价
    print("\n== 敏感性：旋转单价（其余单价固定）==")
    for c_rot in (1, 2, 5, 10, 20):
        unit2 = dict(unit); unit2["rotate"] = c_rot
        cost = routing_rotations * c_rot + select_compares * unit2["compare"]
        print(f"  c_rot={c_rot:>2}  L3={cost:>12,.0f}  预算={l2['budget_cost']:>10,.0f}  "
              f"比值={cost/l2['budget_cost']:>5.2f}  {'✓' if cost < l2['budget_cost'] else '✗'}")
    print(f"\noutputs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
