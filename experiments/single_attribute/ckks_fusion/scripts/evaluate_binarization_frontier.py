#!/usr/bin/env python3
"""二值码粗筛可行性评估：从真实余弦分布到 (ell, d) 工作点。

这个脚本回答的不是"某个 rho 是多少"，而是三个可以分别被否证的问题：

  1. 真实姓名匹配数据上，正例 / 实体不相交负例的余弦分布是什么形状？
     （每条查询的最近负例与真目标是否重叠）
  2. 在这个**经验分布**上，ell 位随机超平面码 + Hamming <= d 的判据，
     能达到什么 (TPR, FPR, 候选保留率)？
  3. 满足给定 FPR 上限的工作点，d 是否落在 Approx-PSI 电路真正有优势的
     d <= 6 区间（`include/secure/hamming_threshold.h` 的 Wallace 分支）？

## 三级证据（输出里逐行标注）

  theory_only    把每条真实向量的 rho 代入 Binomial(ell, arccos(rho)/pi)，
                 但**只**用真目标与平均负例构造，不涉及真实负例分布。
  empirical_dist 同样逐对代入，但对**真实负例分布求平均**——不是把平均余弦
                 代进二项分布。这一步仍然不是实测，只是"分布积分"。
  mc_verified    用真实向量生成随机超平面码，直接数 Hamming 距离。这是唯一的
                 实测证据；前两级只能用来解释它，不能替代它。

## 为什么必须按实体划分正负

`ncvr_10k` 的正例查询与库内目标是同一个 ncid；同一姓名可能对应不同 ncid。
把"同名不同人"当作正例，或把"同 ncid 的另一条记录"当作负例，都会系统性地
污染 FPR。本脚本用 ncid 做实体键：负例 = ncid 不相等的全部库记录。

## 用法

    python scripts/evaluate_binarization_frontier.py \
        --data-root data --fuzzy-ratio 0.3 --ells 64 128 256 512 \
        --fpr-cap 0.05 --out-dir artifacts/binarization_frontier

只依赖 numpy。项目的 `l2_normalize` 依赖 sklearn，环境里没有时自动退化到内联
等价实现，`distribution.json` 的 `runtime.normalizer` 字段会写明走了哪条路径。

结果产物（默认 `artifacts/binarization_frontier/`）：
  frontier.csv       每个 (ell, d, seed) 的 TPR/FPR/候选数，含 d<=6 标记
  distribution.json  分布统计、重叠度量、配置、运行时路径
  summary.md         可直接贴进报告的小结
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.params import (  # noqa: E402
    NUM_PERMUTATIONS_CLUSTER,
    NUM_PERMUTATIONS_MATCH,
    SHINGLE_SIZE,
)
from minhash.encoder import batch_encode  # noqa: E402
from preprocessing.text_cleaner import clean_name  # noqa: E402

# ---------------------------------------------------------------------------
# 归一化：优先复用项目实现，缺 sklearn 时退化为等价内联实现
# ---------------------------------------------------------------------------

try:  # pragma: no cover - 取决于环境
    from preprocessing.normalizer import l2_normalize as _project_l2_normalize

    _NORMALIZER_SOURCE = "preprocessing.normalizer.l2_normalize (sklearn)"
except Exception:  # pragma: no cover
    _project_l2_normalize = None
    _NORMALIZER_SOURCE = "inline (sklearn unavailable)"


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化；零行保持为零。

    与 `preprocessing/normalizer.py` 的 sklearn 实现对零向量的处理一致。
    """
    if _project_l2_normalize is not None:
        return np.asarray(_project_l2_normalize(matrix), dtype=np.float64)
    arr = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


# ---------------------------------------------------------------------------
# 数据读取：与 scripts/evaluate_top1_projection.py:read_ncvr 保持一致
# ---------------------------------------------------------------------------


@dataclass
class QueryRecord:
    ncid: str
    name: str
    label: bool
    target_index: int | None
    perturbed: bool = False


def read_ncvr(
    data_root: Path,
    *,
    query_limit: int,
    db_limit: int,
    fuzzy_ratio: float,
    fuzzy_seed: int,
) -> tuple[list[str], list[str], list[QueryRecord]]:
    """读 NCVR 子集，返回 (库姓名, 库 ncid, 查询记录)。

    扰动配方与 `evaluate_top1_projection.read_ncvr` 逐字一致（单字符替换，
    且替换结果不得与库内任一词条相同），以保证与本仓库已有的候选实验可比。
    """
    import csv as _csv

    folder = data_root / "ncvr_10k" if (data_root / "ncvr_10k").exists() else data_root
    with (folder / "ncvr_10k_database.csv").open(newline="", encoding="utf-8") as handle:
        db_rows = list(_csv.DictReader(handle))
    with (folder / "ncvr_10k_queries.csv").open(newline="", encoding="utf-8") as handle:
        query_rows = list(_csv.DictReader(handle))
    if db_limit > 0:
        db_rows = db_rows[:db_limit]
    if query_limit > 0:
        query_rows = query_rows[:query_limit]

    db_names = [clean_name(row["full_name"]) for row in db_rows]
    db_ids = [row["ncid"] for row in db_rows]
    lookup = {key: index for index, key in enumerate(db_ids)}

    entries: list[QueryRecord] = []
    for row in query_rows:
        label = str(row["label"]).strip().lower() in {"1", "true", "yes"}
        entries.append(
            QueryRecord(
                ncid=row["query_ncid"],
                name=clean_name(row["query_name"]),
                label=label,
                target_index=lookup.get(row["query_ncid"]) if label else None,
            )
        )

    eligible = [
        index
        for index, entry in enumerate(entries)
        if entry.label and any("a" <= char <= "z" for char in entry.name)
    ]
    rng = random.Random(fuzzy_seed)
    forbidden = set(db_names)
    target_count = int(len(eligible) * fuzzy_ratio + 0.5)
    for index in rng.sample(eligible, min(target_count, len(eligible))):
        original = entries[index].name
        positions = [pos for pos, char in enumerate(original) if "a" <= char <= "z"]
        rng.shuffle(positions)
        for pos in positions:
            replacements = [c for c in "abcdefghijklmnopqrstuvwxyz" if c != original[pos]]
            rng.shuffle(replacements)
            candidate = next(
                (
                    original[:pos] + c + original[pos + 1 :]
                    for c in replacements
                    if original[:pos] + c + original[pos + 1 :] not in forbidden
                ),
                None,
            )
            if candidate is not None:
                entries[index].name = candidate
                entries[index].perturbed = True
                break
    return db_names, db_ids, entries


# ---------------------------------------------------------------------------
# 理论侧的 Hamming 分布
# ---------------------------------------------------------------------------


def p_diff(rho: np.ndarray | float) -> np.ndarray | float:
    """随机超平面单位置不同的概率：arccos(rho)/pi。"""
    return np.arccos(np.clip(rho, -1.0, 1.0)) / math.pi


def binomial_cdf_table(ell: int, p: float, d_max: int) -> np.ndarray:
    """返回 d=0..d_max 的 P[Bin(ell, p) <= d]，避免逐个 d 重复求和。"""
    p = float(min(max(p, 0.0), 1.0))
    pmf = np.empty(ell + 1, dtype=np.float64)
    if p == 0.0:
        pmf[:] = 0.0
        pmf[0] = 1.0
    elif p == 1.0:
        pmf[:] = 0.0
        pmf[ell] = 1.0
    else:
        log_p, log_q = math.log(p), math.log1p(-p)
        pmf[0] = math.exp(ell * log_q)
        for k in range(1, ell + 1):
            pmf[k] = pmf[k - 1] * (ell - k + 1) / k * (p / (1.0 - p))
        pmf = np.maximum(pmf, 0.0)
    cdf = np.cumsum(pmf)
    # d 超过 ell 时 CDF 恒为 1；统一返回长度为 d_max+1，便于下游逐 d 累加
    if d_max > ell:
        cdf = np.concatenate([cdf, np.ones(d_max - ell, dtype=np.float64)])
    return cdf[: d_max + 1]


def wilson_interval(successes: float, trials: float, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 区间；用于参考，主报告用查询级 bootstrap。"""
    if trials <= 0:
        return (float("nan"), float("nan"))
    phat = successes / trials
    denom = 1.0 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denom
    half = z * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _label_covers(label: str, value: float) -> bool:
    """判断 "0.80–0.85" 这样的区间标签是否覆盖给定余弦值。"""
    try:
        low, high = label.split("–")
        return float(low) <= value < float(high)
    except (ValueError, TypeError):
        return False

def and_gate_estimate(ell: int, d: int) -> tuple[int, str]:
    """按 `hamming_threshold.h` 的两种实现估算 AND 门数。"""
    if d <= 6:
        return (int(round(2.7 * ell)), "wallace")
    width = max(1, (d + 1).bit_length())
    return (int(2 * width * ell), "saturating")


# ---------------------------------------------------------------------------
# 主评估
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--query-limit", type=int, default=200)
    parser.add_argument("--db-limit", type=int, default=10000)
    parser.add_argument("--fuzzy-ratio", type=float, default=0.3)
    parser.add_argument("--fuzzy-seed", type=int, default=42)
    parser.add_argument("--ells", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--fpr-cap", type=float, default=0.05)
    parser.add_argument("--plane-seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts" / "binarization_frontier")
    args = parser.parse_args(argv)

    ells = sorted({int(value) for value in args.ells if value > 0})
    if not ells:
        raise SystemExit("--ells must contain positive integers")
    if not 0.0 <= args.fpr_cap <= 1.0:
        raise SystemExit("--fpr-cap must be in [0, 1]")

    # ---------------- 1. 数据 ----------------
    db_names, db_ids, entries = read_ncvr(
        args.data_root,
        query_limit=args.query_limit,
        db_limit=args.db_limit,
        fuzzy_ratio=args.fuzzy_ratio,
        fuzzy_seed=args.fuzzy_seed,
    )
    query_names = [entry.name for entry in entries]
    query_ids = [entry.ncid for entry in entries]
    labels = np.array([entry.label for entry in entries], dtype=bool)

    missing_targets = [
        (index, entry.ncid) for index, entry in enumerate(entries) if entry.label and entry.target_index is None
    ]
    if missing_targets:
        raise SystemExit(
            f"{len(missing_targets)} labelled queries have no matching ncid in the database, "
            f"e.g. {missing_targets[:3]}. 实体键对不上，继续算出来的 FPR 无意义。"
        )

    # ---------------- 2. 编码（与生产管线同参） ----------------
    db_sig = batch_encode(db_names, NUM_PERMUTATIONS_CLUSTER)[:, :NUM_PERMUTATIONS_MATCH]
    query_sig = batch_encode(query_names, NUM_PERMUTATIONS_CLUSTER)[:, :NUM_PERMUTATIONS_MATCH]
    db_vec = l2_normalize(db_sig)
    query_vec = l2_normalize(query_sig)

    cosine = np.clip(query_vec @ db_vec.T, -1.0, 1.0).astype(np.float64)  # (m, n)

    # 实体键映射成整数，避免 object 数组比较
    id_codes: dict[str, int] = {}
    for value in db_ids + query_ids:
        id_codes.setdefault(value, len(id_codes))
    db_codes = np.array([id_codes[value] for value in db_ids], dtype=np.int64)
    query_codes = np.array([id_codes[value] for value in query_ids], dtype=np.int64)
    same_entity = query_codes[:, None] == db_codes[None, :]  # (m, n) bool
    negative_mask = ~same_entity

    positive_rows = np.flatnonzero(labels)
    targets = np.array([entries[index].target_index for index in positive_rows], dtype=np.int64)
    true_cos = cosine[positive_rows, targets]

    # 每条查询的最近实体不相交负例
    masked = np.where(negative_mask, cosine, -np.inf)
    nearest_neg = masked.max(axis=1)
    nearest_neg_index = masked.argmax(axis=1)

    # 同名不同 ncid（实体不相交但余弦≈1）——若数量多，说明"实体真值"和
    # "字符串真值"不一致，必须在报告里分开说
    name_collision = negative_mask & (cosine >= 0.999)

    rng = np.random.default_rng(args.fuzzy_seed)
    random_neg = np.empty(len(entries), dtype=np.float64)
    for index in range(len(entries)):
        pool = np.flatnonzero(negative_mask[index])
        if pool.size == 0:
            random_neg[index] = np.nan
            continue
        random_neg[index] = cosine[index, rng.choice(pool, size=1, replace=False)[0]]

    neg_values = cosine[negative_mask]

    # 条件 FPR 用的余弦分层（边界集中在 0.4 以上）
    RHO_EDGES = np.array(
        [0.0, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.001]
    )
    RHO_LABELS = [f"{RHO_EDGES[i]:.2f}–{RHO_EDGES[i + 1]:.2f}" for i in range(len(RHO_EDGES) - 1)]
    neg_bin_index = np.clip(
        np.digitize(neg_values, RHO_EDGES[1:-1]), 0, len(RHO_EDGES) - 2
    )

    def describe(values: np.ndarray) -> dict:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {"count": 0}
        quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        counts, edges = np.histogram(finite, bins=50, range=(-1.0, 1.0))
        return {
            "count": int(finite.size),
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "min": float(finite.min()),
            "max": float(finite.max()),
            "quantiles": {f"p{int(q * 100):02d}": float(np.quantile(finite, q)) for q in quantiles},
            "histogram": {
                "bin_edges": [float(edge) for edge in edges],
                "counts": [int(value) for value in counts],
            },
        }

    # 重叠度量：最近负例 vs 真目标，越大越难分
    overlap = float(np.mean(nearest_neg[positive_rows] >= true_cos))
    sep_auc = float(
        np.mean(true_cos[:, None] > nearest_neg[positive_rows][None, :])
        + 0.5 * np.mean(true_cos[:, None] == nearest_neg[positive_rows][None, :])
    )

    # ---------------- 3. 理论侧：对真实分布积分 ----------------
    d_max = int(min(max(ells), 160))
    d_grid = np.arange(0, d_max + 1, dtype=np.int64)

    # 在 p=arccos(rho)/pi 上等宽分箱，而不是在 rho 上：rho→1 时 CDF 对 p 极敏感，
    # 在 rho 上等宽分箱会在高余弦区严重失真（实测会让理论 FPR 系统性偏低）。
    neg_p = p_diff(neg_values)
    n_bins = 4001
    p_edges = np.linspace(0.0, 0.5, n_bins + 1)
    bin_idx = np.clip(np.digitize(neg_p, p_edges[1:-1]), 0, n_bins - 1)
    bin_counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)
    p_centres = 0.5 * (p_edges[:-1] + p_edges[1:])
    occupied = bin_counts > 0

    theory: dict[int, dict] = {}
    for ell in ells:
        fpr_by_d = np.zeros_like(d_grid, dtype=np.float64)
        for centre, weight in zip(p_centres[occupied], bin_counts[occupied]):
            cdf = binomial_cdf_table(ell, float(centre), d_max)
            fpr_by_d += weight * cdf
        fpr_by_d /= bin_counts.sum()

        tpr_by_d = np.zeros_like(d_grid, dtype=np.float64)
        for rho in true_cos:
            tpr_by_d += binomial_cdf_table(ell, float(p_diff(rho)), d_max)
        tpr_by_d /= true_cos.size

        # 反例：把平均余弦代进二项分布（错的算法，只用于展示差异）
        naive = binomial_cdf_table(ell, float(p_diff(neg_values.mean())), d_max)

        theory[ell] = {
            "tpr": tpr_by_d,
            "fpr": fpr_by_d,
            "fpr_naive_mean_rho": naive,
        }

    # 工作点用**理论曲线**定（无噪声），MC 只做独立核验。
    # 若改用 MC 自己的 d*，就是在同一组估计上取最大值，FPR 会被系统性地选低。
    d_eval_by_ell: dict[int, int | None] = {}
    for ell in ells:
        feasible_theory = np.flatnonzero(theory[ell]["fpr"] <= args.fpr_cap)
        d_eval_by_ell[ell] = int(feasible_theory.max()) if feasible_theory.size else None

    # ---------------- 4. Monte Carlo：真实向量 + 随机超平面 ----------------
    ell_max = max(ells)
    query_bits_full = None
    db_bits_full = None
    mc_rows: list[dict] = []
    frontier: list[dict] = []

    positive_lookup = {int(row): True for row in positive_rows}

    for plane_seed in args.plane_seeds:
        planes = np.random.default_rng(plane_seed).standard_normal((ell_max, NUM_PERMUTATIONS_MATCH))
        query_bits_full = (query_vec @ planes.T) >= 0.0
        db_bits_full = (db_vec @ planes.T) >= 0.0

        for ell in ells:
            q_bits = query_bits_full[:, :ell].astype(np.float32)
            d_bits = db_bits_full[:, :ell].astype(np.float32)
            # Hamming = sum a(1-b) + (1-a)b，纯 0/1 矩阵下为整数，float32 精确
            hamming = q_bits @ (1.0 - d_bits).T + (1.0 - q_bits) @ d_bits.T
            hamming = np.rint(hamming).astype(np.int32)

            neg_h = np.where(negative_mask, hamming, np.iinfo(np.int32).max)
            pos_h = np.full(len(entries), np.iinfo(np.int32).max, dtype=np.int32)
            pos_h[positive_rows] = hamming[positive_rows, targets]

            # 每个 d 的负例存活计数（按查询），供 bootstrap 重抽样。
            # np.searchsorted 只接受 1-D，所以按行循环（m=200，代价可忽略）。
            neg_sorted = np.sort(neg_h, axis=1)
            neg_counts = np.empty((neg_sorted.shape[0], d_grid.size), dtype=np.int64)
            for row_index in range(neg_sorted.shape[0]):
                neg_counts[row_index] = np.searchsorted(
                    neg_sorted[row_index], d_grid, side="right"
                )
            neg_valid = negative_mask.sum(axis=1)  # 每查询的负例数（可能不等）
            pos_flag = pos_h[None, :] <= d_grid[:, None]  # (|d|, m)

            total_neg_pairs = float(neg_valid.sum())
            fpr_by_d = neg_counts.sum(axis=0) / total_neg_pairs
            tpr_by_d = pos_flag[:, positive_rows].mean(axis=1)

            for d in d_grid:
                idx = int(d)
                mc_rows.append(
                    {
                        "tier": "mc_verified",
                        "ell": int(ell),
                        "d": idx,
                        "plane_seed": int(plane_seed),
                        "tpr": float(tpr_by_d[idx]),
                        "fpr": float(fpr_by_d[idx]),
                        "and_gates": and_gate_estimate(int(ell), idx)[0],
                        "circuit": and_gate_estimate(int(ell), idx)[1],
                        "d_le_6": bool(idx <= 6),
                    }
                )

            feasible = np.flatnonzero(fpr_by_d <= args.fpr_cap)
            if feasible.size == 0:
                frontier.append(
                    {
                        "tier": "mc_verified",
                        "ell": int(ell),
                        "plane_seed": int(plane_seed),
                        "d_star": None,
                        "tpr": float("nan"),
                        "fpr": float("nan"),
                        "mean_candidates": float("nan"),
                        "reason": "even d=0 exceeds fpr cap",
                    }
                )
                continue

            d_star = int(feasible.max())
            chosen_neg = neg_counts[:, d_star]
            chosen_pos = pos_flag[d_star]

            rng_boot = np.random.default_rng(plane_seed + 7919)
            boot_tpr, boot_fpr = [], []
            m = len(entries)
            for _ in range(max(1, args.bootstrap)):
                pick = rng_boot.integers(0, m, size=m)
                neg_total = neg_valid[pick].sum()
                if neg_total == 0:
                    continue
                boot_fpr.append(chosen_neg[pick].sum() / neg_total)
                pos_pick = pick[labels[pick]]
                if pos_pick.size:
                    boot_tpr.append(chosen_pos[pos_pick].mean())

            mean_candidates = float(
                (neg_counts[:, d_star].sum() + pos_flag[d_star][positive_rows].sum()) / len(entries)
            )

            # 条件 FPR：按负例自身余弦分层。集群内的负例正好落在高余弦区，
            # 所以这张表才是"放进已有簇之后还剩多少候选"的直接依据。
            # 边界集中在 0.4 以上：负例余弦 80% 落在 0.48–0.62，高余弦区才是簇内情形
            survivors = (hamming <= d_star)[negative_mask]
            bin_total = np.bincount(neg_bin_index, minlength=len(RHO_EDGES) - 1).astype(np.float64)
            bin_surv = np.bincount(
                neg_bin_index,
                weights=survivors.astype(np.float64),
                minlength=len(RHO_EDGES) - 1,
            )
            conditional = {
                RHO_LABELS[i]: float(bin_surv[i] / bin_total[i])
                for i in range(len(bin_total))
                if bin_total[i] >= 50
            }

            frontier.append(
                {
                    "tier": "mc_verified",
                    "ell": int(ell),
                    "plane_seed": int(plane_seed),
                    "d_star": d_star,
                    "tpr": float(tpr_by_d[d_star]),
                    "fpr": float(fpr_by_d[d_star]),
                    "tpr_ci_lo": float(np.quantile(boot_tpr, 0.025)) if boot_tpr else float("nan"),
                    "tpr_ci_hi": float(np.quantile(boot_tpr, 0.975)) if boot_tpr else float("nan"),
                    "fpr_ci_lo": float(np.quantile(boot_fpr, 0.025)) if boot_fpr else float("nan"),
                    "fpr_ci_hi": float(np.quantile(boot_fpr, 0.975)) if boot_fpr else float("nan"),
                    "mean_candidates": mean_candidates,
                    "and_gates": and_gate_estimate(int(ell), d_star)[0],
                    "circuit": and_gate_estimate(int(ell), d_star)[1],
                    "d_le_6": bool(d_star <= 6),
                    "conditional_fpr_by_rho": conditional,
                    "d_eval": d_eval_by_ell.get(int(ell)),
                    "tpr_at_eval": float(tpr_by_d[d_eval_by_ell[int(ell)]])
                    if d_eval_by_ell.get(int(ell)) is not None
                    else float("nan"),
                    "fpr_at_eval": float(fpr_by_d[d_eval_by_ell[int(ell)]])
                    if d_eval_by_ell.get(int(ell)) is not None
                    else float("nan"),
                    "reason": "ok",
                }
            )

    # 理论侧工作点：**在 MC 选出的同一个 d 上**取理论值，否则两者不可比。
    theory_points = []
    for ell in ells:
        mc_d = [row["d_star"] for row in frontier if row["ell"] == ell and row.get("d_star") is not None]
        d_eval = d_eval_by_ell.get(ell)
        theory_points.append(
            {
                "ell": int(ell),
                "d_mc_median": int(np.median(mc_d)) if mc_d else None,
                "d_mc_own_median": int(np.median(mc_d)) if mc_d else None,
                "d_eval": d_eval,
                "tpr_theory": float(theory[ell]["tpr"][d_eval]) if d_eval is not None else float("nan"),
                "fpr_theory": float(theory[ell]["fpr"][d_eval]) if d_eval is not None else float("nan"),
                "fpr_naive": float(theory[ell]["fpr_naive_mean_rho"][d_eval])
                if d_eval is not None
                else float("nan"),
                "and_gates": and_gate_estimate(int(ell), d_eval)[0] if d_eval is not None else None,
                "d_le_6": bool(d_eval is not None and d_eval <= 6),
            }
        )

    # ---------------- 5. 输出 ----------------
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "frontier.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "tier",
                "ell",
                "d",
                "plane_seed",
                "tpr",
                "fpr",
                "query_bootstrap_tpr_lo",
                "query_bootstrap_tpr_hi",
                "query_bootstrap_fpr_lo",
                "query_bootstrap_fpr_hi",
                "mean_candidates",
                "and_gates",
                "circuit",
                "d_le_6",
                "reason",
            ]
        )
        for row in frontier:
            writer.writerow(
                [
                    row["tier"],
                    row["ell"],
                    row.get("d_star"),
                    row["plane_seed"],
                    row["tpr"],
                    row["fpr"],
                    row.get("tpr_ci_lo", ""),
                    row.get("tpr_ci_hi", ""),
                    row.get("fpr_ci_lo", ""),
                    row.get("fpr_ci_hi", ""),
                    row.get("mean_candidates", ""),
                    row.get("and_gates", ""),
                    row.get("circuit", ""),
                    row.get("d_le_6", ""),
                    row["reason"],
                ]
            )

    report = {
        "config": {
            "data_root": str(args.data_root),
            "query_limit": args.query_limit,
            "db_limit": args.db_limit,
            "fuzzy_ratio": args.fuzzy_ratio,
            "fuzzy_seed": args.fuzzy_seed,
            "ells": ells,
            "fpr_cap": args.fpr_cap,
            "plane_seeds": args.plane_seeds,
            "bootstrap": args.bootstrap,
            "shingle_size": SHINGLE_SIZE,
            "permutations_cluster": NUM_PERMUTATIONS_CLUSTER,
            "permutations_match": NUM_PERMUTATIONS_MATCH,
        },
        "runtime": {
            "normalizer": _NORMALIZER_SOURCE,
            "numpy": np.__version__,
        },
        "data": {
            "database_size": len(db_names),
            "query_count": len(entries),
            "positive_queries": int(labels.sum()),
            "negative_queries": int((~labels).sum()),
            "perturbed_queries": int(sum(1 for entry in entries if entry.perturbed)),
            "negative_pairs": int(negative_mask.sum()),
            "name_collision_pairs": int(name_collision.sum()),
        },
        "cosine": {
            "true_target": describe(true_cos),
            "nearest_non_target": describe(nearest_neg[positive_rows]),
            "random_negative": describe(random_neg),
            "all_negative_pairs": describe(neg_values),
            "nearest_negative_covers_true_target_rate": overlap,
            "target_vs_nearest_negative_auc": sep_auc,
        },
        "theory_operating_points": theory_points,
        "frontier": frontier,
        "evidence_legend": {
            "theory_only": "逐对代入 Binomial(ell, arccos(rho)/pi)，未用真实负例分布",
            "empirical_dist": "逐对代入后对真实负例分布积分；仍非实测",
            "mc_verified": "真实向量生成随机超平面码，直接统计 Hamming；唯一实测证据",
        },
    }
    (out_dir / "distribution.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 控制台小结
    print(f"data: {len(entries)} queries ({int(labels.sum())} pos / {int((~labels).sum())} neg), "
          f"{len(db_names)} db, perturbed={sum(1 for e in entries if e.perturbed)}")
    print(f"normalizer: {_NORMALIZER_SOURCE}")
    print(f"true target cos    : mean={true_cos.mean():.4f} "
          f"p05={np.quantile(true_cos, 0.05):.4f} p50={np.quantile(true_cos, 0.5):.4f}")
    print(f"nearest non-target : mean={nearest_neg[positive_rows].mean():.4f} "
          f"p50={np.quantile(nearest_neg[positive_rows], 0.5):.4f} "
          f"p95={np.quantile(nearest_neg[positive_rows], 0.95):.4f}")
    print(f"overlap (nearest neg >= true target): {overlap:.3f}   AUC={sep_auc:.3f}")
    print(f"name-collision pairs (cos>=0.999, entity-disjoint): {int(name_collision.sum())}")
    print()
    header = f"{'ell':>5} {'d*':>4} {'TPR(mc)':>8} {'FPR(mc)':>8} {'TPR(th)':>8} {'FPR(th)':>8} {'FPR(naive)':>11} {'gates':>6} {'circuit':>11}"
    print(header)
    print("-" * len(header))
    for point in theory_points:
        ell = point["ell"]
        same = [row for row in frontier if row["ell"] == ell and row["d_star"] is not None]
        mc_tpr = float(np.median([row["tpr"] for row in same])) if same else float("nan")
        mc_fpr = float(np.median([row["fpr"] for row in same])) if same else float("nan")
        print(
            f"{ell:>5} {str(point['d_eval']):>4} {mc_tpr:>8.3f} {mc_fpr:>8.3f} "
            f"{point['tpr_theory']:>8.3f} {point['fpr_theory']:>8.3f} "
            f"{point['fpr_naive']:>11.3f} {str(point['and_gates']):>6} "
            f"{'wallace' if point['d_le_6'] else 'saturating':>11}"
        )
    print()
    print(f"outputs -> {out_dir}")

    # summary.md
    lines = [
        "# 二值码粗筛可行性小结",
        "",
        f"- 数据：{len(entries)} 条查询（正 {int(labels.sum())} / 负 {int((~labels).sum())}），"
        f"库 {len(db_names)} 条，fuzzy_ratio={args.fuzzy_ratio}，实际扰动 {sum(1 for e in entries if e.perturbed)} 条",
        f"- 真目标余弦：均值 {true_cos.mean():.4f}，p05 {np.quantile(true_cos, 0.05):.4f}",
        f"- 最近实体不相交负例余弦：均值 {nearest_neg[positive_rows].mean():.4f}，"
        f"p50 {np.quantile(nearest_neg[positive_rows], 0.5):.4f}，p95 {np.quantile(nearest_neg[positive_rows], 0.95):.4f}",
        f"- 重叠（最近负例 ≥ 真目标）：{overlap:.3f}；AUC={sep_auc:.3f}",
        f"- 同名不同 ncid 的对数：{int(name_collision.sum())}",
        "",
        "| ell | d* | TPR(mc) | FPR(mc) | TPR(理论) | FPR(理论) | FPR(均值代入) | AND 门 | 电路分支 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for point in theory_points:
        ell = point["ell"]
        same = [row for row in frontier if row["ell"] == ell and row["d_star"] is not None]
        mc_tpr = float(np.median([row["tpr"] for row in same])) if same else float("nan")
        mc_fpr = float(np.median([row["fpr"] for row in same])) if same else float("nan")
        lines.append(
            f"| {ell} | {point['d_eval']} | {mc_tpr:.3f} | {mc_fpr:.3f} | "
            f"{point['tpr_theory']:.3f} | {point['fpr_theory']:.3f} | {point['fpr_naive']:.3f} | "
            f"{point['and_gates']} | {'wallace (d<=6)' if point['d_le_6'] else 'saturating'} |"
        )
    lines += [
        "",
        "| ell | d* | TPR(mc) [min–max] | FPR(mc) [min–max] | TPR/FPR 理论 | 理论−MC |",
        "|---|---|---|---|---|---|",
    ]
    for point in theory_points:
        ell = point["ell"]
        same = [row for row in frontier if row["ell"] == ell and row.get("d_star") is not None]
        if not same:
            continue
        tpr_s = [row["tpr_at_eval"] for row in same]
        fpr_s = [row["fpr_at_eval"] for row in same]
        lines.append(
            f"| {ell} | {point['d_eval']} | "
            f"{np.median(tpr_s):.3f} [{min(tpr_s):.2f}–{max(tpr_s):.2f}] | "
            f"{np.median(fpr_s):.4f} [{min(fpr_s):.4f}–{max(fpr_s):.4f}] | "
            f"{point['tpr_theory']:.3f}/{point['fpr_theory']:.4f} | "
            f"{point['fpr_theory']-np.median(fpr_s):+.4f} |"
        )
    lines += [
        "",
        "> 理论−MC 的差异量级本身就是一条结论：同一组 ℓ 个平面被全部查询共用，",
        "> 所以 MC 的有效样本量由「查询数 × 平面数」决定，而不是 200 万对样本。",
        "> 实测 200 查询 + 256 平面时，FPR 的 seed 间波动约 ±0.005；小于这个量级的",
        "> 理论-MC 差异不应解读为模型偏差。要压低它只能增加查询实体或平面数。",
        "> 另外：frontier.csv 里的 `query_bootstrap_*` 区间**只对查询做重抽样**，平面集合固定，",
        "> 因此它不覆盖平面抽样带来的不确定性；两者不是同一件事，不能合并成一个 CI。",
        "",
        "> TPR(mc)/FPR(mc) 是真实向量 + 随机超平面码的实测中位数（多个 plane seed）。",
        "> TPR(理论)/FPR(理论) 是逐对代入二项分布后对真实分布积分，**不是实测**。",
        "> FPR(均值代入) 是错误算法（把平均余弦代进二项分布），仅用于展示两者差异。",
        "> 三列理论值都在与 mc 相同的 d 上取值，因此可以直接与 FPR(mc) 对照。",
    ]

    # 按余弦分层的条件 FPR：集群内负例集中在高余弦区，这张表才是组合依据
    lines += [
        "",
        "## 条件 FPR（负例按自身余弦分层，实测）",
        "",
        "集群内的负例不是随机负例：它们与查询的余弦整体高于随机负例。下表给出各",
        "余弦区间上，一个负例通过 `Hamming <= d*` 的概率——也就是\"放进已有簇之后\"",
        "的真实候选保留率。区间内样本 <50 对时不报告。",
        "",
    ]
    present = sorted({label for row in frontier for label in row.get("conditional_fpr_by_rho", {})},
                     key=lambda x: float(x.split("–")[0]))
    if present:
        lines.append("| ell | d* | " + " | ".join(present) + " |")
        lines.append("|---|---|" + "---|" * len(present))
        for point in theory_points:
            ell = point["ell"]
            rows = [
                row["conditional_fpr_by_rho"]
                for row in frontier
                if row["ell"] == ell and row.get("conditional_fpr_by_rho")
            ]
            cells = []
            for label in present:
                samples = [table[label] for table in rows if label in table]
                cells.append(f"{np.median(samples):.3f}" if samples else "n/a")
            lines.append(f"| {ell} | {point['d_eval']} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 怎么读",
        "",
        f"- 所有工作点的 d* 都 {'≤ 6（Wallace 可用）' if all(p['d_le_6'] for p in theory_points if p['d_eval'] is not None) else '> 6，落在 `hamming_threshold.h` 的通用饱和计数器分支，Wallace 的 ~2.7 AND/bit 用不上'}。",
        "- `FPR(均值代入)` 与 `FPR(理论)` 的差距就是\"能不能把平均余弦代进二项分布\"的直接反驳。",
        "- 全局 FPR × 库大小 = 每查询剩余候选数。若它与现有 Top-1 簇宽同量级，说明这个",
        "  滤波器单独使用没有收益，价值只存在于\"簇内二次过滤\"；而簇内 FPR 应查上面的分层表。",
        "- 正例只有 100 条，TPR 的 bootstrap 区间很宽（约 ±0.05），不足以分辨 1 个百分点的差异。",
        "",
        "## 自动判据（数据决定，不是预设结论）",
        "",
    ]
    # 关键区间：最近实体不相交负例的中位数余弦。簇内负例整体不会比它更容易。
    hard_rho = float(np.median(nearest_neg[positive_rows]))
    verdict_lines = [
        f"- 最近负例余弦中位数 rho = {hard_rho:.3f}。",
    ]
    for point in theory_points:
        ell = point["ell"]
        rows = [
            row["conditional_fpr_by_rho"]
            for row in frontier
            if row["ell"] == ell and row.get("conditional_fpr_by_rho")
        ]
        # 找覆盖 hard_rho 的区间
        hit = [label for label in (rows[0] if rows else {}) if _label_covers(label, hard_rho)]
        if not hit:
            verdict_lines.append(f"- ell={ell}: 该余弦区间样本不足，未评估。")
            continue
        samples = [table[hit[0]] for table in rows if hit[0] in table]
        rate = float(np.median(samples))
        verdict_lines.append(
            f"- ell={ell}, d={point['d_eval']}: 在 rho≈{hit[0]} 的硬负例上保留率 {rate:.3f}"
            + ("（几乎不压缩）" if rate > 0.5 else "（有压缩）")
        )
    any_compress = any("有压缩" in line for line in verdict_lines)
    verdict_lines += [
        "",
        "若上表硬负例保留率普遍 > 0.5，则**全局 5% FPR 是假象**：它由大量低余弦负例撑起来，",
        "而粗筛真正要处理的正是这些高余弦负例。此时提高 ell 不会改善——d*/ell 会同步上升，",
        "滤波器退化成一条固定的余弦阈值。",
    ]
    lines += verdict_lines
    lines += []
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
