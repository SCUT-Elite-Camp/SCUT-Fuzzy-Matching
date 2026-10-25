#!/usr/bin/env python3
"""可复核性验证器：把报告中每一条数字重新算一遍，与归档结果比对。

目的不是再跑实验，而是让**任何人**能在几分钟内独立确认报告里的数字是否可复现。
每条 claim 显式给出：期望值来源（归档文件）、重算方式、容差、证据等级。

## 分级

    quick     纯 numpy（约 3–5 秒）：召回、布局、tile 算术、适配层阻断
    tenseal   追加 CKKS 实测（约 3 秒）：R3 标志重构、单形状 tile 数
    full      追加 4 形状 × N 次 kernel 计时（默认 N=3，约 8 分钟）

## 用法

    python scripts/verify_claims.py                    # quick
    python scripts/verify_claims.py --level tenseal    # quick + CKKS
    python scripts/verify_claims.py --level full --repeats 3

退出码：0 = 全部通过；1 = 有失败；2 = 依赖缺失（未做任何断言）。

## 不做的事

不改生产基线；不使用 MPC/OT；不探索新参数（所有参数取自归档实验的固定值）。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SLOT_CAPACITY = 4096
BATCH = 200
CLUSTERS = 50
ELL = 512
D_THRESHOLD = 109
PLANE_SEED = 2026


@dataclass
class Claim:
    cid: str
    what: str
    expected: object
    actual: object = None
    ok: bool = False
    note: str = ""
    level: str = "quick"


@dataclass
class Report:
    claims: list[Claim] = field(default_factory=list)

    def add(self, **kw) -> Claim:
        claim = Claim(**kw)
        self.claims.append(claim)
        return claim

    def failed(self) -> list[Claim]:
        return [c for c in self.claims if not c.ok]


def recompute_recall() -> dict:
    """复算原始 / 均衡两种布局的召回指标（与归档实验同一套参数）。"""
    from config.params import NUM_PERMUTATIONS_CLUSTER as K200, NUM_PERMUTATIONS_MATCH as D50
    from minhash.encoder import batch_encode
    from scripts.evaluate_binarization_frontier import l2_normalize, read_ncvr
    from scripts.evaluate_cluster_conditional_filter import cosine_kmeans, select_clusters
    from fusion.evaluate_balanced_layout import balanced_kmeans, evaluate

    db_names, _ids, entries = read_ncvr(
        ROOT / "data", query_limit=200, db_limit=10000, fuzzy_ratio=0.3, fuzzy_seed=42
    )
    db200 = batch_encode(db_names, K200)
    q200 = batch_encode([e.name for e in entries], K200)
    db50, q50 = l2_normalize(db200[:, :D50]), l2_normalize(q200[:, :D50])

    mu = l2_normalize(db200).mean(axis=0)
    sd = l2_normalize(db200).std(axis=0, ddof=0)
    sd = np.where(sd < 1e-10, 1.0, sd)
    db_std = (l2_normalize(db200) - mu) / sd
    q_std = (l2_normalize(q200) - mu) / sd

    planes = np.random.default_rng(PLANE_SEED).standard_normal((ELL, D50))
    db_codes, q_codes = (db50 @ planes.T) >= 0, (q50 @ planes.T) >= 0

    labels = np.array([e.label for e in entries], dtype=bool)
    targets = np.array([e.target_index if e.target_index is not None else -1 for e in entries])

    def survivors_of(assign, sel):
        out = []
        for i in range(len(entries)):
            m = np.flatnonzero(assign == sel[i])
            dist = (q_codes[i, :ELL][None, :] != db_codes[m, :ELL]).sum(axis=1)
            out.append(m[dist <= D_THRESHOLD])
        return out

    cent_a, assign_a = cosine_kmeans(db_std, CLUSTERS, seed=42)
    sel_a = select_clusters(q_std, cent_a)
    ra = evaluate(assign_a, sel_a, survivors_of(assign_a, sel_a), labels, targets)

    cap = math.ceil(len(db_names) / CLUSTERS)
    cent_b, assign_b = balanced_kmeans(db_std, CLUSTERS, cap, seed=42)
    sel_b = select_clusters(q_std, cent_b)
    rb = evaluate(assign_b, sel_b, survivors_of(assign_b, sel_b), labels, targets)

    sizes_a = np.bincount(assign_a, minlength=CLUSTERS)
    sizes_b = np.bincount(assign_b, minlength=CLUSTERS)
    pos = labels
    return {
        "layout_a": {"max_size": int(sizes_a.max()), "tiles": math.ceil(int(sizes_a.max()) / (SLOT_CAPACITY // BATCH)),
                     "top1_recall": ra["top1_recall"], "in_cluster_filter_recall": ra["in_cluster_filter_recall"],
                     "end_to_end_recall": ra["end_to_end_recall"]},
        "layout_b": {"max_size": int(sizes_b.max()), "tiles": math.ceil(int(sizes_b.max()) / (SLOT_CAPACITY // BATCH)),
                     "top1_recall": rb["top1_recall"], "in_cluster_filter_recall": rb["in_cluster_filter_recall"],
                     "end_to_end_recall": rb["end_to_end_recall"]},
        "paired": {"new_misses": int(np.sum(ra["hit_mask"][pos] & ~rb["hit_mask"][pos])),
                   "fixed": int(np.sum(~ra["hit_mask"][pos] & rb["hit_mask"][pos]))},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", choices=["quick", "tenseal", "full"], default="quick")
    ap.add_argument("--repeats", type=int, default=3, help="full 模式下每种形状的重复次数")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "verification" / "report.json")
    args = ap.parse_args()

    rep = Report()
    t_start = time.perf_counter()

    # ---------- C1：tile 算术（纯常量） ----------
    T = SLOT_CAPACITY // BATCH
    rep.add(cid="C1a", what="tile 宽度 T = 4096//200", expected=20, actual=T, ok=(T == 20))
    rep.add(cid="C1b", what="max_size=483 → 25 tiles", expected=25,
            actual=math.ceil(483 / T), ok=(math.ceil(483 / T) == 25))
    rep.add(cid="C1c", what="max_size=200 → 10 tiles", expected=10,
            actual=math.ceil(200 / T), ok=(math.ceil(200 / T) == 10))

    # ---------- C2/C3：召回与布局（重算 vs 归档） ----------
    archive_path = ROOT / "artifacts" / "balanced_layout" / "recall_compare.json"
    if not archive_path.exists():
        rep.add(cid="C2", what="归档召回文件存在", expected=str(archive_path), actual=None,
                ok=False, note="缺少归档文件，无法比对")
    else:
        archived = json.loads(archive_path.read_text(encoding="utf-8"))
        fresh = recompute_recall()
        for layout in ("layout_a", "layout_b"):
            for key in ("max_size", "tiles", "top1_recall", "in_cluster_filter_recall", "end_to_end_recall"):
                exp = archived[layout][key]
                got = fresh[layout][key]
                tol = 0 if isinstance(exp, int) else 1e-9
                rep.add(cid=f"C2-{layout}-{key}", what=f"{layout}.{key} 与归档一致",
                        expected=exp, actual=got, ok=(abs(float(exp) - float(got)) <= tol))
        for key in ("new_misses", "fixed"):
            exp = archived["paired"][key]
            got = fresh["paired"][key]
            rep.add(cid=f"C3-paired-{key}", what=f"paired.{key} 与归档一致",
                    expected=exp, actual=got, ok=(exp == got))

    # ---------- C4：适配层阻断 ----------
    try:
        from fusion.adapter import CandidateEnvelope, FusionBlocked, adapt
        from fusion.database_layout.build_candidate_blocks import build_blocks

        layout = build_blocks(n=40, block_size=10, strategy="stored")
        try:
            adapt(CandidateEnvelope("approx_psi", "secret_share", share_bits=1),
                  layout=layout, max_size=10, clusters=4, dim=50, batch_size=BATCH)
            rep.add(cid="C4", what="secret_share 必须被阻断", expected="FusionBlocked",
                    actual="未抛出", ok=False)
        except FusionBlocked:
            rep.add(cid="C4", what="secret_share 被 FusionBlocked 阻断", expected="FusionBlocked",
                    actual="FusionBlocked", ok=True)
    except Exception as exc:  # noqa: BLE001
        rep.add(cid="C4", what="适配层阻断检查", expected="FusionBlocked", actual=repr(exc), ok=False)

    # ---------- C5/C6：CKKS 实测（可选） ----------
    if args.level in {"tenseal", "full"}:
        try:
            import tenseal  # noqa: F401
        except ImportError:
            rep.add(cid="C5", what="TenSEAL 可用", expected="import ok", actual="ImportError",
                    ok=False, level="tenseal", note="未做任何 CKKS 断言")

        else:
            # C5：R3 探针（复用同一脚本，结果与归档比对）
            from fusion.r3_numeric_probe import main as r3_main  # noqa: F401
            r3_path = ROOT / "artifacts" / "r3_probe" / "numeric.json"
            r3_before = json.loads(r3_path.read_text(encoding="utf-8")) if r3_path.exists() else None
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                r3_main()
            r3_after = json.loads(r3_path.read_text(encoding="utf-8"))
            exp_bits = r3_before["result"]["bits_tested"] if r3_before else 96600
            rep.add(cid="C5a", what="R3 重构位数", expected=exp_bits,
                    actual=r3_after["result"]["bits_tested"], level="tenseal",
                    ok=(r3_after["result"]["bits_tested"] == exp_bits))
            rep.add(cid="C5b", what="R3 位错误数 = 0", expected=0,
                    actual=r3_after["result"]["bit_errors"], level="tenseal",
                    ok=(r3_after["result"]["bit_errors"] == 0))
            rep.add(cid="C5c", what="R3 最大误差 < 1e-3", expected="<1e-3",
                    actual=f"{r3_after['result']['max_abs_error']:.3e}", level="tenseal",
                    ok=(r3_after["result"]["max_abs_error"] < 1e-3))

            # C6：单形状 tile 数（kernel 形状断言）
            from scripts.benchmark_he_batching import run_tiled_benchmark
            for cols, want in ((20, 1), (483, 25)):
                res = run_tiled_benchmark(batch_size=BATCH, clusters=CLUSTERS, columns=cols)
                rep.add(cid=f"C6-{cols}", what=f"columns={cols} → {want} tiles", expected=want,
                        actual=res["dimensions"]["round2_tiles"], level="tenseal",
                        ok=(res["dimensions"]["round2_tiles"] == want))

    # ---------- C7：计时可复现性（仅 full） ----------
    if args.level == "full":
        import csv
        summary_path = ROOT / "artifacts" / "tile_scaling" / "summary.csv"
        arch = {}
        if summary_path.exists():
            for row in csv.DictReader(summary_path.open(encoding="utf-8")):
                if row["stage"] == "round2_b_kernel":
                    arch[int(row["tiles"])] = float(row["median"])
        from scripts.benchmark_he_batching import run_tiled_benchmark
        shapes = {20: 1, 100: 5, 200: 10, 483: 25}
        fresh = {}
        for cols, tiles in shapes.items():
            vals = [run_tiled_benchmark(batch_size=BATCH, clusters=CLUSTERS, columns=cols)["timing_sec"]["round2_b_kernel"]
                    for _ in range(args.repeats)]
            fresh[tiles] = statistics.median(vals)
        for tiles in sorted(fresh):
            exp = arch.get(tiles)
            got = fresh[tiles]
            rel = abs(got - exp) / exp if exp else float("inf")
            rep.add(cid=f"C7-{tiles}t", what=f"{tiles} tiles kernel 中位数与归档一致（±15%）",
                    expected=f"{exp:.2f}s" if exp else "n/a", actual=f"{got:.2f}s",
                    level="full", ok=bool(exp) and rel <= 0.15,
                    note=f"相对偏差 {rel*100:.1f}%")
        ordered = [fresh[t] for t in sorted(fresh)]
        rep.add(cid="C7-monotone", what="kernel 时间随 tiles 单调递增",
                expected="单调", actual="单调" if all(a < b for a, b in zip(ordered, ordered[1:])) else "非单调",
                level="full", ok=all(a < b for a, b in zip(ordered, ordered[1:])))

    # ---------- 输出 ----------
    elapsed = time.perf_counter() - t_start
    print("ID | 检查项 | 期望 | 实际 | 结果")
    print("-" * 78)
    for c in rep.claims:
        exp = f"{c.expected:.6f}" if isinstance(c.expected, float) else str(c.expected)
        act = f"{c.actual:.6f}" if isinstance(c.actual, float) else str(c.actual)
        print(f"{c.cid} | {c.what} | {exp} | {act} | {'PASS' if c.ok else 'FAIL'}")

    failed = rep.failed()
    print(f"\n{len(rep.claims) - len(failed)}/{len(rep.claims)} 通过，耗时 {elapsed:.1f}s，级别 {args.level}")
    if failed:
        print("失败项：", ", ".join(c.cid for c in failed))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "level": args.level, "elapsed_seconds": elapsed,
        "claims": [c.__dict__ for c in rep.claims],
        "failed": [c.cid for c in failed],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"outputs -> {args.out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
