#!/usr/bin/env python3
"""特征维 d 与召回/成本的联合设计：用前 d' 维做匹配估计。

selector 乘法数 = k · d'，与 d' 成正比；d'=25 即减半。
代价是相似度估计的方差变大（同样本量的估计器）。

度量（独立集，随机 target，n=200）：
- 正例召回：target 在选中簇且 sim ≥ τ
- 负例误报：负例中 sim ≥ τ 的比例
"""
from __future__ import annotations
import csv, json, random, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
K, TAU, S = 50, 0.9, 4096
FUZZY_RATIO = 0.3

def main() -> int:
    from config.params import NUM_PERMUTATIONS_CLUSTER as K200, NUM_PERMUTATIONS_MATCH as D50
    from minhash.encoder import batch_encode
    from scripts.evaluate_binarization_frontier import clean_name, l2_normalize
    from scripts.evaluate_cluster_conditional_filter import cosine_kmeans, select_clusters

    folder = ROOT / "data" / "ncvr_10k"
    if not folder.exists(): folder = ROOT / "data"
    with (folder / "ncvr_10k_database.csv").open(newline="", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))[:10000]
    db_names = [clean_name(r["full_name"]) for r in rows]
    forbidden = set(db_names)
    db200 = batch_encode(db_names, K200)
    n200 = l2_normalize(db200)
    mu, sd = n200.mean(0), n200.std(0, ddof=0); sd = np.where(sd < 1e-10, 1.0, sd)
    cent, assign = cosine_kmeans((n200 - mu) / sd, K, seed=42)
    sizes = np.bincount(assign, minlength=K)

    def perturb(name, rng):
        positions = [p for p, c in enumerate(name) if "a" <= c <= "z"]
        rng.shuffle(positions)
        for pos in positions:
            reps = [c for c in "abcdefghijklmnopqrstuvwxyz" if c != name[pos]]
            rng.shuffle(reps)
            for c in reps:
                cand = name[:pos] + c + name[pos + 1:]
                if cand not in forbidden: return cand
        return None

    rng = random.Random(2026)
    tgts = rng.sample(range(len(db_names)), 200)
    pert = set(rng.sample(range(len(tgts)), int(len(tgts)*FUZZY_RATIO+0.5)))
    qn, kept = [], []
    for idx, t in enumerate(tgts):
        q = perturb(db_names[t], rng) if idx in pert else db_names[t]
        if q is not None: qn.append(q); kept.append(t)
    q200 = batch_encode(qn, K200)
    q_std = (l2_normalize(q200) - mu) / sd
    sel = select_clusters(q_std, cent)
    in_cluster = np.array([assign[t] == sel[i] for i, t in enumerate(kept)])
    # 负例：所有不在选中簇的记录里，随机抽样估计误报率
    db50 = l2_normalize(db200[:, :D50]); q50 = l2_normalize(q200[:, :D50])

    print(f"扰动 {FUZZY_RATIO:.0%}");print(f"{'d':>4}{'selector乘法/tile':>20}{'相对 d=50':>11}{'正例召回':>10}{'负例误报':>10}")
    out = []
    for d in (5, 10, 15, 20, 25, 30, 40, 50):
        dq = l2_normalize(q50[:, :d]); dbd = l2_normalize(db50[:, :d])
        sim_t = np.array([float(dq[i] @ dbd[t]) for i, t in enumerate(kept)])
        rec = float(np.mean(in_cluster & (sim_t >= TAU)))
        # 负例误报：每个查询在选中簇内非 target 的记录中，sim ≥ τ 的比例
        fp, tot = 0, 0
        for i in range(len(kept)):
            mem = np.flatnonzero(assign == sel[i])
            mem = mem[mem != kept[i]][:200]
            if len(mem) == 0: continue
            s = dbd[mem] @ dq[i]
            fp += int((s >= TAU).sum()); tot += len(mem)
        out.append({"d": d, "selector_mults_per_tile": K * d,
                    "recall": rec, "fpr": fp / max(tot, 1)})
        print(f"{d:>4}{K*d:>20,}{K*d/(K*50):>11.2f}{rec:>10.3f}{fp/max(tot,1):>10.4f}")

    p = ROOT / "artifacts" / "feature_dim" / "report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"n": len(kept), "tau": TAU, "rows": out}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\noutputs -> {p}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
