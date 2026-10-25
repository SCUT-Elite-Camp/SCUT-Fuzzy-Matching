#!/usr/bin/env python3
"""阈值副产品验证：独立 dev 选 τ → 冻结 → 独立 test 评估。

预注册规则（先于看 test）：
  在 dev 上，选使 target 召回最大、且 dev FPR ≤ 0.5% 的 τ；冻结后只在 test 上报一次。

样本：1200 正例（30% 扰动）+ 1200 负例（库外随机串），各半分为 dev / test。
度量：target 召回、查询级 catch 召回、FPR，均给 Wilson 95% CI。
"""
from __future__ import annotations
import csv, json, math, random, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (max(0.0, c-h), min(1.0, c+h))

def main() -> int:
    from config.params import NUM_PERMUTATIONS_CLUSTER as K200
    from minhash.encoder import batch_encode
    from scripts.evaluate_binarization_frontier import clean_name, l2_normalize
    from scripts.evaluate_cluster_conditional_filter import cosine_kmeans, select_clusters

    folder = ROOT/"data"/"ncvr_10k"
    if not folder.exists(): folder = ROOT/"data"
    rows = list(csv.DictReader((folder/"ncvr_10k_database.csv").open(newline="", encoding="utf-8")))[:10000]
    db_names = [clean_name(r["full_name"]) for r in rows]; forbidden = set(db_names)
    rng = random.Random(7)
    def perturb(n, r):
        ps=[p for p,c in enumerate(n) if 'a'<=c<='z']; r.shuffle(ps)
        for pos in ps:
            reps=[c for c in "abcdefghijklmnopqrstuvwxyz" if c!=n[pos]]; r.shuffle(reps)
            for c in reps:
                cand=n[:pos]+c+n[pos+1:]
                if cand not in forbidden: return cand
        return None

    N_POS, N_NEG = 1200, 1200
    tgts = rng.sample(range(len(db_names)), N_POS)
    pert = set(rng.sample(range(N_POS), int(N_POS*0.3)))
    pos_q, kept = [], []
    for i, t in enumerate(tgts):
        q = perturb(db_names[t], rng) if i in pert else db_names[t]
        if q: pos_q.append(q); kept.append(t)
    neg_q = []
    while len(neg_q) < N_NEG:
        s = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(8,18)))
        if s not in forbidden: neg_q.append(s)
    npos = len(pos_q)
    print(f"正例 {npos}，负例 {len(neg_q)}")

    db200 = batch_encode(db_names, K200); n200 = l2_normalize(db200)
    mu, sd = n200.mean(0), n200.std(0, ddof=0); sd = np.where(sd<1e-10,1,sd)
    cent, assign = cosine_kmeans((n200-mu)/sd, 50, seed=42)
    q200 = batch_encode(pos_q+neg_q, K200)
    sel = select_clusters((l2_normalize(q200)-mu)/sd, cent)

    # 划分 dev/test（按索引，正负各自一半）
    dev_pos = np.arange(0, npos//2); test_pos = np.arange(npos//2, npos)
    dev_neg = np.arange(npos, npos+len(neg_q)//2); test_neg = np.arange(npos+len(neg_q)//2, npos+len(neg_q))

    out = {}
    for d in (25, 50):
        dbd = l2_normalize(db200[:, :d]); qd = l2_normalize(q200[:, :d])
        def stats(idx_pos, idx_neg, th):
            tr = ca = 0
            for i in idx_pos:
                mem = np.flatnonzero(assign == sel[i]); s = dbd[mem] @ qd[i]
                w = np.flatnonzero(mem == kept[i])
                tr += int(len(w) and s[w[0]] >= th); ca += int(s.max() >= th)
            fp = sum(1 for i in idx_neg if (dbd[np.flatnonzero(assign==sel[i])] @ qd[i]).max() >= th)
            return tr/len(idx_pos), ca/len(idx_pos), fp/len(idx_neg), fp
        # dev 选 τ
        best = None
        for th in np.arange(0.80, 1.001, 0.005):
            tr, ca, fpr, _ = stats(dev_pos, dev_neg, th)
            if fpr <= 0.005 and (best is None or tr > best[1]): best = (th, tr, fpr)
        tau = best[0]
        tr_t, ca_t, fpr_t, fp_t = stats(test_pos, test_neg, tau)
        tr_90, ca_90, fpr_90, fp_90 = stats(test_pos, test_neg, 0.90)
        out[d] = {"tau_dev": round(float(tau),3), "dev_recall": round(best[1],4), "dev_fpr": round(best[2],4),
                  "test_recall": round(tr_t,4), "test_catch": round(ca_t,4),
                  "test_fpr": round(fpr_t,4), "test_fp": fp_t, "n_test_neg": len(test_neg),
                  "recall_ci": wilson(int(round(tr_t*len(test_pos))), len(test_pos)),
                  "fpr_ci": wilson(fp_t, len(test_neg)),
                  "at_0.90": {"recall": round(tr_90,4), "fpr": round(fpr_90,4), "fp": fp_90}}
        o = out[d]
        print(f"\nd={d}: dev 选 τ={o['tau_dev']}（dev 召回 {o['dev_recall']:.3f}, FPR {o['dev_fpr']:.3f}）")
        print(f"  test: 召回 {o['test_recall']:.3f} CI[{o['recall_ci'][0]:.3f},{o['recall_ci'][1]:.3f}]  "
              f"FPR {o['test_fpr']:.3f} ({o['test_fp']}/{o['n_test_neg']}) "
              f"CI[{o['fpr_ci'][0]:.4f},{o['fpr_ci'][1]:.4f}]")
        print(f"  对照 τ=0.90: 召回 {o['at_0.90']['recall']:.3f}  FPR {o['at_0.90']['fpr']:.3f} "
              f"({o['at_0.90']['fp']}/{o['n_test_neg']})")
    p = ROOT/"artifacts"/"threshold_validation"/"report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\noutputs -> {p}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
