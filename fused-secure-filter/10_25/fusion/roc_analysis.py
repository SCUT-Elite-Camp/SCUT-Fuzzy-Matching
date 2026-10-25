#!/usr/bin/env python3
"""明文 ROC：不同 d 在相同误报水平下的召回，以及阈值重标定的效果。

对每条查询取"簇内最大相似度"作为判定统计量（与 kernel 的 max 判定一致）。
- 正例：max ≥ τ_th ⇒ catch；同时单独记录 target 自身的 sim ≥ τ_th（真实召回）
- 负例：max ≥ τ_th ⇒ 误报
扫 τ_th，给出 ROC 与"零误报下的召回"。
"""
from __future__ import annotations
import csv, json, random, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main() -> int:
    from config.params import NUM_PERMUTATIONS_CLUSTER as K200
    from minhash.encoder import batch_encode
    from scripts.evaluate_binarization_frontier import clean_name, l2_normalize
    from scripts.evaluate_cluster_conditional_filter import cosine_kmeans, select_clusters

    folder = ROOT/"data"/"ncvr_10k"
    if not folder.exists(): folder = ROOT/"data"
    rows = list(csv.DictReader((folder/"ncvr_10k_database.csv").open(newline="", encoding="utf-8")))[:10000]
    db_names = [clean_name(r["full_name"]) for r in rows]; forbidden = set(db_names)
    rng = random.Random(2026)
    tgts = rng.sample(range(len(db_names)), 200)
    def perturb(n, r):
        ps=[p for p,c in enumerate(n) if 'a'<=c<='z']; r.shuffle(ps)
        for pos in ps:
            reps=[c for c in "abcdefghijklmnopqrstuvwxyz" if c!=n[pos]]; r.shuffle(reps)
            for c in reps:
                cand=n[:pos]+c+n[pos+1:]
                if cand not in forbidden: return cand
        return None
    pert=set(rng.sample(range(len(tgts)),60)); pos_q=[]; kept=[]
    for i,t in enumerate(tgts):
        q=perturb(db_names[t],rng) if i in pert else db_names[t]
        if q: pos_q.append(q); kept.append(t)
    neg_q=[]
    while len(neg_q)<200:
        s="".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(8,18)))
        if s not in forbidden: neg_q.append(s)

    db200=batch_encode(db_names,K200); n200=l2_normalize(db200)
    mu,sd=n200.mean(0),n200.std(0,ddof=0); sd=np.where(sd<1e-10,1,sd)
    cent,assign=cosine_kmeans((n200-mu)/sd,50,seed=42)
    q200=batch_encode(pos_q+neg_q,K200)
    sel=select_clusters((l2_normalize(q200)-mu)/sd,cent)

    out={}
    print(f"正例 {len(kept)} 条");print(f"{'d':>4}{'τ_th':>7}{'正例catch':>10}{'target召回':>11}{'负例误报':>10}")
    for d in (10,25,32,40,50):
        dbd=l2_normalize(db200[:,:d]); qd=l2_normalize(q200[:,:d])
        pos_max=[]; pos_tgt=[]; neg_max=[]
        npos=len(kept)
        for i in range(npos):
            mem=np.flatnonzero(assign==sel[i]); s=dbd[mem]@qd[i]
            pos_max.append(float(s.max()))
            w=np.flatnonzero(mem==kept[i])
            pos_tgt.append(float(s[w[0]]) if len(w) else -1.0)
        for i in range(npos, len(pos_q)+200):
            mem=np.flatnonzero(assign==sel[i]); s=dbd[mem]@qd[i]
            neg_max.append(float(s.max()))
        pos_max=np.array(pos_max); pos_tgt=np.array(pos_tgt); neg_max=np.array(neg_max)
        rows_d=[]
        for th in np.arange(0.80, 1.001, 0.01):
            ca=float((pos_max>=th).mean()); tr=float((pos_tgt>=th).mean()); fp=float((neg_max>=th).mean())
            rows_d.append({"tau": round(float(th),2), "pos_catch": ca, "target_recall": tr, "neg_fpr": fp})
        out[d]=rows_d
        # 零误报下最高召回
        zero=[r for r in rows_d if r["neg_fpr"]==0]
        best0=max(zero, key=lambda r: r["target_recall"]) if zero else None
        print(f"{d:>4}{best0['tau'] if best0 else 0:>7.2f}"
              f"{best0['pos_catch'] if best0 else 0:>10.3f}"
              f"{best0['target_recall'] if best0 else 0:>11.3f}{0:>10}")
        out[f"{d}_zero_fpr"]=best0
    p=ROOT/"artifacts"/"roc"/"report.json"; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\noutputs -> {p}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
