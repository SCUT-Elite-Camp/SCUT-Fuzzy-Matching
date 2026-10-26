#!/usr/bin/env python3
"""端到端可变 d 对照：全计算 / 截断 c=400 × d=50,25,10。

查询 = 200 正例（30% 扰动）+ 200 负例（库外随机姓名），同一批 m=400。
round-1 用 200 维（簇选择不变），round-2 用 d 维。

指标口径（全部显式）：
- Recall@1        = target 在选中簇的正例 / 200
- e2e Recall      = catch 且 target 在选中簇 / 200
- 漏报            = 1 - e2e Recall（分母 200 正例）
- 查询级误报      = catch 的负例 / 200（分母 200 负例）
- 候选级误报      = 选中簇内非 target 候选中 sim>τ 的比例
"""
from __future__ import annotations

import argparse, csv, json, random, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
K, TAU = 50, 0.9

def build_queries():
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

    def perturb(name, rng):
        ps = [p for p, c in enumerate(name) if "a" <= c <= "z"]; rng.shuffle(ps)
        for pos in ps:
            reps = [c for c in "abcdefghijklmnopqrstuvwxyz" if c != name[pos]]; rng.shuffle(reps)
            for c in reps:
                cand = name[:pos] + c + name[pos+1:]
                if cand not in forbidden: return cand
        return None

    rng = random.Random(2026)
    tgts = rng.sample(range(len(db_names)), 200)
    pert = set(rng.sample(range(len(tgts)), 60))          # 30%
    pos_q, kept = [], []
    for i, t in enumerate(tgts):
        q = perturb(db_names[t], rng) if i in pert else db_names[t]
        if q is not None: pos_q.append(q); kept.append(t)
    # 负例：库外随机串
    neg_q = []
    while len(neg_q) < 200:
        n = rng.randint(8, 18)
        s = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n))
        if s not in forbidden: neg_q.append(s)

    db200 = batch_encode(db_names, K200)
    n200 = l2_normalize(db200)
    mu, sd = n200.mean(0), n200.std(0, ddof=0); sd = np.where(sd < 1e-10, 1.0, sd)
    cent, assign = cosine_kmeans((n200 - mu) / sd, K, seed=42)
    q200 = batch_encode(pos_q + neg_q, K200)
    q_std = (l2_normalize(q200) - mu) / sd
    sel = select_clusters(q_std, cent)
    return db200, db_names, cent, assign, q200, q_std, sel, np.array(kept), len(pos_q)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT/"artifacts"/"end_to_end_d"/"report.json")
    args = ap.parse_args()
    try:
        import tenseal as ts
    except ImportError:
        print("tenseal 未安装"); return 2
    from ckks.context import create_ckks_context
    from ckks.keys import serialize_public_context
    from ckks.tiling import (decrypt_tiled_score_tile, encrypt_tiled_feature_batch,
                             iter_tile_slices, make_slot_tile_layout)
    from fusion.kernel_fd import tiled_batch_matching_d
    from party_a.online_querier import choose_clusters_and_build_tiled_request
    from party_b.online_responder import compare_tiled_batch_to_centroids, tiled_batch_matching
    from protocol.transport import (serialize_tiled_first_round_request,
                                    serialize_tiled_second_round_request)
    from protocol.types import TiledFirstRoundRequest, TiledPartyALocalState
    from scripts.evaluate_binarization_frontier import l2_normalize

    db200, db_names, cent, assign, q200, q_std, sel, kept, n_pos = build_queries()
    m = len(q_std)
    ctx = create_ckks_context(); ctx.generate_relin_keys(); pub = serialize_public_context(ctx)
    layout = make_slot_tile_layout(m)
    sizes = np.bincount(assign, minlength=K); L = int(sizes.max())
    in_cluster = np.array([assign[t] == sel[i] for i, t in enumerate(kept)])
    print(f"m={m}（正例 {n_pos} + 负例 {m-n_pos}），簇大小 max={L}")

    def run(d: int, c: int, kernel):
        qd = l2_normalize(q200[:, :d])
        dbd = l2_normalize(db200[:, :d])
        mat = np.zeros((K, L, d))
        for i in range(K):
            mem = np.flatnonzero(assign == i)
            mat[i, :len(mem)] = dbd[mem]
        enc200 = encrypt_tiled_feature_batch(q_std, ctx, layout)
        encd = encrypt_tiled_feature_batch(qd, ctx, layout)
        st = TiledPartyALocalState(secret_context=ctx, encrypted_query_50=encd, layout=layout)
        w1 = serialize_tiled_first_round_request(
            TiledFirstRoundRequest(public_context_bytes=pub, encrypted_query_200=enc200, layout=layout))
        sc = compare_tiled_batch_to_centroids(w1, cent, serialize_output=True)
        req2, _ = choose_clusters_and_build_tiled_request(sc, st, k=K)
        # 绕过 transport 的 expected_count=50 断言（生产基线不可改）
        from protocol.types import TiledSecondRoundRequest
        _ser = lambda cs: [c.serialize() if hasattr(c, "serialize") else c for c in cs]
        w2 = TiledSecondRoundRequest(
            encrypted_query_50=_ser(req2.encrypted_query_50),
            encrypted_selectors=_ser(req2.encrypted_selectors),
            layout=layout,
        )
        sub = mat[:, :c, :]
        t0 = time.perf_counter()
        tiles = list(kernel(sub, w2, pub, tau=TAU, serialize_output=True))
        dt = time.perf_counter() - t0
        dec = np.concatenate([decrypt_tiled_score_tile(b, ctx, layout, w)
                              for b, (_, _, w) in zip(tiles, iter_tile_slices(c, layout))], axis=1)
        catch = (dec > 1e-4).any(axis=1)
        # 候选级误报：正例查询在选中簇内的非 target 候选
        cand_fp = cand_tot = 0
        for i in range(n_pos):
            mem = np.flatnonzero(assign == sel[i])
            s = dbd[mem] @ qd[i]
            cand_fp += int((s >= TAU).sum()); cand_tot += len(mem)
        return {"d": d, "c": c, "tiles": len(tiles), "seconds": dt,
                "selector_mults": len(tiles)*K*d, "query_ctct": len(tiles)*d,
                "recall1": float(in_cluster.mean()),
                "e2e_recall": float(np.mean(catch[:n_pos] & in_cluster)),
                "misses": int(np.sum(~(catch[:n_pos] & in_cluster))),
                "query_fpr": float(np.mean(catch[n_pos:])),
                "false_alarms": int(catch[n_pos:].sum()),
                "cand_fpr": cand_fp/max(cand_tot,1), "cand_denom": cand_tot,
                "dec": dec, "catch": catch}

    rows = []
    ref = run(50, L, tiled_batch_matching)          # 基线（原 kernel）
    rows.append({k: v for k, v in ref.items() if k not in ("dec","catch")})
    same = run(50, 400, tiled_batch_matching_d)     # 一致性：新 kernel, d=50
    diff = float(np.max(np.abs(same["dec"] - ref["dec"][:, :400])))
    sign_mis = int(np.sum(np.sign(same["dec"]) != np.sign(ref["dec"][:, :400])))
    print(f"d=50 新 kernel 一致性: max|diff|={diff:.3e}  符号不一致={sign_mis} / {same['dec'].size}")
    rows.append({k: v for k, v in same.items() if k not in ("dec","catch")})
    for d in (25, 10):
        r = run(d, 400, tiled_batch_matching_d)
        rows.append({k: v for k, v in r.items() if k not in ("dec","catch")})

    print(f"\n{'d':>4}{'c':>6}{'tiles':>7}{'selector乘':>12}{'端到端':>9}{'Recall@1':>10}{'e2e召回':>9}{'漏报':>6}{'误报(查询级)':>14}")
    for r in rows:
        print(f"{r['d']:>4}{r['c']:>6}{r['tiles']:>7}{r['selector_mults']:>12,}"
              f"{r['seconds']:>8.2f}s{r['recall1']:>10.3f}{r['e2e_recall']:>9.3f}"
              f"{r['misses']:>6}{r['query_fpr']:>14.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n_pos": n_pos, "n_neg": m-n_pos,
                                    "consistency": {"max_abs_diff": diff, "sign_mismatch": sign_mis},
                                    "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\noutputs -> {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
