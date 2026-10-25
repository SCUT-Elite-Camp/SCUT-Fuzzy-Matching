"""Evaluate ORIGINAL Top-1 followed by an OPTIONAL plaintext SimHash filter.

No changes to K-Means, top-1 selector, or CKKS protocol; this script deliberately
measures algorithmic feasibility before implementing a secure retrieval protocol.
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from party_b.offline_prep import prepare_party_b_offline
from minhash.encoder import batch_encode
from preprocessing.normalizer import l2_normalize
from preprocessing.text_cleaner import clean_name
from projection.random_hyperplane import make_planes, hash_signatures, collision_mask


def read_ncvr(data_root: Path, query_limit: int, db_limit: int, fuzzy_ratio: float, fuzzy_seed: int):
    folder = data_root / 'ncvr_10k' if (data_root / 'ncvr_10k').exists() else data_root
    with (folder / 'ncvr_10k_database.csv').open(newline='', encoding='utf-8') as f:
        db = list(csv.DictReader(f))
    with (folder / 'ncvr_10k_queries.csv').open(newline='', encoding='utf-8') as f:
        queries = list(csv.DictReader(f))
    if db_limit > 0: db = db[:db_limit]
    if query_limit > 0: queries = queries[:query_limit]
    db_names = [clean_name(row['full_name']) for row in db]
    db_ids = [row['ncid'] for row in db]
    lookup = {key: i for i, key in enumerate(db_ids)}
    entries=[]
    for row in queries:
        label = str(row['label']).strip().lower() in {'1','true','yes'}
        entries.append({'id': row['query_ncid'], 'name': clean_name(row['query_name']), 'label': label,
                        'target_index': lookup.get(row['query_ncid']) if label else None})
    # Same single-substitution recipe as data_pipeline/perturbations.py, with baseline clean_name.
    eligible=[i for i,e in enumerate(entries) if e['label'] and any('a'<=c<='z' for c in e['name'])]
    rng=random.Random(fuzzy_seed)
    forbidden=set(db_names)
    for i in rng.sample(eligible, int(len(eligible)*fuzzy_ratio+0.5)):
        original=entries[i]['name']; positions=[p for p,c in enumerate(original) if 'a'<=c<='z']
        rng.shuffle(positions)
        for p in positions:
            replacements=[c for c in 'abcdefghijklmnopqrstuvwxyz' if c!=original[p]]
            rng.shuffle(replacements)
            candidate=next((original[:p]+c+original[p+1:] for c in replacements
                            if original[:p]+c+original[p+1:] not in forbidden),None)
            if candidate is not None:
                entries[i]['name']=candidate; entries[i]['perturbed']=True; break
    return db_names, db_ids, entries


def safe_ratio(a,b):
    return None if b==0 else a/b


def run(args):
    if not (0<=args.fuzzy_ratio<=1): raise ValueError('fuzzy-ratio must be in [0,1]')
    if not (0<=args.tau<=1): raise ValueError('tau must be in [0,1]')
    if args.query_limit==0 or args.db_limit==0: raise ValueError('limits must be -1 or positive')
    if not args.pairs: raise ValueError('at least one --pairs R S setting is required')
    data_root=Path(args.data_path)
    db_names, db_ids, queries=read_ncvr(data_root,args.query_limit,args.db_limit,args.fuzzy_ratio,args.fuzzy_seed)
    if not db_names or not queries: raise ValueError('empty dataset')
    t0=time.perf_counter()
    # ORIGINAL offline stage: exactly the baseline's prepare_party_b_offline.
    artifacts=prepare_party_b_offline(db_names,k_mode=args.k,random_state=42)
    offline_sec=time.perf_counter()-t0
    assignments=artifacts.cluster_assignments
    # The validation embedding is the SAME EL=50 normalized representation used by CKKS.
    db_vec=l2_normalize(batch_encode(db_names,200)[:,:50]); db_vec=np.asarray(db_vec)
    # Recover the baseline's ACTUAL Top-1 argmax: 200D normalized + standardize + dot centroid.
    query_names=[x['name'] for x in queries]
    query_raw=batch_encode(query_names,200)
    query50=l2_normalize(query_raw[:,:50]); query200=l2_normalize(query_raw)
    query200_scaled=(query200-artifacts.scaler_mean)/(artifacts.scaler_scale+1e-8)
    selected=np.argmax(query200_scaled @ artifacts.centroids.T,axis=1)
    members=[np.flatnonzero(assignments==i) for i in range(artifacts.centroids.shape[0])]
    # Baseline cosine decision on the selected cluster; no projection.
    base_rows=[]
    for qi,e in enumerate(queries):
        idx=members[int(selected[qi])]
        scores=db_vec[idx] @ query50[qi]
        covered = e['target_index'] is not None and assignments[e['target_index']]==selected[qi]
        baseline_match=bool(np.any(scores>=args.tau))
        base_rows.append({'query_index':qi,'query_id':e['id'],'label':e['label'],'perturbed':bool(e.get('perturbed',False)),
                          'target_in_database':e['target_index'] is not None,'target_in_top1':bool(covered),
                          'selected_cluster':int(selected[qi]),'cluster_size':int(len(idx)),
                          'baseline_plain_match':baseline_match,'query_name':e['name']})
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    def confusion(rows,field):
        tp=sum(bool(r[field]) and r['label'] for r in rows)
        fp=sum(bool(r[field]) and not r['label'] for r in rows)
        fn=sum(not bool(r[field]) and r['label'] for r in rows)
        tn=sum(not bool(r[field]) and not r['label'] for r in rows)
        return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':safe_ratio(tp,tp+fp),
                'recall':safe_ratio(tp,tp+fn)}
    pos_with_target=[r for r in base_rows if r['label'] and r['target_in_database']]
    baseline={'mode':'baseline_top1_no_projection','rounds':0,'bits':0,'mean_candidates':float(np.mean([r['cluster_size'] for r in base_rows])),
              'candidate_total':int(sum(r['cluster_size'] for r in base_rows)),
              'target_coverage':safe_ratio(sum(r['target_in_top1'] for r in pos_with_target),len(pos_with_target)),
              'query_metrics':confusion(base_rows,'baseline_plain_match')}
    summaries=[baseline]
    all_rows=[]
    for R,S in args.pairs:
        if R<=0 or S<=0: raise ValueError('R and S must be positive')
        t_proj=time.perf_counter()
        planes=make_planes(50,R,S,args.projection_seed)
        db_keys=hash_signatures(db_vec,planes)
        q_keys=hash_signatures(query50,planes)
        build_seconds=time.perf_counter()-t_proj
        rows=[]
        for qi,e in enumerate(queries):
            t_query=time.perf_counter()
            idx=members[int(selected[qi])]
            mask=collision_mask(q_keys[qi],db_keys[idx])
            kept=idx[mask]
            # IMPORTANT: validation only, plaintext exact same EL=50 cosine threshold;
            # not an encrypted candidate retrieval protocol.
            pred=bool(np.any((db_vec[kept] @ query50[qi])>=args.tau))
            target=e['target_index']
            keep_target=bool(target is not None and np.any(kept==target))
            rows.append({**base_rows[qi], 'rounds':R,'bits':S,'retained_candidates':int(len(kept)),
                         'projection_plain_match':pred,'target_after_projection':keep_target,
                         'projection_query_seconds':time.perf_counter()-t_query})
        target_rows=[r for r in rows if r['label'] and r['target_in_database']]
        covered_rows=[r for r in target_rows if r['target_in_top1']]
        summary={'mode':'top1_plus_projection','rounds':R,'bits':S,
                 'candidate_total':int(sum(r['retained_candidates'] for r in rows)),
                 'mean_candidates':float(np.mean([r['retained_candidates'] for r in rows])),
                 'relative_reduction_within_top1':1-sum(r['retained_candidates'] for r in rows)/max(1,sum(r['cluster_size'] for r in rows)),
                 'target_coverage':safe_ratio(sum(r['target_after_projection'] for r in target_rows),len(target_rows)),
                 'conditional_target_recall':safe_ratio(sum(r['target_after_projection'] for r in covered_rows),len(covered_rows)),
                 'query_metrics':confusion(rows,'projection_plain_match'),
                 'projection_index_build_seconds':build_seconds,
                 'projection_filter_total_seconds':sum(r['projection_query_seconds'] for r in rows)}
        summaries.append(summary)
        all_rows+=rows
        print(f'R={R:>3}, s={S:>3}: mean candidates={summary["mean_candidates"]:.1f}; '+
              f'conditional target recall={summary["conditional_target_recall"]}; '+
              f'plain recall={summary["query_metrics"]["recall"]}')
    with (outdir/'summary.json').open('w',encoding='utf8') as f:
        json.dump({'config':vars(args),'offline_seconds':offline_sec,'baseline':baseline,'results':summaries[1:],
                   'note':'PLAINTEXT FEASIBILITY ONLY. No encrypted projection or secure retrieval is implemented.'},
                  f,ensure_ascii=False,indent=2,default=str)
    def write_csv(filename,rows):
        if not rows:return
        with (outdir/filename).open('w',newline='',encoding='utf8') as f:
            fields=list(dict.fromkeys(key for row in rows for key in row))
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    summary_flat=[]
    for item in summaries:
        row={k:v for k,v in item.items() if k!='query_metrics'}
        row.update({'plain_'+k:v for k,v in item['query_metrics'].items()});summary_flat.append(row)
    write_csv('summary.csv',summary_flat);write_csv('per_query.csv',all_rows)
    print('Original Top-1 baseline:', baseline)
    print('Saved:',outdir.resolve())


def main():
    p=argparse.ArgumentParser(description='Top-1 unchanged, followed by optional random-hyperplane projection (plaintext feasibility study).')
    p.add_argument('--data-path',default=str(ROOT/'data'))
    p.add_argument('--output-dir',default=str(ROOT/'artifacts'/'top1_projection'))
    p.add_argument('--query-limit',type=int,default=200)
    p.add_argument('--db-limit',type=int,default=10000)
    p.add_argument('--k',type=int,default=50,help='Original K-Means number of clusters, not Top-K; Top-1 remains fixed')
    p.add_argument('--tau',type=float,default=.9)
    p.add_argument('--fuzzy-ratio',type=float,default=.3)
    p.add_argument('--fuzzy-seed',type=int,default=42)
    p.add_argument('--projection-seed',type=int,default=2026)
    p.add_argument('--pairs',type=int,nargs=2,action='append',metavar=('R','S'),help='Repeatable: R rounds, S bits per round')
    args=p.parse_args()
    if args.pairs is None: args.pairs=[(4,8),(8,8),(16,8),(8,16),(16,16)]
    run(args)
if __name__=='__main__': main()
