#!/usr/bin/env python3
"""在 fused_multilingual_10k 上跑 Top-k（= 探测簇数）与固定阈值的网格实验。

为什么是"探测簇数"而不是"返回 k 条记录"
--------------------------------------
协议（``multi_attribute/protocol.py``）只回传三样东西：是否越阈、选中的簇号、检查到
第几列，**从不返回命中记录 id**。所以"取回 Top-k 候选记录"在加密链路上无处可接。
本仓库自己的口径也是按簇算的：``--probes`` 表示第二轮按质心分降序探测前几个簇，
数据 README 里"top-3 明文协议参考 TP=133"指的就是 probes=3。故 ``--top-k`` 直接
映射到 ``probes``：Top-1 = 只探最可能的 1 个簇，Top-3 = 探前 3 个簇。

判定口径（与数据 README 一致）
------------------------------
固定阈值 tau 不随 Top-k 变化；标签来自 ``labels.csv``（160 正 / 40 负）：

    TP = 标签为真 且 catch      FP = 标签为假 且 catch
    FN = 标签为真 且 no-catch   TN = 标签为假 且 no-catch

catch 指"探测的 k 个簇里存在某个候选得分 ≥ tau"，即检测口径；因为协议不返回命中 id，
无法区分"命中的是真匹配"还是"同簇冒充者"。报告里必须写明这一点。

另附 ``plaintext_score``（真匹配记录的明文得分）与 ``should_catch``（是否 ≥ tau），
用来把 FN 拆成"明文就不该 catch"和"本该 catch 但协议漏了"。

用法::

    python3 multi_attribute_comparison/experiments_topk_10k/run_topk.py \
        --config baseline_Integration/config/examples/fused_multilingual_10k_multi_attribute.json \
        --limit 20 --top-k 3 --fixed-threshold 0.6135671549269659 \
        --out multi_attribute_comparison/experiments_topk_10k/results/k3_tau0614.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
ROOT = next(
    candidate
    for candidate in (_HERE.parent, _HERE.parent.parent)
    if (candidate / "baseline_Integration").is_dir()
)
sys.path.insert(0, str(ROOT / "baseline_Integration"))

from multi_attribute.encoder import encode_record_vectors  # noqa: E402
from multi_attribute.jobfile import load_data, load_job  # noqa: E402
from multi_attribute.protocol import (  # noqa: E402
    prepare_party_b_multi_offline,
    run_multi_attribute_protocol,
)


def target_scores(database, queries, schema) -> list[float | None]:
    """每条查询对"真匹配记录"的明文相似度（无真值则为 None）。"""

    _, db_match = encode_record_vectors(database, schema)
    _, q_match = encode_record_vectors([q.record for q in queries], schema)
    ids = [getattr(r, "record_id", None) for r in database]
    index_of = {rid: i for i, rid in enumerate(ids)}
    scores: list[float | None] = []
    for i, query in enumerate(queries):
        best = None
        for rid in getattr(query, "expected_record_ids", ()) or ():
            j = index_of.get(rid)
            if j is None:
                continue
            value = float(db_match[j] @ q_match[i])
            best = value if best is None else max(best, value)
        scores.append(best)
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="job 文件（含 schema 与数据路径）")
    parser.add_argument("--limit", type=int, default=10, help="跑多少条查询（0 = 全部）")
    parser.add_argument("--db-limit", type=int, default=0, help="库记录上限（0 = 全部 10000）")
    parser.add_argument("--top-k", type=int, default=1, help="第二轮探测的簇数（= probes）")
    parser.add_argument("--fixed-threshold", type=float, default=None,
                        help="固定相似度阈值；省略则用 job 文件里的 similarity_threshold")
    parser.add_argument("--enable-projection", action="store_true",
                        help="开启投影（multi_attribute 路径暂无实现，开启会直接报错）")
    parser.add_argument("--k-mode", default="sqrt")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out", required=True, help="结果 JSON 路径")
    args = parser.parse_args()

    if args.enable_projection:
        print("ERROR: multi_attribute 路径没有投影实现（projection 只存在于单属性 V1 管线 "
              "protocol/orchestrator.py + party_b/offline_prep.py + clustering/projection.py）。",
              file=sys.stderr)
        return 2

    job = load_job(args.config)
    schema = job["schema_object"]
    tau = schema.similarity_threshold if args.fixed_threshold is None else args.fixed_threshold

    limit = None if args.limit <= 0 else args.limit
    db_limit = None if args.db_limit <= 0 else args.db_limit
    database, queries = load_data(job, limit, db_limit)

    print(f"database={len(database)} queries={len(queries)} k_mode={args.k_mode} "
          f"top_k={args.top_k} tau={tau:.6f}", flush=True)

    t0 = time.perf_counter()
    artifacts = prepare_party_b_multi_offline(
        database, cfg=schema, k_mode=args.k_mode, random_state=args.random_state
    )
    offline_s = time.perf_counter() - t0
    k_clusters = int(artifacts.centroids.shape[0])
    print(f"offline build: {offline_s:.2f}s  k={k_clusters} max_cluster={artifacts.max_size}",
          flush=True)

    plaintext = target_scores(database, queries, schema)

    rows = []
    t_start = time.perf_counter()
    for index, query in enumerate(queries, 1):
        q0 = time.perf_counter()
        run = run_multi_attribute_protocol(
            database,
            query.record,          # 注意：必须传底层 record，传 AttributeQuery 会编码成全零向量
            cfg=schema,
            artifacts=artifacts,
            probes=args.top_k,
            tau=tau,
            early_stop=True,
        )
        elapsed = time.perf_counter() - q0
        label = bool(query.label)
        caught = bool(run.catch)
        if label and caught:
            outcome = "TP"
        elif label and not caught:
            outcome = "FN"
        elif not label and caught:
            outcome = "FP"
        else:
            outcome = "TN"
        score = plaintext[index - 1]
        rows.append(
            {
                "query_id": getattr(query.record, "record_id", None),
                "label": label,
                "plaintext_score": None if score is None else round(score, 6),
                "should_catch": None if score is None else bool(score >= tau),
                "catch": caught,
                "outcome": outcome,
                "columns_checked": int(run.checked_columns),
                "elapsed_ms": round(elapsed * 1000, 1),
            }
        )
        print(f"  [{index}/{len(queries)}] {rows[-1]['query_id']} label={label} "
              f"pt={rows[-1]['plaintext_score']} {outcome} cols={run.checked_columns} "
              f"{elapsed:.1f}s", flush=True)
    online_s = time.perf_counter() - t_start

    tp = sum(1 for r in rows if r["outcome"] == "TP")
    fp = sum(1 for r in rows if r["outcome"] == "FP")
    fn = sum(1 for r in rows if r["outcome"] == "FN")
    tn = sum(1 for r in rows if r["outcome"] == "TN")
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None

    summary = {
        "config": args.config,
        "database_size": len(database),
        "queries": len(rows),
        "top_k": args.top_k,
        "fixed_threshold": tau,
        "k_mode": args.k_mode,
        "clusters": k_clusters,
        "cluster_dim": schema.cluster_dim,
        "match_dim": schema.match_dim,
        "offline_seconds": round(offline_s, 2),
        "online_seconds": round(online_s, 2),
        "mean_query_seconds": round(online_s / len(rows), 2) if rows else None,
        "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "positives_above_tau": sum(1 for r in rows if r["should_catch"]),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rows": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in
                      ("top_k", "queries", "confusion", "positives_above_tau", "precision",
                       "recall", "f1", "online_seconds", "mean_query_seconds")},
                     ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
