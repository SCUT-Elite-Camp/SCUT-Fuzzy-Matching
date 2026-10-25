#!/usr/bin/env python3
"""抽样实测多属性双轮协议的**通信量**（序列化字节数），不猜公式。

做法：直接调用 ``multi_attribute.protocol`` 里的四个阶段函数，把每一步真正要上
线的字节串长度加起来，并按真实早停逻辑在第一个越阈列处停下 —— 和 demo 走的是
同一条代码路径，只是不重新跑评测、只数字节。

口径
----
* **上行 1**：Party A → B，公开 CKKS context + 加密 cluster 向量
* **下行 1**：B → A，k 个质心相似度密文（每个一个标量）
* **上行 2**：A → B，加密 match 向量 + 每个探针一个 one-hot selector 密文
* **下行 2**：B → A，逐列返回的 ``mask*(score-tau)`` 密文，到第一个正例即停

公开 context 每个查询都被 ``prepare_party_a_multi_query`` 重新生成并随上行发出，
这里如实计入每查询通信（真实部署可一次握手复用，故另给"扣除 context"的净值）。

用法::

    python3 multi_attribute_comparison/scripts/measure_multi_attribute_comm.py \
        --config baseline_Integration/config/examples/febrl_multi_attribute.json \
        --db-limit 0 --sample 10 --probes 3 \
        --out baseline_Integration/artifacts/demo/febrl_ma_100/comm_10q.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = next(
    candidate
    for candidate in (_HERE.parent, _HERE.parent.parent)
    if (candidate / "baseline_Integration").is_dir()
)
sys.path.insert(0, str(ROOT / "baseline_Integration"))

import tenseal as ts  # noqa: E402

from multi_attribute.jobfile import load_data, load_job  # noqa: E402
from multi_attribute.protocol import (  # noqa: E402
    choose_multi_clusters,
    column_wise_multi_probe_matching,
    compare_multi_to_centroids,
    prepare_party_a_multi_query,
    prepare_party_b_multi_offline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--db-limit", type=int, default=0)
    parser.add_argument("--sample", type=int, default=10, help="抽多少条查询实测")
    parser.add_argument("--probes", default="3")
    parser.add_argument("--k-mode", default="sqrt")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    job = load_job(args.config)
    schema = job["schema_object"]
    database, queries = load_data(job, None, None if args.db_limit <= 0 else args.db_limit)

    print(f"database={len(database)} queries={len(queries)} cluster_dim={schema.cluster_dim} "
          f"match_dim={schema.match_dim} tau={schema.similarity_threshold}")

    artifacts = prepare_party_b_multi_offline(
        database, cfg=schema, k_mode=args.k_mode, random_state=args.random_state
    )
    k = int(artifacts.centroids.shape[0])
    print(f"k={k} max_size={artifacts.max_size}")

    per_query = []
    for query in queries[: args.sample]:
        req1, state = prepare_party_a_multi_query(query, artifacts, cfg=schema)
        up1 = len(req1.public_context_bytes) + len(req1.encrypted_cluster_query)

        scores = compare_multi_to_centroids(req1, artifacts.centroids)
        down1 = sum(len(item) for item in scores)

        req2, probed = choose_multi_clusters(scores, state, probes=args.probes)
        up2 = len(req2.encrypted_match_query) + sum(len(s) for s in req2.encrypted_selectors)

        down2 = 0
        columns = 0
        hit = False
        for _cluster_idx, _column_idx, cipher in column_wise_multi_probe_matching(
            artifacts.cluster_matrix,
            req2,
            req1.public_context_bytes,
            tau=schema.similarity_threshold,
            cluster_sizes=artifacts.cluster_sizes,
        ):
            down2 += len(cipher)
            columns += 1
            if not hit:
                value = ts.ckks_vector_from(state.secret_context, cipher).decrypt()[0]
                hit = bool(value > 0)
            if hit:
                break

        total = up1 + down1 + up2 + down2
        per_query.append(
            {
                "query_id": getattr(query, "query_id", None),
                "up1_bytes": up1,
                "down1_bytes": down1,
                "up2_bytes": up2,
                "down2_bytes": down2,
                "total_bytes": total,
                "total_no_context_bytes": total - len(req1.public_context_bytes),
                "context_bytes": len(req1.public_context_bytes),
                "columns_scanned": columns,
                "probes": len(probed),
                "catch": hit,
            }
        )
        print(f"  {per_query[-1]['query_id']}: total={total/1024:.0f} KB "
              f"cols={columns} catch={hit}")

    n = len(per_query)
    mean_bytes = sum(r["total_bytes"] for r in per_query) / n
    mean_no_ctx = sum(r["total_no_context_bytes"] for r in per_query) / n
    result = {
        "config": args.config,
        "database_size": len(database),
        "sample_queries": n,
        "k": k,
        "cluster_dim": schema.cluster_dim,
        "match_dim": schema.match_dim,
        "probes": args.probes,
        "one_time_context_bytes": per_query[0]["context_bytes"],
        "mean_total_bytes_per_query": mean_bytes,
        "mean_bytes_per_query_no_context": mean_no_ctx,
        "mean_columns_scanned": sum(r["columns_scanned"] for r in per_query) / n,
        "per_query": per_query,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nmean per query: {mean_bytes/1024:.1f} KB (含 context) / "
          f"{mean_no_ctx/1024:.1f} KB (不含 context)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
