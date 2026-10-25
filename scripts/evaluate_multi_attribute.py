#!/usr/bin/env python3
"""从 demo 落盘的 JSON 计算多属性实验的 P / R / F1（检测口径）。

协议只回传"是否越阈"这一个比特，不返回命中记录的 id，所以这里能算的是
**查询级检测（detection）**的混淆矩阵，而不是 identification 的 P/R：

    TP = 标签为真 且 CATCH
    FP = 标签为假 且 CATCH        （冒充者被误 catch）
    FN = 标签为真 且 no-catch
    TN = 标签为假 且 no-catch

P = TP/(TP+FP)、R = TP/(TP+FN)、F1 = 2PR/(P+R)。标签取 ``rows[i].label``，
即数据集自带真值（FEBRL 由 rec_id 推、多语言由 labels.csv 给）。

同时输出 demo 原生的 ``recall_should_catch`` 与 ``cluster_recall`` 作对照 ——
前者分母含 tau（可被调阈值推高），后者与 tau 无关，是更硬的锚点。

用法::

    python3 scripts/evaluate_multi_attribute.py \
        --run baseline_Integration/artifacts/demo/febrl_ma_200/demo_multi_attribute_dataset.json \
        --queries 200 --online-seconds 762.4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compute(rows: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    for row in rows:
        label = bool(row.get("label"))
        caught = bool((row.get("encrypted") or {}).get("catch"))
        if label and caught:
            tp += 1
        elif label and not caught:
            fn += 1
        elif not label and caught:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return {
        "queries": len(rows),
        "positives": tp + fn,
        "negatives": fp + tn,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="demo_multi_attribute_dataset.json")
    parser.add_argument("--queries", type=int, default=None, help="查询数（默认取 rows 长度）")
    parser.add_argument("--online-seconds", type=float, default=None, help="两轮在线总耗时")
    parser.add_argument("--comm-mb", type=float, default=None, help="通信总量 MB（实测/估计）")
    parser.add_argument("--out", default=None, help="结果写到哪里（默认同目录 *_metrics.json）")
    args = parser.parse_args()

    run_path = Path(args.run)
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    metrics = compute(rows)

    metrics["dataset"] = str(payload.get("data", {}).get("database_path", ""))
    metrics["database_size"] = payload.get("data", {}).get("database_size")
    metrics["tau"] = payload.get("summary", {}).get("tau")
    metrics["probes"] = payload.get("run", {}).get("probes")
    metrics["schema_fingerprint"] = payload.get("schema", {}).get("fingerprint")
    metrics["cluster_dim"] = payload.get("schema", {}).get("cluster_dim")
    metrics["match_dim"] = payload.get("schema", {}).get("match_dim")
    metrics["k"] = payload.get("summary", {}).get("cluster_points") or payload.get("summary", {}).get("clusters")
    summary = payload.get("summary", {})
    metrics["recall_should_catch"] = summary.get("recall_should_catch")
    metrics["cluster_recall"] = summary.get("cluster_recall")
    metrics["cluster_recall_top1_only"] = summary.get("cluster_recall_top1_only")
    metrics["scan_columns_mean"] = summary.get("scan_columns_mean")
    metrics["full_scan_columns"] = summary.get("full_scan_columns")

    n = args.queries or len(rows)
    if args.online_seconds:
        metrics["online_seconds"] = args.online_seconds
        metrics["latency_ms_per_query"] = 1000.0 * args.online_seconds / n
    if args.comm_mb is not None:
        metrics["comm_mb"] = args.comm_mb
        if args.online_seconds:
            metrics["comm_kb_per_query"] = 1024.0 * args.comm_mb / n

    out = Path(args.out) if args.out else run_path.with_name(run_path.stem + "_metrics.json")
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
