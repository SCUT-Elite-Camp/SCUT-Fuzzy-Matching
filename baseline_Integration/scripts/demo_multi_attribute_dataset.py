"""跑真实数据集上的多属性模糊匹配：schema JSON + CSV → 双轮协议。

与 ``demo_multi_attribute.py`` 的区别是数据来源 —— 那个脚本用 5 条硬编码记录，
这个读真实 CSV（FEBRL 或合成数据），并按 schema JSON 决定属性集合与向量布局。

用法（在 baseline_Integration/ 下执行）::

    python scripts/fetch_dataset.py --dataset febrl
    python scripts/demo_multi_attribute_dataset.py --config config/examples/febrl_multi_attribute.json --limit 50

    python scripts/generate_synthetic_attributes.py --records 5000 --seed 42
    python scripts/demo_multi_attribute_dataset.py --config config/examples/multi_attribute_schema.json --limit 50

**关于能报什么指标**：协议本身只回传三样东西 —— 是否catch、选中的簇号、检查到第几列。
终端报告之外，结果同时落盘到 ``artifacts/demo/multi_attribute_dataset/``
（一条记录一份 JSON + CSV），与 ``demo_ncvr_matches.py`` / ``demo_sage_cross_script.py``
的约定一致。CSV 里每条查询一行，带真匹配分、冒充者分、逐属性相似度与加密判定，
是标定阈值时真正要拿去画图的那张表。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import CKKS_SLOT_LIMIT  # noqa: E402

from multi_attribute.dataset import (  # noqa: E402
    apply_labels,
    load_attribute_queries,
    load_attribute_records,
    load_febrl_pair,
)
from multi_attribute.encoder import (  # noqa: E402
    attribute_similarities,
    encode_record_vectors,
)
from multi_attribute.protocol import (  # noqa: E402
    prepare_party_b_multi_offline,
    run_multi_attribute_protocol,
)
from multi_attribute.schema import AttributeSchema  # noqa: E402


# ---------------------------------------------------------------------------
# job 文件
# ---------------------------------------------------------------------------


def _resolve(path: str | Path) -> Path:
    """相对路径按仓库根目录（baseline_Integration/）解析。"""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


def load_job(path: str | Path) -> dict:
    """读 job 文件。

    两种形态都认：

    * ``{"attributes": [...]}`` —— 裸 schema，列名默认与属性名同名；
    * ``{"schema": {...}, "attribute_columns": {...}, ...}`` —— 带数据映射的完整 job。

    下划线开头的键（``_comment`` 之类）一律当注释忽略。
    """

    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: job file must be a JSON object")

    if "attributes" in raw:
        job = {"schema": raw, "attribute_columns": None}
    else:
        job = {k: v for k, v in raw.items() if not k.startswith("_")}
        if "schema" not in job:
            raise SystemExit(f"{path}: job file needs either 'attributes' or 'schema'")

    schema = AttributeSchema.from_dict(job["schema"])
    columns = job.get("attribute_columns") or {spec.name: spec.name for spec in schema.attributes}

    missing = [spec.name for spec in schema.attributes if spec.name not in columns]
    if missing:
        raise SystemExit(f"{path}: attribute_columns is missing entries for {missing}")

    job["schema_object"] = schema
    job["attribute_columns"] = columns
    return job


def load_data(job: dict, limit: int | None, database_limit: int | None):
    schema: AttributeSchema = job["schema_object"]
    columns = job["attribute_columns"]
    database_path = job.get("database")
    queries_path = job.get("queries")
    if not database_path or not queries_path:
        raise SystemExit("job file needs 'database' and 'queries' paths")

    database_path, queries_path = _resolve(database_path), _resolve(queries_path)

    if job.get("link_mode") == "febrl":
        database, queries = load_febrl_pair(
            database_path,
            queries_path,
            columns,
            id_column=job.get("id_column", "rec_id"),
            limit=limit,
            database_limit=database_limit,
        )
    else:
        database = load_attribute_records(
            database_path, columns, id_column=job.get("id_column"), limit=database_limit
        )
        queries = load_attribute_queries(
            queries_path,
            columns,
            id_column=job.get("query_id_column"),
            label_column=job.get("label_column"),
            expected_ids_column=job.get("expected_ids_column"),
            limit=limit,
        )
        if job.get("labels"):
            queries = apply_labels(queries, _resolve(job["labels"]))

    if not database:
        raise SystemExit(f"no usable records loaded from {database_path}")
    if not queries:
        raise SystemExit(f"no usable queries loaded from {queries_path}")
    return database, queries


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def _audit_dates(database, queries, schema: AttributeSchema) -> list[str]:
    """检查 date 属性上的脏值 —— ``on_invalid="missing"`` 会静默清零，要让它可见。"""

    from multi_attribute.hashing import normalize_dob

    notes: list[str] = []
    for spec in schema.attributes:
        if spec.kind != "date" or spec.params.get("on_invalid", "raise") != "missing":
            continue
        day_first = bool(spec.params.get("day_first", False))
        bad, empty, total = 0, 0, 0
        for record in [*database, *(q.record for q in queries)]:
            value = (record.values or {}).get(spec.name)
            total += 1
            if value is None or not str(value).strip():
                empty += 1
                continue
            try:
                normalize_dob(value, day_first=day_first)
            except ValueError:
                bad += 1
        if bad or empty:
            notes.append(
                f"{spec.name}: {bad}/{total} unparseable -> zeroed, {empty} blank"
            )
    return notes


def print_schema(schema: AttributeSchema, database_size: int, limit: int | None) -> None:
    print("=" * 88)
    print(f"schema fingerprint {schema.fingerprint()}   threshold {schema.similarity_threshold}")
    print("=" * 88)
    print(f"{'attribute':<14}{'kind':<13}{'weight':>8}{'cluster':>10}{'match':>8}   params")
    cluster_layout, match_layout = schema.layout(), schema.match_layout()
    for spec in schema.attributes:
        c = cluster_layout[spec.name]
        m = match_layout[spec.name]
        params = " ".join(f"{k}={v}" for k, v in spec.params.items())
        print(
            f"{spec.name:<14}{spec.kind:<13}{spec.weight:>8.2f}"
            f"{c.stop - c.start:>10}{m.stop - m.start:>8}   {params}"
        )
    print(
        f"{'TOTAL':<14}{'':<13}{sum(schema.weights):>8.2f}"
        f"{schema.cluster_dim:>10}{schema.match_dim:>8}"
    )
    print(
        f"\nCKKS slots: {CKKS_SLOT_LIMIT}; headroom cluster {CKKS_SLOT_LIMIT - schema.cluster_dim} / "
        f"match {CKKS_SLOT_LIMIT - schema.match_dim}"
    )

    # 第二轮的 cluster_matrix 形状是 (k, cluster_size, match_dim)，是全流程最吃内存的地方。
    k = int(round(np.sqrt(database_size)))
    size = max(1, -(-database_size // k))
    est_mb = k * size * schema.match_dim * 8 / 1024**2
    print(
        f"database {database_size}"
        + (f" (truncated by --db-limit {limit})" if limit else "")
        + f" -> k~{k}, <= {size} rows/cluster -> cluster_matrix ~{est_mb:.1f} MB"
    )
    print()


def _best_plaintext_match(database, db_matrix, query, schema):
    """库向量矩阵与查询向量做一次点积，取 argmax 与「冒充者」最高分。

    比逐条 ``plaintext_similarity`` 快几个数量级 —— 库只编码一次，之后每条查询
    就是一次 ``(N, d) @ (d,)``。

    **冒充者分（impostor）**：库中**不属于**该查询真值的记录里的最高分。FEBRL 这类
    没有负例的数据集拿不到假正例统计（所有真值都是 True），冒充者分就是校准阈值
    所需的另一半：真匹配分要压过它。负例查询的真值集合为空，于是所有库记录都是
    冒充者，返回值即该查询的最高分。
    """

    query_vector = encode_record_vectors([query.record], schema)[1][0]
    scores = db_matrix @ query_vector
    order = np.argsort(scores)[::-1]
    best = int(order[0])

    expected = query.expected_record_ids
    impostor = float(scores[best])
    for idx in order:
        if not expected or database[int(idx)].record_id not in expected:
            impostor = float(scores[int(idx)])
            break

    return (
        database[best].record_id,
        float(scores[best]),
        attribute_similarities(query.record, database[best], schema),
        impostor,
    )


# ---------------------------------------------------------------------------
# 产物
# ---------------------------------------------------------------------------

# 与其他 demo 一致：产物落在 artifacts/demo/<去掉 demo_ 前缀的脚本名>/ 下，
# 文件名保留完整脚本名（对照 artifacts/demo/ncvr_matches/demo_ncvr_matches.json）。
# artifacts/ 已被 .gitignore 忽略。
_ARTIFACT_NAME = "demo_multi_attribute_dataset"
_ARTIFACT_DIR_NAME = _ARTIFACT_NAME.removeprefix("demo_")


def _csv_rows(rows: list[dict]) -> list[dict]:
    """把嵌套字段摊平成 CSV 列。

    JSON 里 ``similarities`` / ``encrypted`` 是嵌套的（读起来清楚），但 CSV 需要
    扁平的列才能直接拿去画标定图，所以在这里展开而不是把嵌套结构 repr 进单元格。
    ``sim_<属性名>`` 的列名跟着 schema 走，换 schema 就换列。
    """

    flat: list[dict] = []
    for row in rows:
        out = {k: v for k, v in row.items() if k not in {"similarities", "encrypted"}}
        out["expected_record_ids"] = "|".join(row.get("expected_record_ids") or [])

        encrypted = row.get("encrypted")
        if encrypted is None:
            out["encrypted"] = ""
            out["enc_agrees_with_label"] = ""
            out["enc_selected_cluster"] = ""
            out["enc_checked_columns"] = ""
            out["enc_true_match_cluster"] = ""
            out["enc_cluster_hit"] = ""
        else:
            out["encrypted"] = "CATCH" if encrypted["catch"] else "no-catch"
            out["enc_agrees_with_label"] = encrypted["agrees_with_label"]
            out["enc_selected_cluster"] = encrypted["selected_cluster"]
            out["enc_checked_columns"] = encrypted["checked_columns"]
            out["enc_true_match_cluster"] = encrypted.get("true_match_cluster", "")
            out["enc_cluster_hit"] = encrypted.get("cluster_hit", "")

        for name, value in (row.get("similarities") or {}).items():
            out[f"sim_{name}"] = value
        flat.append(out)
    return flat


def _save_outputs(result: dict, output_dir: str | Path) -> tuple[Path | None, Path | None]:
    """写 JSON + CSV；失败只告警。

    报告已经打到终端了，写盘失败不该让整个 run 白跑（Windows 上目录被占住是常事），
    所以和 ``demo_sage_cross_script.py`` 一样吞掉 IOError 而不是抛出去。
    """

    root = Path(output_dir)
    json_path = root / f"{_ARTIFACT_NAME}.json"
    csv_path = root / f"{_ARTIFACT_NAME}.csv"
    rows = result["rows"]
    try:
        root.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if rows:
            flat = _csv_rows(rows)
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
                writer.writeheader()
                writer.writerows(flat)
    except OSError as exc:
        print(f"Warning: could not save demo artifacts to {root}: {exc}")
        return None, None
    return json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a job / schema JSON file")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="max queries to run (default 20; each query is a full two-round HE run)",
    )
    parser.add_argument(
        "--db-limit",
        type=int,
        default=500,
        help="max database rows to load (default 500; 0 = unlimited)",
    )
    parser.add_argument(
        "--no-encrypted",
        action="store_true",
        help="only run the plaintext breakdown, skip the CKKS two-round protocol",
    )
    parser.add_argument("--k-mode", default="sqrt", help="cluster-count policy (default sqrt)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "demo" / _ARTIFACT_DIR_NAME),
        help="where to write the JSON + CSV report (default artifacts/demo/%s)"
        % _ARTIFACT_DIR_NAME,
    )
    args = parser.parse_args()

    job = load_job(args.config)
    schema: AttributeSchema = job["schema_object"]
    limit = None if args.limit <= 0 else args.limit
    database_limit = None if args.db_limit <= 0 else args.db_limit
    database, queries = load_data(job, limit, database_limit)

    print_schema(schema, len(database), database_limit)
    quality_notes = _audit_dates(database, queries, schema)
    for note in quality_notes:
        print(f"data quality: {note}")
    if any(spec.kind == "date" for spec in schema.attributes):
        print()

    # 库只编码一次，后续每条查询就是一次矩阵-向量乘。
    db_matrix = encode_record_vectors(database, schema)[1]

    # B 方离线阶段（k-means + 加密整个 cluster 矩阵）只做一次，N 条查询共用。
    artifacts = None
    cluster_of: dict[str, int] = {}
    if not args.no_encrypted:
        artifacts = prepare_party_b_multi_offline(
            database, cfg=schema, k_mode=args.k_mode, random_state=args.random_state
        )
        cluster_of = {
            record.record_id: int(cluster)
            for record, cluster in zip(database, artifacts.cluster_assignments)
        }

    agreements = 0
    cluster_hits = 0
    positive_catches = 0
    selected_ok = 0
    top1_hits = 0
    positives = sum(1 for q in queries if q.label)
    # 校准阈值只需要两个数：真匹配分最低能到多少、冒充者分最高能到多少。
    true_scores: list[float] = []
    impostor_scores: list[float] = []
    rows: list[dict] = []

    print(f"{'query':<16}{'label':<7}{'true':>7}{'imp':>7}  {'top-1':<14}{'attr breakdown':<40}{'encrypted'}")
    print("-" * 88)
    for query in queries:
        best_id, best_score, best_attr, impostor = _best_plaintext_match(
            database, db_matrix, query, schema
        )
        # 一律转成 Python float —— numpy 标量 json.dumps 序列化不了。
        best_score, impostor = float(best_score), float(impostor)
        impostor_scores.append(impostor)
        if query.label:
            # 只在正例上算 top-1 —— 负例的正确记录根本不在库里，必然"未命中"，
            # 混在一起算会把命中率稀释成没有意义的数字。
            true_scores.append(best_score)
            top1_hits += int(best_id is not None and best_id in query.expected_record_ids)

        breakdown = " ".join(f"{k}={v:.2f}" for k, v in best_attr.items())

        row: dict = {
            "query_id": query.record.record_id,
            "label": bool(query.label),
            "expected_record_ids": sorted(query.expected_record_ids),
            "top1_record_id": best_id,
            "top1_score": round(best_score, 6),
            "impostor_score": round(impostor, 6),
            "similarities": {k: round(float(v), 6) for k, v in best_attr.items()},
            "encrypted": None,
        }

        if args.no_encrypted:
            tail = ""
        else:
            run = run_multi_attribute_protocol(
                database,
                query.record,
                cfg=schema,
                k_mode=args.k_mode,
                random_state=args.random_state,
                artifacts=artifacts,
            )
            verdict = "CATCH" if run.catch else "no-catch"
            agreements += int(run.catch == query.label)
            row["encrypted"] = {
                "catch": bool(run.catch),
                "agrees_with_label": bool(run.catch == query.label),
                "selected_cluster": int(run.selected_cluster),
                "checked_columns": int(run.checked_columns),
            }
            # 簇召回：真匹配是否落在第一轮选中的簇里。第二轮只看一个簇，
            # 落在别处的真匹配**根本没有被检查的机会**，这是加密路径的主要误差来源。
            if query.label:
                selected_ok += 1
                positive_catches += int(run.catch)
                truth = next(iter(query.expected_record_ids), None)
                row["encrypted"]["true_match_cluster"] = cluster_of.get(truth) if truth else None
                row["encrypted"]["cluster_hit"] = bool(
                    truth is not None and cluster_of.get(truth) == run.selected_cluster
                )
                if row["encrypted"]["cluster_hit"]:
                    cluster_hits += 1
            tail = (
                f"  {verdict:<9} cluster={run.selected_cluster:<4} "
                f"checked={run.checked_columns}"
            )

        rows.append(row)
        print(
            f"{str(query.record.record_id):<16}{str(query.label):<7}{best_score:>7.3f}"
            f"{impostor:>7.3f}  {(best_id or '-'):<14}{breakdown:<40}{tail}"
        )

    print("-" * 88)
    total = len(queries)
    negatives = total - positives
    print(f"queries: {total} ({positives} positive / {negatives} negative)")

    summary: dict = {"queries": total, "positives": positives, "negatives": negatives}

    if positives:
        recall = top1_hits / positives
        summary["plaintext_top1_recall_positives_only"] = recall
        print(
            f"plaintext top-1 recall, positives only (debug reference): "
            f"{top1_hits}/{positives} = {recall:.1%}"
        )

    # 负例被误判为命中的比例 —— 只要最高分越过阈值就会发生。
    false_positives = sum(
        1
        for q, s in zip(queries, impostor_scores)
        if not q.label and s >= schema.similarity_threshold
    )
    summary["plaintext_false_positives_at_tau"] = false_positives
    if negatives:
        summary["plaintext_false_positive_rate"] = false_positives / negatives
        print(
            f"plaintext false positives at tau={schema.similarity_threshold:g} "
            f"(debug reference): {false_positives}/{negatives} = "
            f"{false_positives / negatives:.1%}"
        )

    # 阈值该定在哪：真匹配分要高于冒充者分。两个数都从明文算出，与协议输出无关。
    if true_scores or impostor_scores:
        lo = min(true_scores) if true_scores else float("nan")
        hi = max(impostor_scores) if impostor_scores else float("nan")
        # nan 不是合法 JSON，落盘时写成 null。
        summary["lowest_true_match_score"] = lo if true_scores else None
        summary["highest_impostor_score"] = hi if impostor_scores else None
        print(
            f"plaintext calibration window: lowest true-match {lo:.3f} / "
            f"highest impostor {hi:.3f}"
        )
        if true_scores and impostor_scores:
            if lo > hi:
                inside = hi < schema.similarity_threshold < lo
                summary["calibration_window"] = {"low": hi, "high": lo}
                summary["tau_inside_calibration_window"] = inside
                print(
                    f"  -> tau must be inside ({hi:.3f}, {lo:.3f}); "
                    f"current tau={schema.similarity_threshold:g} "
                    + ("is inside" if inside else "is OUTSIDE")
                )
            else:
                summary["tau_inside_calibration_window"] = False
                summary["true_matches_overlap_impostors"] = True
                print(
                    "  -> true matches and impostors OVERLAP here; no tau separates "
                    "them. Add attributes, raise weights, or accept errors."
                )

    if not args.no_encrypted:
        agreement = agreements / total
        summary["encrypted_agreement"] = agreement
        print(
            f"encrypted catch-vs-label agreement: {agreements}/{total} = "
            f"{agreement:.1%}"
        )
        if selected_ok:
            # 召回率的定义域只在正例上 —— 负例没有真匹配，"召回"对它没有意义。
            # 分母刻意与 cluster recall 取同一个（跑过加密的正例数），两行才能直接比较。
            recall = positive_catches / selected_ok
            summary["encrypted_recall_positives_only"] = recall
            print(
                f"encrypted recall, positives only: "
                f"{positive_catches}/{selected_ok} = {recall:.1%}"
            )
            cluster_recall = cluster_hits / selected_ok
            summary["cluster_recall"] = cluster_recall
            print(
                f"cluster recall (true match landed in the selected cluster): "
                f"{cluster_hits}/{selected_ok} = {cluster_recall:.1%}"
            )
            print(
                "  -> round 2 only examines ONE cluster, so a true match sitting in\n"
                "     another cluster is never even checked. Cluster recall is a hard\n"
                "     ceiling on the encrypted catch rate; the rest of the loss is the\n"
                "     per-column threshold test. Lower k (--k-mode log2) usually\n"
                "     raises cluster recall at the cost of scanning more columns."
            )
        print(
            "\nNOTE: the protocol returns only the sign of the threshold test, the\n"
            "selected cluster index, and the column index. It never returns the id of\n"
            "the matched record -- and a CATCH means \"some record in the database\n"
            "scores above tau\", not \"the right record was found\". So the encrypted\n"
            "path yields only a *detection* rate: recall counts a CATCH as a hit\n"
            "without knowing whether it hit the right record, and precision needs\n"
            "negatives to be meaningful. Identification precision / recall are the\n"
            "ones that stay uncomputable. The plaintext figures above are a debug\n"
            "reference, not protocol output."
        )

    cluster_layout, match_layout = schema.layout(), schema.match_layout()
    result = {
        "config_path": str(args.config),
        "schema": {
            "fingerprint": schema.fingerprint(),
            "similarity_threshold": float(schema.similarity_threshold),
            "name_attribute": schema.name_attribute,
            "cluster_dim": schema.cluster_dim,
            "match_dim": schema.match_dim,
            "ckks_slot_limit": CKKS_SLOT_LIMIT,
            "attributes": [
                {
                    "name": spec.name,
                    "kind": spec.kind,
                    "weight": float(spec.weight),
                    "cluster_dim": cluster_layout[spec.name].stop - cluster_layout[spec.name].start,
                    "match_dim": match_layout[spec.name].stop - match_layout[spec.name].start,
                    "params": dict(spec.params),
                }
                for spec in schema.attributes
            ],
        },
        "data": {
            "database_path": str(_resolve(job["database"])),
            "queries_path": str(_resolve(job["queries"])),
            "database_size": len(database),
            "database_limit": database_limit,
            "query_limit": limit,
        },
        "data_quality_notes": quality_notes,
        "run": {
            "encrypted": not args.no_encrypted,
            "k_mode": args.k_mode,
            "random_state": args.random_state,
        },
        "summary": summary,
        "rows": rows,
    }

    json_path, csv_path = _save_outputs(result, args.output_dir)
    if json_path is not None and csv_path is not None:
        print(f"Saved JSON: {json_path}")
        print(f"Saved CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
