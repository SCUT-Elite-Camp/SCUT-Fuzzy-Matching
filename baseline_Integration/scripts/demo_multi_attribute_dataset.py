"""跑真实数据集上的多属性模糊匹配：schema JSON + CSV → 双轮协议。

与 ``demo_multi_attribute.py`` 的区别是数据来源 —— 那个脚本用 5 条硬编码记录，
这个读真实 CSV（FEBRL 或合成数据），并按 schema JSON 决定属性集合与向量布局。

用法（在 baseline_Integration/ 下执行）::

    python scripts/fetch_dataset.py --dataset febrl
    python scripts/demo_multi_attribute_dataset.py --config config/examples/febrl_multi_attribute.json --limit 50

    python scripts/generate_synthetic_attributes.py --records 5000 --seed 42
    python scripts/demo_multi_attribute_dataset.py --config config/examples/multi_attribute_schema.json --limit 50

**关于能报什么指标**：协议本身只回传三样东西 —— 是否catch、选中的簇号、检查到第几列。

**召回率的分母**取的是 ``should_catch``，即**真匹配那条记录的明文相似度 > tau**，
而不是"标签为真"：被扰动到低于阈值的副本按协议判据本来就不该 catch，把它们算进
分母等于让召回率背上一份协议不负责的债。这条口径下"完全一致的查询（明文分 1.0）
必能 catch"是个可断言的不变量。

代价是分母里含有 tau —— **调高 tau 会自动删掉最难的查询、把召回率推向 100%，
而一条错误都没修**。FEBRL 实测：tau 从 0.6 抬到 0.9，分母 99 → 50，召回显示 100.0%，
而集群漏掉的仍是 13 条（``cluster recall`` 87/100 不变）。所以报告里每一行都带绝对
计数，并把唯一真正与 tau 无关的 ``cluster recall`` 和标定窗口打印出来互相制约。
**注意漏掉条数本身是 tau 相关的**（它的分母就是 should-catch 集合），屏幕上会标注。
别把这里的召回率当优化目标；它是对阈值的一致性报告。

终端报告之外，结果同时落盘到 ``artifacts/demo/multi_attribute_dataset/``
（一条记录一份 JSON + CSV），与 ``demo_ncvr_matches.py`` / ``demo_sage_cross_script.py``
的约定一致。CSV 里每条查询一行，带真匹配分（``expected_score``）、``tau``、
``should_catch``、``expected_score_margin``（= 真匹配分 - tau，符号即是否应 catch）、
``enc_miss_cause``（漏掉的归因：round1_cluster / boundary / unexplained）、冒充者分、
逐属性相似度与加密判定，是标定阈值时真正要拿去画图的那张表。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Mapping, NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import (  # noqa: E402
    CKKS_SLOT_LIMIT,
    DEFAULT_MULTI_ATTRIBUTE_PROBES,
    MULTI_ATTRIBUTE_DECRYPT_EPS,
)

from multi_attribute.encoder import (  # noqa: E402
    attribute_similarities,
    encode_record_vectors,
)
from multi_attribute.jobfile import (  # noqa: E402
    load_data,
    load_job,
    resolve_path as _resolve,
)
from multi_attribute.protocol import (  # noqa: E402
    prepare_party_b_multi_offline,
    run_multi_attribute_protocol,
)
from multi_attribute.schema import AttributeSchema  # noqa: E402


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


class PlaintextMatch(NamedTuple):
    """一次 ``(N, d) @ (d,)`` 点积的全部产物。

    ``top1_*`` 是 argmax；``expected_score`` 是**真匹配那一条**的分。两者在实测里
    总是同一条记录，但**不能假设** —— argmax 落到冒充者上时，把 ``top1_score``
    当真匹配分用会把冒充者的分数记成真匹配的下界，让数据看起来比实际更难分。

    用 NamedTuple 而不是裸 5-tuple：``impostor_score`` 和 ``expected_score`` 都是
    float 且相邻，位置写错会把"最高冒充者"和"真匹配"整个调包，而任何测试都察觉不到。
    """

    top1_record_id: str | None
    top1_score: float
    similarities: dict[str, float]
    impostor_score: float
    expected_score: float | None


def _should_catch(expected_score: float | None, tau: float) -> bool | None:
    """明文判据，与协议第二轮的判据保持同一个符号。

    第二轮把阈值烘进密文：``add_plain(enc_score, -mask * float(tau))``，再判
    ``value > eps``（``protocol.py:297`` / ``protocol.py:315``）。mask 恒正，
    所以判据等价于 ``score > tau`` —— **严格的 >，不是 >=**。明文侧必须一致，
    否则 ``score == tau`` 这个整齐的分数会在两边得到不同答案。

    真匹配没被载入库里时返回 ``None``：那是"没载入"，不是"没匹配上"，当成
    ``False`` 会静默混进分母，把数据问题伪装成匹配失败。
    """

    return None if expected_score is None else bool(expected_score > tau)


# 第二轮的实际判据是 score > tau + eps/mask，mask ∈ [1, 10]，
# 所以预期分落在 (tau, tau + eps] 这一段的查询卡在 CKKS 噪声里，两边倒都正常。
_BOUNDARY_EPS = MULTI_ATTRIBUTE_DECRYPT_EPS


def _is_identical(expected_score: float | None) -> bool:
    """完全一致（未扰动）的查询 —— 明文分恰为 1.0。"""

    return expected_score is not None and expected_score >= 1.0 - 1e-9


def _miss_cause(
    should_catch: bool | None,
    catch: bool,
    cluster_hit: bool,
    expected_score: float | None,
    tau: float,
) -> str | None:
    """归因一条「应 catch 却没 catch」。没漏的时候返回 ``None``。

    - ``round1_cluster``：真匹配不在选中的簇里，第二轮压根没检查到它。
      这一项**与 tau 无关**，调阈值消不掉。
    - ``boundary``：分数落在 ``(tau, tau + eps]`` 这段 CKKS 噪声带里，两边倒都正常。
    - ``unexplained``：真匹配就在选中簇里、分数又超出噪声带，第二轮没有任何理由漏。
      **这一项应当恒为 0** —— 非 0 就是协议 bug，汇总里会显式报警。
    """

    if should_catch is not True or catch:
        return None
    if not cluster_hit:
        return "round1_cluster"
    if expected_score is not None and 0.0 < expected_score - tau <= _BOUNDARY_EPS:
        return "boundary"
    return "unexplained"


def _best_plaintext_match(database, db_matrix, query, schema, db_index: Mapping[str, int] | None = None):
    """库向量矩阵与查询向量做一次点积，取 argmax、真匹配分与「冒充者」最高分。

    比逐条 ``plaintext_similarity`` 快几个数量级 —— 库只编码一次，之后每条查询
    就是一次 ``(N, d) @ (d,)``。查询向量的 MinHash 编码是这里最贵的一步，所以
    ``expected_score`` 在这同一个函数里顺手取出，而不是到外面重算一遍。

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

    # 真值集合不保证只有一条（CSV 里是 "|" 分隔的多 id），所以取其中最高的那一条。
    expected_score = None
    if expected and db_index:
        hits = [db_index[rid] for rid in expected if rid in db_index]
        if hits:
            expected_score = float(max(scores[i] for i in hits))

    return PlaintextMatch(
        top1_record_id=database[best].record_id,
        top1_score=float(scores[best]),
        similarities=attribute_similarities(query.record, database[best], schema),
        impostor_score=impostor,
        expected_score=expected_score,
    )


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def _pct(numerator: int, denominator: int) -> str:
    """比率一律带绝对计数输出 —— 这是防刷分的第一道护栏，别把它拆掉。"""

    if not denominator:
        return f"{numerator}/{denominator} = n/a"
    return f"{numerator}/{denominator} = {numerator / denominator:.1%}"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    """最近秩百分位（nearest-rank）。纯 Python 实现 —— ``_summarize`` 不引入 numpy。"""

    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def _summarize(
    rows: list[dict],
    tau: float,
    *,
    encrypted: bool,
    database_size: int | None = None,
) -> dict:
    """从 ``rows`` 推导全部指标。

    刻意做成纯函数：只吃 rows，不碰 numpy / tenseal / 文件，可以直接单测；也保证
    ``summary`` 与同一个 ``result`` 里的 ``rows``、CSV 三者不可能漂移（早先的版本在
    循环里并行累加计数器，summary 与它旁边的产物完全可能对不上）。

    **召回率的分母是 ``should_catch``** —— 真匹配那条记录的明文分 > tau，而不是
    "标签为真"。被扰动到低于阈值的副本按协议判据本来就不该 catch，算进分母等于让
    口径背上一份协议不负责的债。代价是分母里含 tau，所以每个比率都连同绝对计数
    一起报，并且把与 tau 无关的 ``cluster_recall`` / 标定窗口并列打印，见
    ``print_summary``。
    """

    total = len(rows)
    positives = [row for row in rows if row["label"]]
    negatives = total - len(positives)
    tau = float(tau)

    summary: dict = {
        "queries": total,
        "positives": len(positives),
        "negatives": negatives,
        "tau": tau,
        "recall_definition": (
            "should_catch = plaintext true-match score > tau; "
            "recall = #{catch and should_catch} / #{should_catch}"
        ),
    }

    if positives:
        # 只在正例上算 top-1 —— 负例的正确记录根本不在库里，必然"未命中"，
        # 混在一起算会把命中率稀释成没有意义的数字。
        summary["plaintext_top1_hits"] = sum(
            1
            for row in positives
            if row["top1_record_id"] is not None
            and row["top1_record_id"] in row["expected_record_ids"]
        )
        summary["plaintext_top1_recall_positives_only"] = (
            summary["plaintext_top1_hits"] / len(positives)
        )

    # 负例被误判为命中 —— 只要最高分越过阈值就会发生。负例没有真匹配，
    # 它的 impostor_score 就是全库最高分。
    false_positives = sum(
        1
        for row in rows
        if not row["label"] and _should_catch(row["impostor_score"], tau)
    )
    summary["plaintext_false_positives_at_tau"] = false_positives
    if negatives:
        summary["plaintext_false_positive_rate"] = false_positives / negatives

    # ---- 明文量：--no-encrypted 下同样成立，别挪进加密分支 ----

    # should_catch 一律由 (expected_score, tau) **现算**，不读 row["should_catch"]。
    # 那个字段是给 CSV/JSON 留的序列化产物；汇总必须以传进来的 tau 为准，否则换一个
    # tau 汇总同一批 rows 会读出上一轮的判定，报出来的分母与自己的 tau 参数对不上。
    def should_catch_of(row: dict) -> bool | None:
        return _should_catch(row["expected_score"], tau)

    summary["should_catch_positives"] = sum(
        1 for row in positives if should_catch_of(row) is True
    )
    below_tau = [row for row in positives if should_catch_of(row) is False]
    orphaned = [row for row in positives if row["expected_score"] is None]
    summary["positives_below_tau"] = len(below_tau)
    summary["positives_below_tau_ids"] = [row["query_id"] for row in below_tau]
    # 真匹配没被载入库里。这一项必须为 0，否则分母是悄悄缩水的。
    summary["positives_missing_true_match_in_db"] = len(orphaned)
    summary["positives_missing_true_match_in_db_ids"] = [
        row["query_id"] for row in orphaned
    ]
    summary["identical_queries"] = sum(
        1 for row in positives if _is_identical(row["expected_score"])
    )

    # 阈值该定在哪：真匹配分要高于冒充者分。两个数都从明文算出，与协议输出无关。
    expected_scores = [
        row["expected_score"] for row in positives if row["expected_score"] is not None
    ]
    impostor_scores = [row["impostor_score"] for row in rows if row["impostor_score"] is not None]
    lo = min(expected_scores) if expected_scores else None
    hi = max(impostor_scores) if impostor_scores else None
    summary["lowest_expected_match_score"] = lo
    summary["highest_impostor_score"] = hi
    if lo is not None and hi is not None:
        if lo > hi:
            summary["calibration_window"] = {"low": hi, "high": lo}
            summary["tau_inside_calibration_window"] = hi < tau < lo
        else:
            summary["tau_inside_calibration_window"] = False
            summary["true_matches_overlap_impostors"] = True

    summary["tau_guard"] = {
        "tau": tau,
        "max_impostor": hi,
        # 高于最高冒充者分之后，再抬阈值挡不掉任何假正例，只会把正例从召回分母里
        # 删掉 —— 数字变好看，错误一条没修。
        "tau_exceeds_max_impostor": hi is not None and tau > hi,
        # tau == 1.0 时完全一致的查询（分数恰为 1.0）会被 strict > 全部拒掉，
        # 而 similarity_threshold 的校验是 [0, 1] 闭区间，所以这个值可达。
        "tau_at_or_above_one": tau >= 1.0,
        "positives_dropped_by_tau": len(below_tau),
    }
    # tau > 最高冒充者分本身是**标定正确的样子**，不是问题：阈值就该盖住冒充者上限。
    # 真正的风险是"盖住了却还在删正例"，所以护栏必须按代价分级，否则健康 run 和刷分
    # run 会打出同一句警告，读者学会忽略它之后，真出事那次也就没人看了。
    if hi is None:
        verdict = "unknown"
    elif tau <= hi:
        # 这个方向才该紧张：冒充者能直接越过阈值产生假正例。
        verdict = "below_ceiling"
    elif below_tau:
        verdict = "above_ceiling_with_cost"
    else:
        verdict = "above_ceiling_no_cost"
    summary["tau_guard"]["verdict"] = verdict

    if not encrypted:
        return summary

    # ---- 加密量 ----

    should_catch_rows = [row for row in positives if should_catch_of(row) is True]
    caught_should = [row for row in should_catch_rows if row["encrypted"]["catch"]]
    misses = [row for row in should_catch_rows if not row["encrypted"]["catch"]]
    causes = [row["encrypted"].get("miss_cause") for row in misses]

    summary["positives_scored"] = len(positives)
    summary["encrypted_catches_positives"] = len(caught_should)
    summary["recall_should_catch"] = (
        len(caught_should) / len(should_catch_rows) if should_catch_rows else None
    )
    summary["should_catch_misses"] = len(misses)
    summary["should_catch_miss_ids"] = [row["query_id"] for row in misses]
    summary["misses_round1_cluster"] = causes.count("round1_cluster")
    summary["misses_boundary"] = causes.count("boundary")
    # 这一项应当恒为 0；非 0 就是协议 bug，print_summary 会显式报警。
    summary["misses_unexplained"] = causes.count("unexplained")
    summary["misses_unexplained_ids"] = [
        row["query_id"]
        for row in misses
        if row["encrypted"].get("miss_cause") == "unexplained"
    ]

    identical = [row for row in positives if _is_identical(row["expected_score"])]
    summary["identical_caught"] = sum(1 for row in identical if row["encrypted"]["catch"])
    summary["identical_violations"] = [
        row["query_id"] for row in identical if not row["encrypted"]["catch"]
    ]

    # 与 tau 无关的老口径，留着做对照：改 tau 会让它和新口径反向变化。
    summary["encrypted_catches_all_positives"] = sum(
        1 for row in positives if row["encrypted"]["catch"]
    )
    summary["encrypted_recall_all_positives"] = (
        summary["encrypted_catches_all_positives"] / len(positives) if positives else None
    )
    summary["cluster_hits"] = sum(
        1 for row in positives if row["encrypted"].get("cluster_hit")
    )
    summary["cluster_recall"] = (
        summary["cluster_hits"] / len(positives) if positives else None
    )
    # 只探 top-1 时真匹配会落在哪。**这才是那个 tau 无关又有区分度的锚点**：
    # probes="all" 下 cluster_recall 按构造恒为 1.0，报它等于什么都没说；而
    # top-1 命中率完全由数据决定，改 tau、改探针数都动不了它 —— 所以簇召回要跟
    # top-1 那条线一起读，才知道当前探针数到底买回了多少漏检。
    summary["cluster_hits_top1_only"] = sum(
        1 for row in positives if row["encrypted"].get("cluster_hit_top1_only")
    )
    summary["cluster_recall_top1_only"] = (
        summary["cluster_hits_top1_only"] / len(positives) if positives else None
    )
    ranks = [
        row["encrypted"]["cluster_rank"]
        for row in positives
        if row["encrypted"].get("cluster_rank") is not None
    ]
    summary["cluster_rank_mean"] = _mean(ranks)
    summary["cluster_rank_p95"] = _percentile(ranks, 0.95)
    summary["cluster_rank_max"] = int(max(ranks)) if ranks else None

    # ---- 第二轮扫描成本：多簇探测后这才是真正的判据 ----
    # 成本单位是第二轮扫描列数（密文-密文点积次数）。全库线性扫描是 database_size 列，
    # 所以这一节直接给出省了多少。
    checked = [row["encrypted"]["checked_columns"] for row in rows]
    summary["probes_per_query"] = (
        rows[0]["encrypted"].get("probed_clusters") if rows else None
    )
    summary["scan_columns_mean"] = _mean(checked)
    summary["scan_columns_median"] = _percentile(checked, 0.5)
    summary["scan_columns_p95"] = _percentile(checked, 0.95)
    summary["scan_columns_max"] = int(max(checked)) if checked else None
    summary["scan_columns_mean_positives"] = _mean(
        [row["encrypted"]["checked_columns"] for row in positives if row["encrypted"]]
    )
    # 负例（库里根本没有这条记录）在多簇穷举下必然扫满整库 —— early stop 没有可停的
    # 地方，因为整库都不含它。这一列必须单独报，否则"平均只扫 1%"会被读成对任意查询
    # 都成立：它对**命中**的查询成立，对空查不成立。
    summary["scan_columns_mean_negatives"] = _mean(
        [row["encrypted"]["checked_columns"] for row in rows if not row["label"]]
    )
    if database_size is not None:
        summary["full_scan_columns"] = int(database_size)
        if summary["scan_columns_mean"] is not None and database_size:
            summary["scan_columns_fraction"] = summary["scan_columns_mean"] / database_size
    # tau 无关的绝对漏掉条数：真匹配压根没被检查过的正例。**判 run 该看这个**——
    # 上面那个 should_catch_misses 的分母含 tau，抬阈值就能把它压到 0（FEBRL tau=0.9
    # 时 12 -> 0），而这一项在任何 tau 下都是 13。
    summary["cluster_misses"] = len(positives) - summary["cluster_hits"]
    summary["agreements"] = sum(
        1 for row in rows if row["encrypted"]["agrees_with_label"]
    )
    summary["encrypted_agreement"] = summary["agreements"] / total if total else None
    # 负例被 catch —— 新口径下的假正例。
    summary["over_reports"] = sum(
        1 for row in rows if not row["label"] and row["encrypted"]["catch"]
    )
    summary["over_report_rate"] = (
        summary["over_reports"] / negatives if negatives else None
    )
    return summary


def print_summary(summary: dict) -> None:
    """终端报告。

    **每个比率都必须与绝对计数同行输出**，并且把 ``cluster recall``（唯一与 tau 无关
    的量）紧跟在新召回率下面。这不是排版偏好：新召回率的分母含 tau，单看一个数时调高
    tau 就能把它刷到 100% 而错误一条不减 —— FEBRL tau=0.9 实测就是 100.0%，同屏的
    ``cluster recall`` 仍是 87/100。

    漏掉条数（``misses among those``）**不是** tau 无关的：它的分母就是 should-catch
    集合，抬 tau 会把这批查询整批移出分母，于是显示 0。屏幕上必须写明这一点，否则
    "0 misses" 会被读成"没有错误"。
    """

    tau = summary["tau"]
    positives = summary["positives"]
    negatives = summary["negatives"]
    probed = summary.get("probes_per_query")
    policy = (
        ""
        if probed is None
        else f"   round-2 probes/query={probed}"
    )
    print(
        f"queries: {summary['queries']} ({positives} positive / {negatives} negative)"
        f"   tau={tau:.3f}{policy}"
    )

    if positives:
        print(
            "plaintext top-1 recall, positives only (debug reference): "
            + _pct(summary["plaintext_top1_hits"], positives)
        )
    if negatives:
        print(
            f"plaintext false positives at tau={tau:g} (debug reference): "
            + _pct(summary["plaintext_false_positives_at_tau"], negatives)
        )

    # 分母的定义必须写在数字旁边，否则这个数会被读成"识别召回率"。
    print("\nrecall denominator: should_catch = plaintext true-match score > tau")
    print(f"  identical queries (expected score = 1.000): {summary['identical_queries']}")
    dropped = summary["positives_below_tau_ids"]
    excluded = f" (below tau, excluded: {', '.join(dropped)})" if dropped else ""
    print(
        f"  should-catch positives: {summary['should_catch_positives']}/{positives}{excluded}"
    )
    if summary["positives_missing_true_match_in_db"]:
        print(
            f"  WARNING: {summary['positives_missing_true_match_in_db']} positive(s) have no "
            "true match in the loaded database and are NOT in the denominator: "
            + ", ".join(summary["positives_missing_true_match_in_db_ids"])
        )

    if summary.get("positives_scored"):
        print(
            "\nencrypted recall, should-catch only : "
            + _pct(summary["encrypted_catches_positives"], summary["should_catch_positives"])
        )
        print(
            f"  misses among those: {summary['should_catch_misses']}   "
            f"[round-1 cluster miss {summary['misses_round1_cluster']} / "
            f"boundary {summary['misses_boundary']} / "
            f"unexplained {summary['misses_unexplained']}]"
        )
        # 这一项**不是** tau 无关的：它的分母就是上面那个 should-catch 集合，抬 tau 会把
        # 第一轮漏掉的查询整批移出分母，于是这里显示 0。FEBRL tau=0.9 就是 12 -> 0，而
        # 那 12 条错误一条没修。真正 tau 无关的是紧跟着的 cluster recall。
        print(
            f"  ^ tau-DEPENDENT: the denominator is the should-catch set, so raising tau moves\n"
            f"    round-1 misses out of it and this count drops. tau-free: "
            f"{summary['cluster_misses']} of {positives}\n"
            "    positives were never checked at all -- that is the line below."
        )
        probed = summary["probes_per_query"]
        exhaustive = summary["cluster_hits"] >= positives and probed and probed > 1
        print(
            f"cluster recall (true match landed in the {probed} probed cluster(s), tau-free): "
            + _pct(summary["cluster_hits"], positives)
        )
        # 探针数 > 1 时上面这个数会趋近 100%，于是它不再是"数据有多难"的读数，而是
        # "我们探了多少"的读数。真正没被调参动过的是只探 top-1 的命中率 —— 它由簇
        # 几何决定，改 tau、改探针数都不动。两个一起打，读者才知道成本买到了什么。
        if probed and probed > 1:
            print(
                "  ^ with more than one probe this saturates by construction; the "
                "tau-free anchor\n    is the top-1-only line below, which the probe count "
                "cannot move."
            )
            print(
                "  top-1-only cluster recall (what a single probe would have caught): "
                + _pct(summary["cluster_hits_top1_only"], positives)
                + f"   true-cluster rank mean {summary['cluster_rank_mean']:.2f} / "
                f"p95 {summary['cluster_rank_p95']:.0f} / max {summary['cluster_rank_max']}"
            )
        if exhaustive:
            print(
                f"  scan cost: mean {summary['scan_columns_mean']:.1f} columns "
                f"({summary['scan_columns_fraction']:.1%} of a full scan) / "
                f"median {summary['scan_columns_median']:.0f} / "
                f"p95 {summary['scan_columns_p95']:.0f} / "
                f"max {summary['scan_columns_max']}   [full scan = "
                f"{summary['full_scan_columns']} columns]"
            )
            if summary["scan_columns_mean_negatives"] is not None:
                print(
                    f"  same cost on queries with NO true match: "
                    f"{summary['scan_columns_mean_negatives']:.0f} columns -- exhaustive\n"
                    "    probing cannot early-stop on an absent record, so those queries "
                    "pay the\n    whole database. The sublinear saving is real only for "
                    "queries that hit."
                )
        if summary["misses_unexplained"]:
            print(
                f"  !! {summary['misses_unexplained']} UNEXPLAINED miss(es): the true match "
                "was in the selected cluster and scored clear of tau, so round 2 had no "
                "reason to miss it. Protocol bug, not a tuning issue: "
                + ", ".join(summary["misses_unexplained_ids"])
            )
        print(
            "encrypted recall, all positives (whole population, for contrast): "
            + _pct(summary["encrypted_catches_all_positives"], positives)
        )
        if negatives:
            print(
                "over-reports (negatives caught above tau): "
                + _pct(summary["over_reports"], negatives)
            )

        total_identical = summary["identical_queries"]
        if total_identical:
            verdict = "OK" if not summary["identical_violations"] else "VIOLATED"
            print(
                f"identical-query invariant: {summary['identical_caught']}/{total_identical} "
                f"caught [{verdict}]"
            )
            if summary["identical_violations"]:
                print(
                    "  !! identical queries MUST catch: "
                    + ", ".join(summary["identical_violations"])
                )
        else:
            # 没被触发过 ≠ 通过。这个状态要说出来，否则绿灯是空跑的。
            print("identical-query invariant: not exercised (no identical queries)")

    # ---- tau 的标定依据：全报告里唯一与 tau 无关的锚点 ----
    lo = summary["lowest_expected_match_score"]
    hi = summary["highest_impostor_score"]
    if lo is not None and hi is not None:
        print(
            "\nplaintext calibration (tau anchoring only, not an evaluation metric): "
            f"lowest expected-match {lo:.3f} / highest impostor {hi:.3f}"
        )
        if summary.get("true_matches_overlap_impostors"):
            print(
                "  -> true matches and impostors OVERLAP here; no tau separates them. "
                "Add attributes, raise weights, or accept errors."
            )
        elif "tau_inside_calibration_window" in summary:
            print(
                f"  -> tau must be inside ({hi:.3f}, {lo:.3f}); current tau={tau:g} "
                + (
                    "is inside"
                    if summary["tau_inside_calibration_window"]
                    else "is OUTSIDE"
                )
            )

    guard = summary["tau_guard"]
    verdict = guard["verdict"]
    if verdict == "below_ceiling":
        print(
            f"\ntau guard: tau={tau:.3f} is BELOW the highest impostor score {hi:.3f}.\n"
            "  An impostor can score above tau, so the protocol can report CATCH for a\n"
            "  query whose record is not in the database at all. Raise tau into "
            f"({hi:.3f}, {lo:.3f}]."
        )
    elif verdict == "above_ceiling_with_cost":
        print(
            f"\ntau guard: tau={tau:.3f} sits above the impostor ceiling {hi:.3f}, and it "
            f"cost\n  {guard['positives_dropped_by_tau']} positive(s) to get there "
            f"({positives} -> {summary['should_catch_positives']}).\n"
            f"  Everything above {hi:.3f} suppresses nothing -- no impostor can reach it --\n"
            "  so that entire span is pure subtraction from the recall denominator. Judge a\n"
            "  run by the tau-free cluster recall line, which this cannot move."
        )
    elif verdict == "above_ceiling_no_cost":
        # 健康形态：盖住冒充者上限且一条正例没删。必须说"没问题"，否则真出事的
        # 那次会淹没在一贯的警告里。
        print(
            f"\ntau guard: tau={tau:.3f} is above the impostor ceiling {hi:.3f} and drops no\n"
            "  positive from the denominator. Calibrated correctly -- nothing to flag."
        )
    if guard["tau_at_or_above_one"]:
        print(
            f"\ntau guard: tau={tau:g} >= 1.0. An identical query scores exactly 1.0 and the\n"
            "  protocol's test is a STRICT >, so every identical query must miss -- the\n"
            "  identical-query invariant cannot hold at this tau."
        )

    print(
        "\nNOTE: the protocol returns only the sign of the threshold test, the selected\n"
        "cluster index, and the column index -- never the id of the matched record. So a\n"
        "CATCH means \"some record in the selected cluster scored above tau\", not \"the\n"
        "right record was found\": the recall above is a DETECTION rate, not identification\n"
        "precision/recall. It also has tau in its own denominator, so raising tau moves the\n"
        "hardest queries out of the denominator and pushes the number toward 100% without\n"
        "fixing anything. It reports agreement with the threshold; it is not a target."
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
            out["enc_probed_clusters"] = ""
            out["enc_true_match_cluster"] = ""
            out["enc_true_match_cluster_rank"] = ""
            out["enc_cluster_hit"] = ""
            out["enc_cluster_hit_top1_only"] = ""
            out["enc_miss_cause"] = ""
        else:
            out["encrypted"] = "CATCH" if encrypted["catch"] else "no-catch"
            out["enc_agrees_with_label"] = encrypted["agrees_with_label"]
            out["enc_selected_cluster"] = encrypted["selected_cluster"]
            out["enc_checked_columns"] = encrypted["checked_columns"]
            out["enc_probed_clusters"] = encrypted.get("probed_clusters", "")
            out["enc_true_match_cluster"] = encrypted.get("true_match_cluster", "")
            out["enc_true_match_cluster_rank"] = encrypted.get("cluster_rank", "")
            out["enc_cluster_hit"] = encrypted.get("cluster_hit", "")
            out["enc_cluster_hit_top1_only"] = encrypted.get("cluster_hit_top1_only", "")
            out["enc_miss_cause"] = encrypted.get("miss_cause") or ""

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
    parser.add_argument(
        "--k-mode",
        default="sqrt",
        help="cluster-count policy: sqrt / log2 / auto / an integer / fixed:<k> "
        "(default sqrt; 'auto' is the measured optimum under exhaustive probing)",
    )
    parser.add_argument(
        "--probes",
        default=str(DEFAULT_MULTI_ATTRIBUTE_PROBES),
        help="round-2 clusters to probe, in descending centroid score: an integer, "
        "or 'all' for exhaustive-with-early-stop (default %s)"
        % DEFAULT_MULTI_ATTRIBUTE_PROBES,
    )
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

    # 真匹配分明文算，`--no-encrypted` 的便宜 sweep 也要用，所以放在加密分支之外。
    db_index = {record.record_id: i for i, record in enumerate(database)}
    if len(db_index) != len(database):
        # 整套 expected_score 机制以 record_id 为键，上游并不保证 id 唯一。
        print(
            f"Warning: {len(database) - len(db_index)} duplicate record_id(s) in the "
            "database; expected-score lookup keeps the last occurrence"
        )

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

    rows: list[dict] = []
    tau = float(schema.similarity_threshold)

    print(f"{'query':<16}{'label':<7}{'true':>7}{'imp':>7}  {'top-1':<14}{'attr breakdown':<40}{'encrypted'}")
    print("-" * 88)
    for query in queries:
        match = _best_plaintext_match(database, db_matrix, query, schema, db_index)
        # 一律转成 Python float —— numpy 标量 json.dumps 序列化不了。
        best_score = float(match.top1_score)
        impostor = float(match.impostor_score)
        expected_score = None if match.expected_score is None else float(match.expected_score)
        should_catch = _should_catch(expected_score, tau)

        breakdown = " ".join(f"{k}={v:.2f}" for k, v in match.similarities.items())

        row: dict = {
            "query_id": query.record.record_id,
            "label": bool(query.label),
            "expected_record_ids": sorted(query.expected_record_ids),
            "top1_record_id": match.top1_record_id,
            "top1_score": round(best_score, 6),
            "expected_score": None if expected_score is None else round(expected_score, 6),
            # 离阈值多远。符号就是 should_catch，绝对值是到边界的距离 ——
            # 标定 tau 时最该看的一列。
            "expected_score_margin": (
                None if expected_score is None else round(expected_score - tau, 6)
            ),
            "tau": tau,
            "should_catch": should_catch,
            "impostor_score": round(impostor, 6),
            "similarities": {k: round(float(v), 6) for k, v in match.similarities.items()},
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
                probes=args.probes,
            )
            verdict = "CATCH" if run.catch else "no-catch"
            # 真匹配的簇落在第一轮质心排序的第几位。下面两个不同的量都由它决定：
            # ``cluster_hit`` 看的是**探测过**的集合（probes="all" 时恒为真，没有
            # 区分度），``cluster_rank`` 看的是**只探 top-1** 时会怎样 —— 后者才是
            # 那个与 tau 无关、又能被调参刷不动的锚点。
            # 真值集合可能是多 id，所以用 any / min 而不是只看第一条。
            true_clusters = [
                cluster_of[rid]
                for rid in sorted(query.expected_record_ids)
                if rid in cluster_of
            ]
            probed = tuple(run.probed_clusters)
            rank = next(
                (probed.index(c) for c in true_clusters if c in probed),
                None,
            )
            encrypted: dict = {
                "catch": bool(run.catch),
                "agrees_with_label": bool(run.catch == query.label),
                "selected_cluster": int(run.selected_cluster),
                "checked_columns": int(run.checked_columns),
                "probed_clusters": len(probed),
            }
            if query.label:
                encrypted["true_match_cluster"] = true_clusters[0] if true_clusters else None
                encrypted["cluster_rank"] = rank
                encrypted["cluster_hit"] = bool(true_clusters) and rank is not None
                encrypted["cluster_hit_top1_only"] = rank == 0
                encrypted["miss_cause"] = _miss_cause(
                    should_catch,
                    run.catch,
                    bool(true_clusters) and rank is not None,
                    expected_score,
                    tau,
                )
            row["encrypted"] = encrypted
            rank_note = "" if rank is None else f" rank={rank}"
            tail = (
                f"  {verdict:<9} cluster={run.selected_cluster:<4} "
                f"checked={run.checked_columns}{rank_note}"
            )

        rows.append(row)
        print(
            f"{str(query.record.record_id):<16}{str(query.label):<7}{best_score:>7.3f}"
            f"{impostor:>7.3f}  {(match.top1_record_id or '-'):<14}{breakdown:<40}{tail}"
        )

    print("-" * 88)
    summary = _summarize(
        rows, tau, encrypted=not args.no_encrypted, database_size=len(database)
    )
    print_summary(summary)

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
            "probes": args.probes,
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
