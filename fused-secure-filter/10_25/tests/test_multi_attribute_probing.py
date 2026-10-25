"""多簇探测 + k 选择策略 + 扫描成本报告。

分三块：
1. ``choose_k`` 的 log2 / auto 档，以及 ``probes`` 参数的归一化；
2. 手工搭一份「真匹配不在 top-1 簇」的离线产物，验证 ``probes`` 真的把它捞回来
   —— 这是多簇探测存在的唯一理由，单探针下那条真匹配**没有被检查的机会**；
3. demo 汇总/报告里新加的扫描成本口径（含"空查必然扫满整库"这条反直觉结论）。

``cluster_sizes`` 的宽度裁剪单独测：第 2 块用它把第二轮列数从 ``max_size`` 压到
真实成员数，测错的话成本数字会静默偏大。
"""

import importlib.util

import numpy as np
import pytest

from config.params import choose_k
from multi_attribute import AttributeRecord, AttributeSchema
from multi_attribute.protocol import (
    EXHAUSTIVE_PROBE_MODES,
    MultiOfflineArtifacts,
    _resolve_probes,
    choose_multi_clusters,
    column_wise_multi_probe_matching,
    prepare_party_a_multi_query,
    run_multi_attribute_protocol,
)

HAS_TENSEAL = importlib.util.find_spec("tenseal") is not None


# ---------------------------------------------------------------------------
# 1. k 档位与 probes 归一化
# ---------------------------------------------------------------------------


def test_choose_k_log2_is_floor_log2():
    assert choose_k(5000, "log2") == 12
    assert choose_k(1024, "log2") == 10
    # 1 的 log2 是 0，而 k 必须 >= 1。
    assert choose_k(1, "log2") == 1
    assert choose_k(2, "log2") == 1


def test_choose_k_auto_is_the_measured_factor():
    assert choose_k(5000, "auto") == 99
    assert choose_k(500, "auto") == 31
    assert choose_k(5000, "auto") > choose_k(5000, "sqrt")


def test_choose_k_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported k selection mode"):
        choose_k(100, "cbrt")


@pytest.mark.parametrize("mode", sorted(EXHAUSTIVE_PROBE_MODES))
def test_resolve_probes_exhaustive_tokens(mode):
    assert _resolve_probes(mode, k=70) == 70
    assert _resolve_probes(mode.upper(), k=70) == 70


def test_resolve_probes_clamps_and_rejects():
    assert _resolve_probes(None, k=12) == 12
    assert _resolve_probes("5", k=12) == 5
    # 请求的探针数超过簇数时收敛到簇数，而不是报错 —— 穷举和"探满"是一回事。
    assert _resolve_probes(999, k=12) == 12
    with pytest.raises(ValueError, match="probes must be >= 1"):
        _resolve_probes(0, k=12)
    with pytest.raises(ValueError, match="probes must be a positive int"):
        _resolve_probes("most", k=12)


# ---------------------------------------------------------------------------
# 2. 真匹配不在 top-1 簇里
# ---------------------------------------------------------------------------

# 单属性 exact，dim=64：取值直接映射到一个哈希桶，同值向量完全相同。
_SCHEMA = AttributeSchema.from_dict(
    {
        "attributes": [
            {"name": "code", "kind": "exact", "weight": 1.0,
             "blocks": 1, "buckets_per_block": 64},
        ],
        "similarity_threshold": 0.5,
    }
)

_QUERY = AttributeRecord({"code": "alpha-0001"})
_MATCH = AttributeRecord({"code": "alpha-0001"})   # 真匹配，与查询同值 -> 分 1.0
_DECOY = AttributeRecord({"code": "omega-9999"})   # 冒充者，同簇但不该命中


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _artifacts(*, cluster_sizes: np.ndarray | None = None, max_size: int = 2):
    """真匹配被放进簇 1，而查询的质心最近邻是簇 0。

    ``centroids`` / ``cluster_matrix`` 全部手工给，不跑 k-means —— 这里要测的是
    ``probes`` 的选择与扫描，不是聚类质量。
    """

    from multi_attribute.encoder import encode_record_vectors

    _, q_match = encode_record_vectors([_QUERY], _SCHEMA)
    _, m_match = encode_record_vectors([_MATCH], _SCHEMA)
    _, d_match = encode_record_vectors([_DECOY], _SCHEMA)
    q_vec = q_match[0]

    # centroid 0 就是查询向量本身（cos=1，稳居第一）；1 与查询有重叠但不完全相同；
    # 2 与查询正交（cos=0，必然垫底）。
    other = d_match[0]
    centroids = np.stack(
        [q_vec, _unit(q_vec + 0.25 * other), other], axis=0
    ).astype(np.float64)

    cluster_matrix = np.zeros((3, max_size, q_vec.shape[0]), dtype=np.float64)
    # 簇 0 是和查询无关的记录，扫两列都不该越阈（这正是单探针漏掉的那次扫描）。
    cluster_matrix[0, 0, :] = other
    cluster_matrix[0, 1, :] = d_match[0]
    # 真匹配落在簇 1 的第 0 列。
    cluster_matrix[1, 0, :] = m_match[0]

    return MultiOfflineArtifacts(
        centroids=centroids,
        cluster_matrix=cluster_matrix,
        scaler_mean=np.zeros(q_vec.shape[0], dtype=np.float64),
        scaler_scale=np.ones(q_vec.shape[0], dtype=np.float64),
        cluster_assignments=np.array([0, 1], dtype=np.int32),
        max_size=max_size,
        schema=_SCHEMA,
        cluster_sizes=cluster_sizes,
    )


@pytest.mark.skipif(not HAS_TENSEAL, reason="TenSEAL not installed")
def test_single_probe_misses_when_the_true_cluster_ranks_second():
    """单探针是本功能的对照组：它必须漏，否则这个测试什么也没证明。"""

    run = run_multi_attribute_protocol(
        [], _QUERY, cfg=_SCHEMA, artifacts=_artifacts(), probes=1, early_stop=True
    )
    assert run.catch is False
    assert run.probed_clusters == (0,)
    assert run.selected_cluster == 0
    assert run.checked_columns == 2  # 簇 0 的两列


@pytest.mark.skipif(not HAS_TENSEAL, reason="TenSEAL not installed")
def test_second_probe_recovers_the_match_the_first_probe_missed():
    run = run_multi_attribute_protocol(
        [], _QUERY, cfg=_SCHEMA, artifacts=_artifacts(), probes=2, early_stop=True
    )
    assert run.catch is True
    assert run.probed_clusters == (0, 1)
    assert run.first_positive_cluster == 1
    assert run.first_positive_column == 0
    # 簇 0 扫满 2 列 + 簇 1 第 0 列即刻早停。
    assert run.checked_columns == 3


@pytest.mark.skipif(not HAS_TENSEAL, reason="TenSEAL not installed")
def test_exhaustive_probing_orders_clusters_by_centroid_score():
    run = run_multi_attribute_protocol(
        [], _QUERY, cfg=_SCHEMA, artifacts=_artifacts(), probes="all", early_stop=False
    )
    # 3 个簇全探；顺序按质心分降序 —— 早停能不能省成本全靠这个顺序。
    assert run.probed_clusters == (0, 1, 2)
    # 不早停就要把三个簇都扫完：2 + max_size + max_size。
    assert run.checked_columns == 2 + 2 + 2


@pytest.mark.skipif(not HAS_TENSEAL, reason="TenSEAL not installed")
def test_cluster_sizes_trim_the_scan_to_real_membership():
    """补零对齐的列不该被扫 —— 多探针会把这个浪费乘以探针数。"""

    artifacts = _artifacts(cluster_sizes=np.array([2, 1, 0], dtype=np.int64), max_size=2)
    assert [artifacts.width_of(c) for c in range(3)] == [2, 1, 0]

    run = run_multi_attribute_protocol(
        [], _QUERY, cfg=_SCHEMA, artifacts=artifacts, probes=1, early_stop=False
    )
    # 簇 0 只有 2 个真实成员，max_size 也是 2 -> 2 列；簇 2 宽度 0 时一列都不扫。
    assert run.checked_columns == 2

    full = run_multi_attribute_protocol(
        [], _QUERY, cfg=_SCHEMA, artifacts=artifacts, probes="all", early_stop=False
    )
    assert full.probed_clusters == (0, 1, 2)
    assert full.checked_columns == 2 + 1 + 0


@pytest.mark.skipif(not HAS_TENSEAL, reason="TenSEAL not installed")
def test_missing_cluster_sizes_degrades_to_max_size():
    """手工构造的产物没带 cluster_sizes 时，退化成扫满 max_size 而不是崩掉。"""

    artifacts = _artifacts(cluster_sizes=None, max_size=2)
    assert artifacts.width_of(0) == 2
    run = run_multi_attribute_protocol(
        [], _QUERY, cfg=_SCHEMA, artifacts=artifacts, probes="all", early_stop=False
    )
    assert run.checked_columns == 2 * 3


@pytest.mark.skipif(not HAS_TENSEAL, reason="TenSEAL not installed")
def test_probe_generator_yields_cluster_and_column_tags():
    artifacts = _artifacts(cluster_sizes=np.array([2, 1, 1], dtype=np.int64))
    first_req, state = prepare_party_a_multi_query(_QUERY, artifacts, cfg=_SCHEMA)
    from multi_attribute.protocol import compare_multi_to_centroids

    request, probed = choose_multi_clusters(
        compare_multi_to_centroids(first_req, artifacts.centroids), state, probes="all"
    )
    assert probed == (0, 1, 2)
    tags = [
        (cluster, column)
        for cluster, column, _ in column_wise_multi_probe_matching(
            artifacts.cluster_matrix,
            request,
            first_req.public_context_bytes,
            tau=0.5,
            cluster_sizes=artifacts.cluster_sizes,
        )
    ]
    assert tags == [(0, 0), (0, 1), (1, 0), (2, 0)]


@pytest.mark.skipif(not HAS_TENSEAL, reason="TenSEAL not installed")
def test_legacy_single_selector_request_still_scans_full_width():
    """老调用方手工搭的 request 没有 probed_clusters -> 标签退化为 -1，宽度按 max_size。"""

    artifacts = _artifacts(cluster_sizes=np.array([2, 1, 1], dtype=np.int64))
    first_req, state = prepare_party_a_multi_query(_QUERY, artifacts, cfg=_SCHEMA)
    from multi_attribute.protocol import compare_multi_to_centroids

    request, _ = choose_multi_clusters(
        compare_multi_to_centroids(first_req, artifacts.centroids), state, probes=1
    )
    legacy = type(request)(
        encrypted_match_query=request.encrypted_match_query,
        encrypted_selector=request.encrypted_selector,
    )
    tags = [
        (cluster, column)
        for cluster, column, _ in column_wise_multi_probe_matching(
            artifacts.cluster_matrix, legacy, first_req.public_context_bytes, tau=0.5
        )
    ]
    # 标签 -1 = 分不出是哪个簇；宽度取 max_size=2，而不是 cluster_sizes 里的真实宽度。
    assert tags == [(-1, 0), (-1, 1)]


# ---------------------------------------------------------------------------
# 3. demo 报告里的扫描成本口径
# ---------------------------------------------------------------------------


@pytest.fixture
def demo():
    pytest.importorskip("tenseal")
    import scripts.demo_multi_attribute_dataset as module

    return module


def _rows(*, probe_count: int, positives: int = 2):
    """造最小 rows：positives 条命中 + 1 条空查（库里没有它）。"""

    rows = []
    for index in range(positives):
        rows.append(
            {
                "query_id": f"q-{index}",
                "label": True,
                "expected_record_ids": [f"e-{index}"],
                "top1_record_id": f"e-{index}",
                "expected_score": 0.9,
                "impostor_score": 0.4,
                "similarities": {},
                "encrypted": {
                    "catch": True,
                    "agrees_with_label": True,
                    "selected_cluster": 0,
                    "checked_columns": 10,
                    "probed_clusters": probe_count,
                    "true_match_cluster": 0,
                    "cluster_rank": 0,
                    "cluster_hit": True,
                    "cluster_hit_top1_only": True,
                    "miss_cause": None,
                },
            }
        )
    rows.append(
        {
            "query_id": "q-neg",
            "label": False,
            "expected_record_ids": [],
            "top1_record_id": "e-0",
            "expected_score": None,
            "impostor_score": 0.4,
            "similarities": {},
            "encrypted": {
                "catch": False,
                "agrees_with_label": True,
                "selected_cluster": 0,
                "checked_columns": 400,   # 空查扫满整库
                "probed_clusters": probe_count,
                "miss_cause": None,
            },
        }
    )
    return rows


def test_summary_reports_scan_cost_split_by_hit_and_blank_query(demo):
    summary = demo._summarize(
        _rows(probe_count=4), tau=0.6, encrypted=True, database_size=400
    )
    assert summary["probes_per_query"] == 4
    assert summary["scan_columns_mean_positives"] == 10
    # 空查没有早停点，必然扫满整库 —— 这个数必须单独可见，否则"平均只扫 2%"会被
    # 读成对任意查询都成立。
    assert summary["scan_columns_mean_negatives"] == 400
    assert summary["scan_columns_mean"] == pytest.approx((10 + 10 + 400) / 3)
    assert summary["full_scan_columns"] == 400
    assert summary["scan_columns_max"] == 400


def test_summary_keeps_a_top1_only_anchor_when_probing_saturates(demo):
    """探针数 > 1 时 cluster recall 按构造饱和，必须另留一个探针数动不了的锚点。"""

    summary = demo._summarize(_rows(probe_count=4), tau=0.6, encrypted=True)
    assert summary["cluster_recall"] == 1.0
    assert summary["cluster_recall_top1_only"] == 1.0
    assert summary["cluster_rank_mean"] == 0.0
    assert summary["cluster_rank_max"] == 0


def test_summary_without_database_size_omits_the_fraction(demo):
    summary = demo._summarize(_rows(probe_count=1), tau=0.6, encrypted=True)
    assert "scan_columns_fraction" not in summary
    assert summary["scan_columns_mean"] is not None


def test_percentile_is_nearest_rank(demo):
    values = [float(v) for v in range(1, 101)]
    assert demo._percentile(values, 0.5) == 50
    assert demo._percentile(values, 0.95) == 95
    assert demo._percentile(values, 1.0) == 100
    assert demo._percentile([], 0.5) is None
    assert demo._percentile([7.0], 0.95) == 7.0


def test_report_flags_that_a_single_probe_scan_is_not_reported_as_saturated(demo, capsys):
    """probes=1 时不打"饱和"那段 —— 那一档的 cluster recall 本来就有区分度。"""

    summary = demo._summarize(_rows(probe_count=1), tau=0.6, encrypted=True, database_size=400)
    demo.print_summary(summary)
    out = capsys.readouterr().out
    assert "saturates by construction" not in out
    assert "round-2 probes/query=1" in out


def test_report_explains_why_an_exhaustive_run_reads_100_percent(demo, capsys):
    summary = demo._summarize(_rows(probe_count=4), tau=0.6, encrypted=True, database_size=400)
    demo.print_summary(summary)
    out = capsys.readouterr().out
    assert "saturates by construction" in out
    assert "top-1-only cluster recall" in out
    assert "same cost on queries with NO true match" in out
