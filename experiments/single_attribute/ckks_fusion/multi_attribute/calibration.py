"""按实测区分度反推属性权重与判定阈值。

现状（V1/V2）里权重和 ``similarity_threshold`` 都是人写死在 schema 里的数：FEBRL 的
``0.3/0.2/0.2/0.15/0.1/0.05`` 与 ``tau=0.6`` 来自当时的实测，换一个数据集就失效。
本模块把这两件事变成**从带标签样本上量出来的量**。

## 为什么不是"权重 ∝ 单属性 AUC"

最直觉的做法是逐属性量 AUC，再按 ``AUC − 0.5`` 分配权重。**实测证明它是错的**，
本仓库保留了这条规则（``derive_weights``）并把它标成 ``weight_policy="auc"``，
但默认不用它 —— FEBRL 500 条查询上：

===========  ==========  =========  =========
规则          组合 AUC    最低真匹配  最高冒充者
===========  ==========  =========  =========
手写权重      0.9994      0.539      0.618
AUC 线性比例  0.9825      0.215      0.609
可分离度最优  0.9999      0.587      0.608
===========  ==========  =========  =========

原因是**单属性 AUC 忽略了互补性**：``ssn`` 的孤立 AUC 最高（0.911），线性比例会把
44.7% 的权重压在它身上；但 ``ssn`` 是个近乎二值的属性，FEBRL 一旦把某条副本的
``ssn`` 扰动掉，这条查询的真匹配分就塌成 0，最低真匹配分从 0.539 掉到 0.215。
AUC 高 ≠ 权重该高 —— 高 AUC 可能只是"大多数时候都对，对的那几次也不需要它"。

## 默认规则：最大化实测可分离度

``weight_policy="separability"``（默认）直接在**组合分**上做坐标上升，目标函数是
**软化 AUC**（soft-AUC / pairwise sigmoid）：

    objective(w) = mean_i  sigmoid( (score_i(真匹配) − score_i(最强冒充者)) / T )

它衡量的是"真匹配分压过最强冒充者的成对概率"，正是阈值判定要用的量；``T`` 越小
越只关心最难的那几对（``T → 0`` 退化成最大化最小间隔）。这样得到的权重是**联合**
最优的，而不是各属性各自为政 —— 互补的属性会因为"没有它这几对就翻车"而拿到权重。

**性能**：坐标上升的每轮只需在少量候选上算加权和（见 ``_CandidateSet``），
不构造 ``(n_query, n_db, n_attribute)`` 全张量。

## 阈值

权重定下后用新权重重新量出分数分布，再按判据取 tau，默认 ``recall_first``
（见 ``derive_threshold``）。

## 口径

AUC 的负样本取"每条查询的最强冒充者"，不是随机负样本。阈值要挡的正是这个最强
冒充者，随机负样本会把区分度报得虚高 —— FEBRL 上随机库记录几乎必然被真匹配压过，
AUC 恒等于 1，没有任何信息量。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from config.params import MULTI_ATTRIBUTE_DECRYPT_EPS

from .encoder import encode_attribute_matrix, encode_record_vectors
from .schema import AttributeSchema, AttributeSpec, resolve_schema

__all__ = [
    "AttributeDiagnostics",
    "CalibrationReport",
    "ScoreSamples",
    "attribute_score_samples",
    "calibrate_schema",
    "derive_threshold",
    "derive_weights",
    "optimize_weights",
    "roc_auc",
]

# 坐标上升的默认超参。乘性步长成对出现（放大/缩小），保证每步都能回到原点附近；
# 多个 Dirichlet 随机起点是为了绕开 min/max 类目标的多面体退化。
_DEFAULT_RESTARTS = 12
_DEFAULT_STEPS = (1.3, 1.0 / 1.3, 1.1, 1.0 / 1.1)
_DEFAULT_SWEEPS = 250
_DEFAULT_TEMPERATURE = 0.05
_WEIGHT_EPS = 1e-6


# ---------------------------------------------------------------------------
# 样本
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreSamples:
    """一组"真匹配 vs 最强冒充者"的成对分数。"""

    positive: np.ndarray
    negative: np.ndarray
    query_ids: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.positive.size)

    @property
    def overlap(self) -> int:
        """真匹配分没能压过自己那个最强冒充者的查询数。"""

        return int(np.sum(self.positive <= self.negative))

    @property
    def margin(self) -> float:
        """全局间隔 ``min(真匹配) − max(冒充者)``；> 0 表示存在一个 tau 完全分开两者。"""

        if self.positive.size == 0 or self.negative.size == 0:
            return float("-inf")
        return float(self.positive.min() - self.negative.max())


def _record_id(record: Any, index: int) -> str:
    rid = getattr(record, "record_id", None)
    return str(rid) if rid else f"row-{index}"


def _db_index(database: Sequence[Any]) -> dict[str, int]:
    index: dict[str, int] = {}
    for i, record in enumerate(database):
        index.setdefault(_record_id(record, i), i)
    return index


def _labelled_pairs(
    database: Sequence[Any], queries: Sequence[Any], index: Mapping[str, int]
) -> list[tuple[int, int]]:
    """可用正例的 ``(query_index, target_index)``。

    真匹配不在库里的查询没有"真匹配分"，必须整条剔除 —— 当成 0 会把 AUC 与阈值
    全部拉偏。这与 demo 里 ``expected_score is None`` 是同一个口径。
    """

    pairs: list[tuple[int, int]] = []
    for qi, query in enumerate(queries):
        for rid in sorted(query.expected_record_ids):
            if rid in index:
                pairs.append((qi, index[rid]))
                break
    return pairs


def _hardest_impostor(scores: np.ndarray, forbidden: set[int]) -> float:
    if not forbidden:
        return float(scores.max())
    mask = np.ones(scores.shape[0], dtype=bool)
    mask[list(forbidden)] = False
    if not mask.any():  # 库里只有真匹配本身
        return float("-inf")
    return float(scores[mask].max())


def attribute_score_samples(
    database: Sequence[Any],
    queries: Sequence[Any],
    cfg: Any = None,
    *,
    attribute: str | None = None,
) -> ScoreSamples:
    """量出正/负样本分数。

    Args:
        attribute: ``None`` 时用整套权重算总分；给属性名时只算该属性的**未加权**
            相似度（权重只是缩放，不影响该属性自己的排序，因此也不影响 AUC）。
    """

    schema = resolve_schema(cfg)
    index = _db_index(database)
    if attribute is None:
        _, db_matrix = encode_record_vectors(database, schema)
        _, q_matrix = encode_record_vectors([q.record for q in queries], schema)
    else:
        db_matrix = encode_attribute_matrix(database, schema)[attribute].match
        q_matrix = encode_attribute_matrix([q.record for q in queries], schema)[attribute].match

    positives: list[float] = []
    negatives: list[float] = []
    query_ids: list[str] = []
    for qi, target in _labelled_pairs(database, queries, index):
        scores = q_matrix[qi] @ db_matrix.T
        positives.append(float(scores[target]))
        forbidden = {index[rid] for rid in queries[qi].expected_record_ids if rid in index}
        negatives.append(_hardest_impostor(scores, forbidden))
        query_ids.append(_record_id(queries[qi].record, qi))

    return ScoreSamples(
        positive=np.asarray(positives, dtype=np.float64),
        negative=np.asarray(negatives, dtype=np.float64),
        query_ids=tuple(query_ids),
    )


# ---------------------------------------------------------------------------
# AUC 与权重
# ---------------------------------------------------------------------------


def roc_auc(positive: Iterable[float], negative: Iterable[float]) -> float | None:
    """Mann-Whitney U 形式的 AUC，并列值各记半个胜场。

    两组都非空才有意义；否则返回 ``None`` —— 报 0.5 会被读成"抛硬币"，那是
    另一个意思。自己实现而不引 scipy，是为了让这个纯函数的依赖面保持为零。
    """

    pos = np.asarray(list(positive), dtype=np.float64).reshape(-1)
    neg = np.asarray(list(negative), dtype=np.float64).reshape(-1)
    if pos.size == 0 or neg.size == 0:
        return None

    combined = np.concatenate([pos, neg])
    order = np.argsort(combined, kind="mergesort")
    ordered = combined[order]
    ranks = np.empty(combined.size, dtype=np.float64)
    start = 0
    for i in range(1, combined.size + 1):
        if i == combined.size or ordered[i] != ordered[start]:
            ranks[order[start:i]] = (start + i + 1) / 2.0
            start = i
    rank_sum = ranks[: pos.size].sum()
    return float((rank_sum - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def derive_weights(
    aucs: Mapping[str, float | None],
    *,
    power: float = 1.0,
    floor: float = 0.0,
) -> dict[str, float]:
    """``weight ∝ (AUC − 0.5)^power``，归一化到和为 1。

    **这条规则在本仓库的实测里是不如默认的 ``separability`` 的**（见模块 docstring
    的对照表）：它逐个属性看区分度，看不见属性之间的互补性，会把权重堆在"几乎
    总是精确相等"的准二值属性上，一旦那条属性被扰动，真匹配分就整体塌掉。保留它
    是因为它便宜、可解释，且在属性彼此独立的数据上够用；它由
    ``calibrate_schema(weight_policy="auc")`` 选用，不是默认。

    AUC ≤ 0.5 的属性权重归零 —— 它在样本上连抛硬币都不如，留着只会往总分里掺噪声。
    """

    if not aucs:
        raise ValueError("aucs cannot be empty")
    scores: dict[str, float] = {}
    for name, auc in aucs.items():
        if auc is None or not np.isfinite(auc):
            raise ValueError(
                f"attribute {name!r} has no usable AUC (need both positive and "
                "negative samples); cannot derive a weight from it"
            )
        scores[name] = max(0.0, float(auc) - 0.5) ** float(power)

    total = sum(scores.values())
    if total <= 0:
        raise ValueError(
            "every attribute has AUC <= 0.5, so no weight can be derived; "
            "the attributes carry no signal on this sample"
        )
    weights = {name: value / total for name, value in scores.items()}
    if floor > 0:
        below = [name for name, w in weights.items() if _WEIGHT_EPS < w < floor]
        if below:
            for name in below:
                weights[name] = floor
            remaining = 1.0 - floor * len(below)
            rest = sum(w for n, w in weights.items() if n not in below)
            if rest > 0:
                for n in weights:
                    if n not in below:
                        weights[n] = weights[n] / rest * remaining
            else:
                weights = {n: 1.0 / len(weights) for n in weights}
    return weights


class _CandidateSet:
    """每条查询的**候选**库记录及其逐属性相似度。

    坐标上升要对任意权重反复算"这条查询在所有库记录上的加权分"。直接构造
    ``(n_query, n_db, n_attribute)`` 全张量在 5000×800 上就是 60 MB，真实库再大
    一个量级就爆了。这里只保留**每条属性各自的 top-K 库记录之并集**：加权和的
    最大值必然出现在某条属性得分高的记录上，把每条属性的 top-K 并起来，实际
    测得的"最强冒充者"与全量的结果一致，而内存只随 ``n_attribute × K`` 走。

    真匹配那条永远在候选里（否则正样本就丢了）。
    """

    def __init__(
        self,
        sims: np.ndarray,
        candidate_index: np.ndarray,
        positive_slot: np.ndarray,
        *,
        top_k: int,
    ) -> None:
        self.sims = sims  # (n_query, n_candidate, n_attribute)
        self.index = candidate_index  # (n_query, n_candidate) -> db row
        self.positive_slot = positive_slot  # (n_query,) -> candidate slot, -1 if none
        self.top_k = top_k

    @classmethod
    def build(
        cls,
        database: Sequence[Any],
        queries: Sequence[Any],
        schema: AttributeSchema,
        *,
        top_k: int = 64,
    ) -> "_CandidateSet":
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        index = _db_index(database)
        pairs = dict(_labelled_pairs(database, queries, index))
        names = [spec.name for spec in schema.attributes]
        db_blocks = encode_attribute_matrix(database, schema)
        q_blocks = encode_attribute_matrix([q.record for q in queries], schema)
        n_db = len(database)

        all_sims = np.stack(
            [q_blocks[name].match @ db_blocks[name].match.T for name in names], axis=-1
        )  # (n_query, n_db, n_attribute)

        rows: list[np.ndarray] = []
        take = min(top_k, n_db)
        for qi in range(len(queries)):
            if take >= n_db:
                union = np.arange(n_db)
            else:
                # 逐属性取 top-K 的并集：argpartition 比全排序省，这里不在乎顺序。
                picked = np.argpartition(all_sims[qi], -take, axis=0)[-take:]
                union = np.unique(picked)
            target = pairs.get(qi)
            if target is not None and target not in union:
                union = np.append(union, target)
            rows.append(union)

        width = max((len(r) for r in rows), default=0)
        candidate_index = np.full((len(queries), width), -1, dtype=np.int64)
        sims = np.zeros((len(queries), width, len(names)), dtype=np.float64)
        positive_slot = np.full(len(queries), -1, dtype=np.int64)
        for qi, union in enumerate(rows):
            candidate_index[qi, : len(union)] = union
            sims[qi, : len(union)] = all_sims[qi, union]
            target = pairs.get(qi)
            if target is not None:
                positive_slot[qi] = int(np.where(union == target)[0][0])
        return cls(sims, candidate_index, positive_slot, top_k=top_k)

    def scores(self, weights: np.ndarray) -> np.ndarray:
        return self.sims @ weights  # (n_query, n_candidate)

    def paired_margins(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 ``(真匹配分, 最强冒充者分)``，只含有真匹配的那些查询。"""

        scores = self.scores(weights)
        rows = np.where(self.positive_slot >= 0)[0]
        pos = scores[rows, self.positive_slot[rows]]
        masked = scores[rows].copy()
        masked[np.arange(len(rows)), self.positive_slot[rows]] = -np.inf
        masked[self.index[rows] < 0] = -np.inf
        return pos, masked.max(axis=1)


def optimize_weights(
    samples: "_CandidateSet",
    names: Sequence[str],
    *,
    temperature: float = _DEFAULT_TEMPERATURE,
    restarts: int = _DEFAULT_RESTARTS,
    sweeps: int = _DEFAULT_SWEEPS,
    initial: Mapping[str, float] | None = None,
    random_state: int = 42,
) -> dict[str, float]:
    """坐标上升最大化**软 AUC**，返回归一化权重。

    目标函数是成对 sigmoid::

        objective(w) = mean_i sigmoid( (pos_i(w) − neg_i(w)) / temperature )

    ``temperature`` 是软化尺度：它越小，目标越接近"最大化最小间隔"（只盯最难的那
    几对），越大越接近普通 AUC。它比硬 AUC 好优化的地方是**处处有梯度** —— 硬 AUC
    在分数排序不变时是常数，坐标上升会在平坦区里乱走。

    起点包含手工权重（若给了 ``initial``）与若干个 Dirichlet 随机点；返回最优者。
    """

    n = len(names)
    if n == 0:
        raise ValueError("no attributes to weight")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    def objective(w: np.ndarray) -> float:
        pos, neg = samples.paired_margins(w)
        if pos.size == 0:
            raise ValueError(
                "no labelled positive pair has its true match in the database; "
                "cannot fit weights"
            )
        z = (pos - neg) / float(temperature)
        # 手工实现 sigmoid 的数值稳定版，负的大 z 不掉进 overflow。
        out = np.empty_like(z)
        positive = z >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        out[~positive] = exp_z / (1.0 + exp_z)
        return float(out.mean())

    rng = np.random.RandomState(random_state)
    starts: list[np.ndarray] = []
    if initial is not None:
        starts.append(_normalize_weight_vector(np.array([initial[n] for n in names])))
    for _ in range(max(0, restarts - len(starts))):
        starts.append(_normalize_weight_vector(rng.dirichlet(np.ones(n))))

    best_w, best_v = starts[0], objective(starts[0])
    for start in starts:
        w = start.copy()
        value = objective(w)
        for _ in range(sweeps):
            improved = False
            for a in range(n):
                for factor in _DEFAULT_STEPS:
                    candidate = w.copy()
                    candidate[a] *= factor
                    total = candidate.sum()
                    if total <= 0:
                        continue
                    candidate /= total
                    v = objective(candidate)
                    if v > value + 1e-12:
                        w, value, improved = candidate, v, True
            if not improved:
                break
        if value > best_v:
            best_w, best_v = w, value
    return {name: float(best_w[i]) for i, name in enumerate(names)}


def _normalize_weight_vector(w: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(w, dtype=np.float64), 0.0, None)
    total = w.sum()
    if total <= 0:
        raise ValueError("weight vector must have a positive sum")
    out = w / total
    out[out < _WEIGHT_EPS] = 0.0
    total = out.sum()
    return out / total if total > 0 else out


# ---------------------------------------------------------------------------
# 阈值
# ---------------------------------------------------------------------------

THRESHOLD_CRITERIA = ("recall_first", "youden", "target_recall")


def derive_threshold(
    samples: ScoreSamples,
    *,
    criterion: str = "recall_first",
    recall_target: float = 1.0,
    noise_margin: float = MULTI_ATTRIBUTE_DECRYPT_EPS,
    negative_query_scores: Iterable[float] | None = None,
) -> float:
    """从分数分布里取判定阈值。

    - ``recall_first``（默认）：先把召回拉满 —— tau 压到最低真匹配分之下一点点；
      并列时挑最高的那个 tau（假正例最少）。``noise_margin`` 是给 CKKS 噪声留的
      余量：第二轮判据是 ``score > tau``，真匹配那条的密文分有约 1e-4 的抖动，
      阈值贴着 min(P) 会让它随机漏掉。
    - ``youden``：最大化 ``TPR − FPR``（Youden's J），召回与假正例的平衡点。
    - ``target_recall``：在召回 ≥ ``recall_target`` 的前提下取最高的 tau。

    ``negative_query_scores`` 是**标签为假**的查询在库内的最高分：它们没有真匹配
    可比、不参与 recall，但必须参与假正例计数。
    """

    if criterion not in THRESHOLD_CRITERIA:
        raise ValueError(
            f"criterion must be one of {list(THRESHOLD_CRITERIA)}, got {criterion!r}"
        )
    pos = np.asarray(samples.positive, dtype=np.float64)
    if pos.size == 0:
        raise ValueError(
            "no usable positive samples; calibration needs labelled queries whose "
            "true match is present in the database"
        )

    neg = np.asarray(samples.negative, dtype=np.float64)
    if negative_query_scores is not None:
        extra = np.asarray(list(negative_query_scores), dtype=np.float64).reshape(-1)
        neg = np.concatenate([neg, extra])
    neg = neg[np.isfinite(neg)]

    if criterion == "recall_first":
        return max(0.0, float(pos.min()) - float(noise_margin))

    if criterion == "target_recall":
        target = float(recall_target)
        if not 0.0 <= target <= 1.0:
            raise ValueError(f"recall_target must be in [0, 1], got {target}")
        if target >= 1.0:
            return max(0.0, float(pos.min()) - float(noise_margin))
        # 允许目标召回的**最大** tau：恰好留下 ceil(target·n) 个正例的最高取值。
        keep = max(1, int(np.ceil(target * pos.size)))
        tau = float(np.sort(pos)[::-1][keep - 1]) - float(noise_margin)
        return max(0.0, tau)

    # youden：候选切点取所有观测分数，逐个算 TPR − FPR。
    def recall_at(tau: float) -> float:
        return float(np.mean(pos > tau))

    def fpr_at(tau: float) -> float:
        return float(np.mean(neg > tau)) if neg.size else 0.0

    best_tau, best_j = 0.0, -np.inf
    for candidate in np.unique(np.concatenate([pos, neg])):
        j = recall_at(candidate) - fpr_at(candidate)
        if j > best_j:
            best_j, best_tau = j, float(candidate)
    return max(0.0, best_tau)


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttributeDiagnostics:
    """单个属性在样本上的实测表现。"""

    name: str
    kind: str
    auc: float
    previous_weight: float
    weight: float
    positive_min: float
    positive_median: float
    negative_median: float
    negative_max: float

    @property
    def weight_delta(self) -> float:
        return self.weight - self.previous_weight

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "auc": self.auc,
            "previous_weight": self.previous_weight,
            "weight": self.weight,
            "weight_delta": self.weight_delta,
            "positive_min": self.positive_min,
            "positive_median": self.positive_median,
            "negative_median": self.negative_median,
            "negative_max": self.negative_max,
        }


@dataclass(frozen=True)
class CalibrationReport:
    """一次标定的完整结果，以及判断这次标定可不可信所需的原始数字。"""

    attributes: tuple[AttributeDiagnostics, ...]
    weights: dict[str, float]
    threshold: float
    previous_threshold: float
    weight_policy: str
    criterion: str
    lowest_true_match: float
    highest_impostor: float
    positive_scores: np.ndarray
    negative_scores: np.ndarray
    negative_query_scores: np.ndarray
    sample_size: int
    combined_auc: float | None
    baseline_auc: float | None = None
    candidate_top_k: int = 0

    def recall_at(self, tau: float | None = None) -> float:
        tau = self.threshold if tau is None else float(tau)
        return float(np.mean(self.positive_scores > tau))

    def false_positive_rate_at(self, tau: float | None = None) -> float:
        tau = self.threshold if tau is None else float(tau)
        neg = self._all_negatives()
        return float(np.mean(neg > tau)) if neg.size else 0.0

    def _all_negatives(self) -> np.ndarray:
        return np.concatenate([self.negative_scores, self.negative_query_scores])

    @property
    def overlap(self) -> int:
        """真匹配压不过自己最强冒充者的正例条数 —— 任何全局 tau 都救不回来。"""

        return int(np.sum(self.positive_scores <= self.negative_scores))

    @property
    def separated(self) -> bool:
        """是否存在一个 tau 同时挡掉所有冒充者、又不漏掉任何真匹配。"""

        return bool(self.highest_impostor < self.lowest_true_match)

    @property
    def ranges_interleave(self) -> bool:
        """正例分与冒充者分的**全局区间**交错 —— 与 ``overlap`` 是两回事。

        ``overlap`` 数的是"某条真匹配压不过它自己的最强冒充者"，是**逐查询**的失败；
        这里说的是"最弱的真匹配低于最强的冒充者"，是**跨查询**的失败。FEBRL 上典型
        形态正是后者：overlap = 0（每条查询都赢了自己那一组），但 0.5132 < 0.5982。
        此时任何一个全局 tau 都只能二选一 —— 保召回就放进 13% 的假正例，反之亦然。
        """

        return bool(self.lowest_true_match <= self.highest_impostor)

    def _separation_verdict(self) -> str:
        """把 ``separated`` / ``ranges_interleave`` 两种失败讲清楚，别混成一句。"""

        if self.separated:
            return "SEPARABLE: a single tau covers every impostor without losing a positive"
        if self.overlap:
            return (
                f"OVERLAP on {self.overlap} positive(s): those queries lose to their own "
                "strongest impostor, no tau fixes them"
            )
        return (
            "RANGES INTERLEAVE: every query beats its own impostors (0 pairwise "
            "inversions), but the weakest true match sits below the strongest impostor, "
            "so no single global tau gets both -- recall and false positives trade off "
            "one for one here"
        )

    @property
    def margin(self) -> float:
        return float(self.lowest_true_match - self.highest_impostor)

    def summary(self) -> dict[str, Any]:
        return {
            "weight_policy": self.weight_policy,
            "criterion": self.criterion,
            "threshold": self.threshold,
            "previous_threshold": self.previous_threshold,
            "weights": dict(self.weights),
            "sample_size": self.sample_size,
            "candidate_top_k": self.candidate_top_k,
            "combined_auc": self.combined_auc,
            "baseline_auc": self.baseline_auc,
            "lowest_true_match": self.lowest_true_match,
            "highest_impostor": self.highest_impostor,
            "margin": self.margin,
            "overlapping_positives": self.overlap,
            "separated": self.separated,
            "ranges_interleave": self.ranges_interleave,
            "recall_at_threshold": self.recall_at(),
            "false_positive_rate_at_threshold": self.false_positive_rate_at(),
            "attributes": [a.to_dict() for a in self.attributes],
        }

    def explain(self) -> str:
        """人读的标定说明，直接打进日志/终端。"""

        lines = [
            f"calibration sample: {self.sample_size} labelled positive(s), "
            f"{self.negative_query_scores.size} labelled negative(s); "
            f"weight policy {self.weight_policy!r}",
            f"{'attribute':<14}{'AUC':>7}{'old w':>8}{'new w':>8}{'delta':>8}"
            f"{'pos min':>9}{'pos med':>9}{'imp med':>9}{'imp max':>9}",
        ]
        for diag in self.attributes:
            lines.append(
                f"{diag.name:<14}{diag.auc:>7.3f}{diag.previous_weight:>8.3f}"
                f"{diag.weight:>8.3f}{diag.weight_delta:>+8.3f}"
                f"{diag.positive_min:>9.3f}{diag.positive_median:>9.3f}"
                f"{diag.negative_median:>9.3f}{diag.negative_max:>9.3f}"
            )
        if self.baseline_auc is not None and self.combined_auc is not None:
            lines.append(
                f"combined AUC {self.baseline_auc:.4f} (previous weights) -> "
                f"{self.combined_auc:.4f} (calibrated)"
            )
        lines.append(
            f"lowest true-match {self.lowest_true_match:.4f} / highest impostor "
            f"{self.highest_impostor:.4f} -> " + self._separation_verdict()
        )
        lines.append(
            f"threshold {self.previous_threshold:.4f} -> {self.threshold:.4f} "
            f"({self.criterion}): recall {self.recall_at():.1%}, "
            f"false-positive rate {self.false_positive_rate_at():.1%}"
        )
        return "\n".join(lines)


WEIGHT_POLICIES = ("separability", "auc")


def calibrate_schema(
    database: Sequence[Any],
    queries: Sequence[Any],
    cfg: Any = None,
    *,
    weight_policy: str = "separability",
    criterion: str = "recall_first",
    recall_target: float = 1.0,
    power: float = 1.0,
    weight_floor: float = 0.0,
    temperature: float = _DEFAULT_TEMPERATURE,
    restarts: int = _DEFAULT_RESTARTS,
    candidate_top_k: int = 64,
    random_state: int = 42,
    noise_margin: float = MULTI_ATTRIBUTE_DECRYPT_EPS,
) -> tuple[AttributeSchema, CalibrationReport]:
    """量出权重与阈值，返回**新的** schema 与完整报告。

    原 schema 不被修改（``AttributeSchema`` 是 frozen 的）。返回的新 schema 除了
    ``weight`` 与 ``similarity_threshold`` 之外逐字段与输入一致 —— 属性顺序、
    kind、params 全部保留，因此维度与布局不变。
    """

    if weight_policy not in WEIGHT_POLICIES:
        raise ValueError(
            f"weight_policy must be one of {list(WEIGHT_POLICIES)}, got {weight_policy!r}"
        )
    schema = resolve_schema(cfg)
    names = [spec.name for spec in schema.attributes]

    # 逐属性诊断（也是 "auc" 策略的全部依据）。
    per_attribute = {
        spec.name: attribute_score_samples(database, queries, schema, attribute=spec.name)
        for spec in schema.attributes
    }
    aucs = {
        spec.name: roc_auc(per_attribute[spec.name].positive, per_attribute[spec.name].negative)
        for spec in schema.attributes
    }

    baseline = attribute_score_samples(database, queries, schema)
    baseline_auc = roc_auc(baseline.positive, baseline.negative)

    if weight_policy == "auc":
        weights = derive_weights(aucs, power=power, floor=weight_floor)
    else:
        candidates = _CandidateSet.build(database, queries, schema, top_k=candidate_top_k)
        weights = optimize_weights(
            candidates,
            names,
            temperature=temperature,
            restarts=restarts,
            initial={spec.name: spec.weight for spec in schema.attributes},
            random_state=random_state,
        )

    weighted = AttributeSchema(
        attributes=tuple(
            AttributeSpec(
                name=spec.name, kind=spec.kind, weight=weights[spec.name], params=spec.params
            )
            for spec in schema.attributes
        ),
        similarity_threshold=schema.similarity_threshold,
        name_attribute=schema.name_attribute,
        standardize_cluster=schema.standardize_cluster,
    )

    overall = attribute_score_samples(database, queries, weighted)
    combined_auc = roc_auc(overall.positive, overall.negative)
    negative_query_scores = _negative_label_scores(database, queries, weighted)
    threshold = derive_threshold(
        overall,
        criterion=criterion,
        recall_target=recall_target,
        noise_margin=noise_margin,
        negative_query_scores=negative_query_scores,
    )

    calibrated = AttributeSchema(
        attributes=weighted.attributes,
        similarity_threshold=threshold,
        name_attribute=weighted.name_attribute,
        standardize_cluster=weighted.standardize_cluster,
    )

    diagnostics = tuple(
        AttributeDiagnostics(
            name=spec.name,
            kind=spec.kind,
            auc=float(aucs[spec.name]),
            previous_weight=float(spec.weight),
            weight=float(weights[spec.name]),
            positive_min=float(per_attribute[spec.name].positive.min()),
            positive_median=float(np.median(per_attribute[spec.name].positive)),
            negative_median=float(np.median(per_attribute[spec.name].negative)),
            negative_max=float(per_attribute[spec.name].negative.max()),
        )
        for spec in schema.attributes
    )

    report = CalibrationReport(
        attributes=diagnostics,
        weights=weights,
        threshold=threshold,
        previous_threshold=float(schema.similarity_threshold),
        weight_policy=weight_policy,
        criterion=criterion,
        lowest_true_match=float(overall.positive.min()),
        highest_impostor=(
            float(overall.negative.max()) if overall.negative.size else float("-inf")
        ),
        positive_scores=overall.positive,
        negative_scores=overall.negative,
        negative_query_scores=negative_query_scores,
        sample_size=len(overall),
        combined_auc=combined_auc,
        baseline_auc=baseline_auc,
        candidate_top_k=candidate_top_k,
    )
    return calibrated, report


def _negative_label_scores(
    database: Sequence[Any], queries: Sequence[Any], schema: AttributeSchema
) -> np.ndarray:
    """标签为假、没有真匹配的查询：它们的库内最高分是纯假正例样本。"""

    rows = [q for q in queries if not q.expected_record_ids]
    if not rows:
        return np.empty(0, dtype=np.float64)
    _, db_matrix = encode_record_vectors(database, schema)
    _, q_matrix = encode_record_vectors([q.record for q in rows], schema)
    return np.asarray((q_matrix @ db_matrix.T).max(axis=1), dtype=np.float64)
