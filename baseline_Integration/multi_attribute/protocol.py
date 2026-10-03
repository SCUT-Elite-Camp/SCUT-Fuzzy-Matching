from __future__ import annotations

from dataclasses import dataclass
from random import SystemRandom
from typing import Iterable

import numpy as np
import tenseal as ts
from sklearn.preprocessing import StandardScaler

from ckks.context import create_ckks_context
from ckks.keys import encrypt, serialize_public_context
from ckks.operations import add_plain, dot_ct_ct, dot_ct_pt, matmul_ct_pt
from clustering.kmeans_cosine import run_cosine_kmeans
from config.params import (
    CLUSTER_MATRIX_MAX_BYTES,
    KMEANS_ITERATIONS,
    MULTI_ATTRIBUTE_DECRYPT_EPS,
    RANDOM_MASK_MAX,
    RANDOM_MASK_MIN,
    choose_k,
)

from .encoder import encode_record_vectors
from .schema import AttributeSchema, resolve_schema

_RNG = SystemRandom()


@dataclass
class MultiOfflineArtifacts:
    centroids: np.ndarray
    cluster_matrix: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    cluster_assignments: np.ndarray
    max_size: int
    # 生成该组离线产物时使用的 schema。A 侧据此校验双方布局一致（见
    # prepare_party_a_multi_query）。默认 None 以兼容手工构造的测试产物。
    schema: AttributeSchema | None = None
    # 每个簇的**真实**成员数。cluster_matrix 按 max_size 补零对齐，逐列扫描到
    # max_size 会把大量零行也算一遍 —— 零行减掉 tau 后必然为负，不产生误报，纯粹
    # 是浪费。多簇探测把这个浪费乘以探针数，所以第二轮按真实宽度扫。默认 None 表示
    # 调用方没提供，退化成扫满 max_size（手工构造的测试产物走这条路）。
    cluster_sizes: np.ndarray | None = None

    def width_of(self, cluster_index: int) -> int:
        """该簇要扫多少列。"""

        if self.cluster_sizes is None:
            return int(self.max_size)
        return int(self.cluster_sizes[int(cluster_index)])


@dataclass
class MultiFirstRoundRequest:
    public_context_bytes: bytes
    encrypted_cluster_query: bytes


@dataclass
class MultiPartyAState:
    secret_context: ts.Context
    encrypted_match_query: ts.CKKSVector


@dataclass
class MultiSecondRoundRequest:
    encrypted_match_query: bytes
    encrypted_selector: bytes
    # 多簇探测：每个探针一个 one-hot selector。``encrypted_selector`` 是其中第一个
    # （探针排名最高的那个），保持单探针调用方的字段不变。空元组表示"只有
    # encrypted_selector 这一个探针"。
    encrypted_selectors: tuple[bytes, ...] = ()
    # 与 encrypted_selectors 一一对应，按质心相似度降序 —— 第二轮按这个顺序扫，
    # early-stop 才能最先撞上最可能的那个簇。
    probed_clusters: tuple[int, ...] = ()


@dataclass
class MultiProtocolRun:
    catch: bool
    selected_cluster: int
    checked_columns: int
    first_positive_column: int | None
    artifacts: MultiOfflineArtifacts
    # 本次实际探测的簇（按质心分降序），以及第一个越阈的列落在哪个簇。
    probed_clusters: tuple[int, ...] = ()
    first_positive_cluster: int | None = None


def _build_cluster_matrix(
    match_vectors: np.ndarray,
    assignments: np.ndarray,
    k: int,
) -> tuple[np.ndarray, int, np.ndarray]:
    matrix = np.asarray(match_vectors, dtype=np.float64)
    assignments = np.asarray(assignments, dtype=np.int32)
    if matrix.ndim != 2:
        raise ValueError("match_vectors must be 2-D")
    if assignments.shape != (matrix.shape[0],):
        raise ValueError("cluster assignment count mismatch")

    sizes = np.bincount(assignments, minlength=k)
    max_size = int(sizes.max()) if len(sizes) else 0

    # 多属性宽向量下这块 3-D 矩阵会随 match_dim 线性膨胀，而第二轮的 HE 循环
    # 次数等于 max_size。超限时给出明确指向 k_mode 的报错，而不是 OOM。
    nbytes = k * max_size * matrix.shape[1] * np.dtype(np.float64).itemsize
    if nbytes > CLUSTER_MATRIX_MAX_BYTES:
        raise ValueError(
            f"cluster matrix would need {nbytes / 1024 ** 3:.2f} GiB "
            f"(k={k}, max_size={max_size}, match_dim={matrix.shape[1]}); "
            "reduce k via k_mode or shrink the schema's match_dim"
        )

    out = np.zeros((k, max_size, matrix.shape[1]), dtype=np.float64)
    for cluster_idx in range(k):
        members = matrix[assignments == cluster_idx]
        out[cluster_idx, : len(members), :] = members
    return out, max_size, sizes


def prepare_party_b_multi_offline(
    records_b: Iterable[Any],
    *,
    cfg: Any = None,
    k_mode: str | int = "sqrt",
    random_state: int = 42,
) -> MultiOfflineArtifacts:
    schema = resolve_schema(cfg)
    records = list(records_b)
    if not records:
        raise ValueError("records_b cannot be empty")

    cluster_vectors, match_vectors = encode_record_vectors(records, schema)

    if schema.standardize_cluster:
        scaler = StandardScaler()
        standardized = scaler.fit_transform(cluster_vectors)
        scaler_mean = scaler.mean_.astype(np.float64)
        scaler_scale = scaler.scale_.astype(np.float64)
    else:
        # 不标准化：A 侧的 (v - mean)/scale 退化为恒等变换。
        standardized = cluster_vectors
        scaler_mean = np.zeros(schema.cluster_dim, dtype=np.float64)
        scaler_scale = np.ones(schema.cluster_dim, dtype=np.float64)

    # StandardScaler may produce exact zero scale for a constant feature; sklearn
    # exposes scale_=1 for such features, so A-side division remains safe.
    k = choose_k(len(records), mode=k_mode)
    centroids, assignments = run_cosine_kmeans(
        standardized,
        k=k,
        iterations=KMEANS_ITERATIONS,
        random_state=random_state,
    )
    cluster_matrix, max_size, cluster_sizes = _build_cluster_matrix(
        match_vectors, assignments, centroids.shape[0]
    )

    return MultiOfflineArtifacts(
        centroids=centroids,
        cluster_matrix=cluster_matrix,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        cluster_assignments=assignments,
        max_size=max_size,
        schema=schema,
        cluster_sizes=cluster_sizes,
    )


def prepare_party_a_multi_query(
    query: Any,
    artifacts: MultiOfflineArtifacts,
    *,
    cfg: Any = None,
) -> tuple[MultiFirstRoundRequest, MultiPartyAState]:
    schema = resolve_schema(cfg)

    # 布局校验必须 fail closed：两个 schema 总维度相同但属性顺序/种类不同时，
    # 点积会算出一个「看起来合理」的错值。这是引入可配置 schema 后唯一新增的
    # 静默错误风险，所以宁可报错。
    if artifacts.schema is not None and artifacts.schema != schema:
        raise ValueError(
            "schema mismatch between Party A query and Party B artifacts: "
            f"A fingerprint {schema.fingerprint()} != B fingerprint "
            f"{artifacts.schema.fingerprint()}"
        )

    cluster_vec, match_vec = encode_record_vectors([query], schema)
    cluster_vec = cluster_vec[0]
    match_vec = match_vec[0]

    if artifacts.scaler_mean.shape != cluster_vec.shape:
        raise ValueError(
            f"cluster vector dim {cluster_vec.shape} does not match scaler "
            f"{artifacts.scaler_mean.shape}"
        )
    query_cluster_std = (
        cluster_vec - artifacts.scaler_mean
    ) / artifacts.scaler_scale

    ctx = create_ckks_context()
    ctx.generate_relin_keys()
    enc_cluster = encrypt(query_cluster_std, ctx)
    enc_match = encrypt(match_vec, ctx)
    public_bytes = serialize_public_context(ctx)

    return (
        MultiFirstRoundRequest(
            public_context_bytes=public_bytes,
            encrypted_cluster_query=enc_cluster.serialize(),
        ),
        MultiPartyAState(
            secret_context=ctx,
            encrypted_match_query=enc_match,
        ),
    )


def _load_public_context(public_context: bytes | ts.Context) -> ts.Context:
    if isinstance(public_context, bytes):
        ctx = ts.Context.load(public_context)
    else:
        ctx = public_context
    if ctx.is_private():
        raise ValueError("Party B must receive a public-only CKKS context")
    return ctx


def _load_cipher(
    value: ts.CKKSVector | bytes, context: ts.Context
) -> ts.CKKSVector:
    if isinstance(value, bytes):
        return ts.ckks_vector_from(context, value)
    return value


def compare_multi_to_centroids(
    request: MultiFirstRoundRequest,
    centroids: np.ndarray,
) -> list[bytes]:
    matrix = np.asarray(centroids, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("centroids must be 2-D")
    ctx = _load_public_context(request.public_context_bytes)
    enc_query = _load_cipher(request.encrypted_cluster_query, ctx)
    if enc_query.size() != matrix.shape[1]:
        raise ValueError(
            f"query dim {enc_query.size()} does not match centroid dim {matrix.shape[1]}"
        )
    return [dot_ct_pt(enc_query, centroid).serialize() for centroid in matrix]


def _decrypt_centroid_scores(
    encrypted_scores: list[ts.CKKSVector | bytes],
    state: MultiPartyAState,
) -> np.ndarray:
    if not encrypted_scores:
        raise ValueError("encrypted_scores cannot be empty")
    scores = []
    for item in encrypted_scores:
        if isinstance(item, bytes):
            item = ts.ckks_vector_from(state.secret_context, item)
        plain = np.asarray(item.decrypt(), dtype=np.float64).reshape(-1)
        if plain.size != 1:
            raise ValueError("centroid score must decrypt to one scalar")
        scores.append(float(plain[0]))
    return np.asarray(scores, dtype=np.float64)


EXHAUSTIVE_PROBE_MODES = frozenset({"all", "exhaustive"})


def _resolve_probes(probes: int | str | None, k: int) -> int:
    """把 ``probes`` 归一化成 ``[1, k]`` 内的整数探针数。

    ``None`` / ``"all"`` / ``"exhaustive"`` 表示把全部 k 个簇按分排完 —— 这是唯一
    能把簇召回推到 100% 的设置，代价是查询在库里**确实没有**匹配时第二轮要扫满
    整库（早停救不了它，因为根本没有可停的地方）。
    """

    if probes is None:
        return k
    if isinstance(probes, str):
        token = probes.strip().lower()
        if token in EXHAUSTIVE_PROBE_MODES:
            return k
        if not token.isdigit():
            raise ValueError(
                f"probes must be a positive int, None, or one of "
                f"{sorted(EXHAUSTIVE_PROBE_MODES)}, got {probes!r}"
            )
        probes = int(token)
    value = int(probes)
    if value < 1:
        raise ValueError(f"probes must be >= 1, got {value}")
    return max(1, min(value, k))


def choose_multi_clusters(
    encrypted_scores: list[ts.CKKSVector | bytes],
    state: MultiPartyAState,
    *,
    probes: int | str | None = 1,
) -> tuple[MultiSecondRoundRequest, tuple[int, ...]]:
    """选出要探测的簇，并为每个探针发一个 one-hot selector。

    探针按质心分**降序**排列：真实簇在列表里的位置越靠前，第二轮 early-stop
    越早撞上它，平均扫描列数越少。这一项很值钱 —— FEBRL 全库下真实簇的平均排名
    只有 0.3 左右，穷举探测因此只扫全库的 ~1%，而不是均匀顺序下的 ~50%。

    为什么无条件探前 ``probes`` 个而不是按分值加阈值：实测（FEBRL 5000 库）真实
    匹配所在簇没排到第一时，与最高分的差**不是**几乎打平 —— p50≈0.16、
    p90≈0.41~0.67、p99 最高到 12。任何"分差小于 delta 就多探一个"的策略，要么
    delta 小到救不回召回，要么大到把大部分库都探一遍。所以探针数是个常数旋钮，
    由召回/成本曲线实测决定，不做查询自适应。

    实测（FEBRL 5000 库 / 500 查询，k=100，k-means 种子 7/13/42 平均）：top-1 召回
    0.887、top-8 到 0.988 —— **top-m 永远到不了 1.0**，长尾里有真实簇排在第 8~20 位的
    查询。而穷举探测（``probes=None``）召回 1.0000，平均只扫 72 列（全库 5000 列的
    1.4%），因为绝大多数查询在头两个簇就早停了。也就是说穷举在**两个指标上都优于固定
    top-m**（k=100 时 top-8 要扫 406 列才拿到 0.988），top-m 唯一的用处是给没匹配的
    查询一个硬性成本上限。

    这些列数是**部署口径**：B 把探测到的整簇列都算完。实现里生成器按列提前停止，
    实际算得更少（demo 在 k=99 上实测均值 34 列），两者差的是真匹配所在簇的尾部。

    Returns:
        ``(request, probed_clusters)``，``probed_clusters`` 与
        ``request.encrypted_selectors`` 一一对应。
    """

    scores = _decrypt_centroid_scores(encrypted_scores, state)
    n_probes = _resolve_probes(probes, scores.size)
    order = np.argsort(scores)[::-1]
    probed = tuple(int(c) for c in order[:n_probes])

    selectors = []
    for cluster_idx in probed:
        selector = np.zeros(scores.size, dtype=np.float64)
        selector[cluster_idx] = 1.0
        selectors.append(encrypt(selector, state.secret_context).serialize())

    return (
        MultiSecondRoundRequest(
            encrypted_match_query=state.encrypted_match_query.serialize(),
            encrypted_selector=selectors[0],
            encrypted_selectors=tuple(selectors),
            probed_clusters=probed,
        ),
        probed,
    )


def choose_multi_cluster(
    encrypted_scores: list[ts.CKKSVector | bytes],
    state: MultiPartyAState,
) -> tuple[MultiSecondRoundRequest, int]:
    """单探针便捷入口（等价于 ``choose_multi_clusters(..., probes=1)``）。"""

    request, probed = choose_multi_clusters(encrypted_scores, state, probes=1)
    return request, probed[0]


def column_wise_multi_probe_matching(
    cluster_matrix: np.ndarray,
    request: MultiSecondRoundRequest,
    public_context: bytes | ts.Context,
    *,
    tau: float,
    cluster_sizes: np.ndarray | None = None,
):
    """依次对每个探针簇逐列算出 ``mask * (score - tau)`` 密文。

    Yields:
        ``(cluster_index, column_index, ciphertext_bytes)``。``cluster_index`` 取
        ``request.probed_clusters`` 中的值；手工构造、没填该字段的请求退化为
        ``-1``（分不出是哪个簇，但不影响判定）。

    ``cluster_sizes`` 给出每个簇的真实成员数时，只扫该簇的前 ``size`` 列：
    ``cluster_matrix`` 是按 ``max_size`` 补零对齐的，扫到 ``max_size`` 会把大量
    零行也跑一遍密文-密文点积。零行减 tau 必然为负，不会产生误报，纯粹是浪费
    —— 而多探针正好把这个浪费乘以探针数。
    """

    matrix = np.asarray(cluster_matrix, dtype=np.float64)
    if matrix.ndim != 3:
        raise ValueError("cluster_matrix must be 3-D")
    k, max_size, match_dim = matrix.shape
    ctx = _load_public_context(public_context)
    enc_query = _load_cipher(request.encrypted_match_query, ctx)
    if enc_query.size() != match_dim:
        raise ValueError(
            f"encrypted match query dim {enc_query.size()} != {match_dim}"
        )

    selectors = tuple(request.encrypted_selectors) or (request.encrypted_selector,)
    probed = tuple(request.probed_clusters) or (-1,) * len(selectors)
    if len(probed) != len(selectors):
        raise ValueError(
            f"probed_clusters ({len(probed)}) must align with selectors "
            f"({len(selectors)})"
        )

    for cluster_idx, selector_bytes in zip(probed, selectors):
        enc_selector = _load_cipher(selector_bytes, ctx)
        if enc_selector.size() != k:
            raise ValueError(f"selector dim {enc_selector.size()} != k={k}")
        width = max_size
        if cluster_sizes is not None and cluster_idx >= 0:
            width = int(cluster_sizes[int(cluster_idx)])
        for column_idx in range(width):
            # Every row is one cluster; encrypted one-hot selector privately picks
            # the candidate vector from the selected cluster in this column.
            mask = _RNG.uniform(RANDOM_MASK_MIN, RANDOM_MASK_MAX)
            # 掩码乘在**明文**列和明文阈值上，而不是事后去乘密文：ct-ct 点积之后
            # CKKS 的 scale 已经到 2^80，再乘一次明文会超出模数链并抛
            # "scale out of bounds"。数学上等价于 mask * (score - tau)，也与生产
            # 路径 party_b/online_responder.py 的做法一致。
            # 逐列取所有簇在同一列上的候选行，再用密文 one-hot selector 私有地
            # 选出被探的那一簇 —— B 侧不需要把自己的簇分配明文暴露给 A。
            enc_candidate = matmul_ct_pt(
                enc_selector, mask * matrix[:, column_idx, :]
            )
            enc_score = dot_ct_ct(enc_candidate, enc_query)
            enc_score = add_plain(enc_score, -mask * float(tau))
            yield cluster_idx, column_idx, enc_score.serialize()


def column_wise_multi_matching(
    cluster_matrix: np.ndarray,
    request: MultiSecondRoundRequest,
    public_context: bytes | ts.Context,
    *,
    tau: float,
):
    """单探针便捷入口：只产出密文，不带簇/列标签。"""

    for _, _, item in column_wise_multi_probe_matching(
        cluster_matrix, request, public_context, tau=tau
    ):
        yield item


def decrypt_multi_probe(
    tagged_scores,
    state: MultiPartyAState,
    *,
    eps: float = MULTI_ATTRIBUTE_DECRYPT_EPS,
    early_stop: bool = True,
) -> tuple[bool, int, int | None, int | None]:
    """消费 ``(cluster_index, column_index, ciphertext)`` 流，判定是否命中。

    多探针下 early-stop 跨簇生效：第一个越阈的列一出现就停，后面的探针不再解密
    —— 按质心分降序扫描时，真实簇大概率就在最前面。
    """

    checked = 0
    first_positive = None
    first_cluster = None
    for cluster_idx, column_idx, item in tagged_scores:
        if isinstance(item, bytes):
            item = ts.ckks_vector_from(state.secret_context, item)
        value = float(np.asarray(item.decrypt(), dtype=np.float64).reshape(-1)[0])
        checked += 1
        if value > eps and first_positive is None:
            first_positive = column_idx
            first_cluster = cluster_idx
            if early_stop:
                return True, checked, first_positive, first_cluster
    return first_positive is not None, checked, first_positive, first_cluster


def decrypt_multi_match(
    encrypted_scores,
    state: MultiPartyAState,
    *,
    eps: float = MULTI_ATTRIBUTE_DECRYPT_EPS,
    early_stop: bool = True,
) -> tuple[bool, int, int | None]:
    tagged = ((-1, i, item) for i, item in enumerate(encrypted_scores))
    catch, checked, first_positive, _ = decrypt_multi_probe(
        tagged, state, eps=eps, early_stop=early_stop
    )
    return catch, checked, first_positive


def run_multi_attribute_protocol(
    records_b: Iterable[Any],
    query: Any,
    *,
    cfg: Any = None,
    k_mode: str | int = "sqrt",
    random_state: int = 42,
    early_stop: bool = True,
    tau: float | None = None,
    eps: float | None = None,
    artifacts: MultiOfflineArtifacts | None = None,
    probes: int | str | None = 1,
) -> MultiProtocolRun:
    """Run the complete two-round attribute-based privacy-preserving protocol.

    双轮结构与 V1 完全一致，只是向量宽度由 schema 决定：
    第一轮发 cluster 维密文选簇，第二轮发 match 维密文 + selector 逐列判定。

    Args:
        cfg: ``None`` / ``AttributeSchema`` / ``MultiAttributeConfig`` / mapping。
        tau: 覆盖 schema 的 ``similarity_threshold``。
        eps: 覆盖解密阈值判据的容差。
        artifacts: 复用已经算好的 B 方离线产物。B 方离线阶段要做 k-means 并把整个
            cluster 矩阵加密，是多查询场景下的绝对瓶颈；同一批库上跑 N 条查询时
            应当只做一次。传进来的 artifacts 与 ``cfg`` 的 schema 不符会直接报错。
        probes: 第二轮探测的簇数，按质心分降序取前 ``probes`` 个。``1`` 就是 V1 的
            单簇选择；``None`` / ``"all"`` 表示穷举（簇召回实测 1.0000，均摊成本
            仍只有全库扫描的百分之几）。真实簇没排到质心第一是召回率的主要漏点，
            多探几个是唯一实测有效的补救手段（见 ``choose_multi_clusters``）。
    """

    schema = resolve_schema(cfg)
    if artifacts is None:
        artifacts = prepare_party_b_multi_offline(
            records_b,
            cfg=schema,
            k_mode=k_mode,
            random_state=random_state,
        )
    elif artifacts.schema is not None and artifacts.schema != schema:
        raise ValueError(
            "schema mismatch between cfg and the supplied artifacts: "
            f"cfg fingerprint {schema.fingerprint()} != "
            f"artifacts fingerprint {artifacts.schema.fingerprint()}"
        )
    first_req, state = prepare_party_a_multi_query(query, artifacts, cfg=schema)
    enc_centroid_scores = compare_multi_to_centroids(first_req, artifacts.centroids)
    second_req, probed_clusters = choose_multi_clusters(
        enc_centroid_scores, state, probes=probes
    )
    enc_match_scores = column_wise_multi_probe_matching(
        artifacts.cluster_matrix,
        second_req,
        first_req.public_context_bytes,
        tau=schema.similarity_threshold if tau is None else tau,
        cluster_sizes=artifacts.cluster_sizes,
    )
    decrypt_kwargs = {"early_stop": early_stop}
    if eps is not None:
        decrypt_kwargs["eps"] = eps
    catch, checked, first_positive, first_cluster = decrypt_multi_probe(
        enc_match_scores,
        state,
        **decrypt_kwargs,
    )
    return MultiProtocolRun(
        catch=catch,
        selected_cluster=probed_clusters[0],
        checked_columns=checked,
        first_positive_column=first_positive,
        artifacts=artifacts,
        probed_clusters=probed_clusters,
        first_positive_cluster=first_cluster,
    )
