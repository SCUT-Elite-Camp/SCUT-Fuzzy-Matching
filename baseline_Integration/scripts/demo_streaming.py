#!/usr/bin/env python3
"""流式终端演示脚本 v2：按步骤聚合展示所有查询，精简关键信息。

设计目标：
1. 按步骤聚合：Step 1 一次性展示所有查询的聚类结果，Step 2 一次性展示所有查询的加密结果，以此类推
2. 精简输出：只展示关键信息，去掉冗余细节
3. 横向对比：每个步骤可以看到所有查询的处理情况
4. 高度可复用：支持更换数据库（names_b）和查询内容（names_a/query）

使用方式：
    cd baseline_Integration
    python scripts/demo_streaming_v2.py
    python scripts/demo_streaming_v2.py --query "John Smith" --db-limit 50
    python scripts/demo_streaming_v2.py --query-indices 2,7,26 --db-limit 100 --k 10
"""

from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import time
from dataclasses import dataclass, field
from random import SystemRandom
from typing import Any

import numpy as np

from protocol.types import (
    OfflineArtifacts,
    FirstRoundRequest,
    PartyALocalState,
    SecondRoundRequest,
)
from preprocessing.text_cleaner import clean_name
from preprocessing.normalizer import l2_normalize
from party_b.offline_prep import fit_scaler, build_cluster_matrix
from party_a.online_querier import decrypt_sim_scores, build_selector, encrypt_selector
from party_a.local_prep import encode_query_vectors
from minhash.encoder import batch_encode, _generate_shingles, _hash_shingle
from evaluation.dataset_loader import load_dataset
from clustering.kmeans_cosine import run_cosine_kmeans
from ckks.operations import dot_ct_pt, dot_ct_ct, matmul_ct_pt, add_plain
from ckks.keys import encrypt, serialize_public_context
from ckks.context import create_ckks_context
from config.params import (
    NUM_PERMUTATIONS_CLUSTER,
    NUM_PERMUTATIONS_MATCH,
    SIMILARITY_THRESHOLD,
    choose_k,
    RANDOM_MASK_MIN,
    RANDOM_MASK_MAX,
)


_RNG = SystemRandom()


def pca_project_3d(
    data: np.ndarray,
    cluster_assignments: np.ndarray | None = None,
    k: int | None = None,
    max_points: int = 300,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """把高维签名投影到 3D，用于可视化点云。

    为了让同簇的点在视觉上明显聚拢，投影基由**簇质心**的 PCA 决定（判别式投影，
    类似 LDA 的思路：优先保留簇间差异方向），而不是全体数据的 PCA。这样投影后
    同簇点会比普通 PCA 更紧凑、不同簇更分开。无簇信息时回退到普通 PCA。

    返回 (coords_3d, sample_idx)：
        coords_3d  — shape (m, 3)，已归一化到约 [-1, 1]
        sample_idx — shape (m,)，被选中样本在原数组中的行索引（对齐簇标签）
    """
    n = data.shape[0]
    if n > max_points:
        rng = np.random.default_rng(random_state)
        sample_idx = np.sort(rng.choice(n, size=max_points, replace=False))
    else:
        sample_idx = np.arange(n)

    x_full = data.astype(np.float64)
    mean = x_full.mean(axis=0, keepdims=True)
    x = x_full[sample_idx] - mean

    basis = None
    if cluster_assignments is not None and k is not None and k >= 3:
        # 计算每个簇质心（在全体数据上），对质心做 PCA 取投影基。
        centers = []
        for c in range(k):
            members = x_full[cluster_assignments == c]
            if len(members) > 0:
                centers.append(members.mean(axis=0))
        centers = np.asarray(centers)
        if len(centers) >= 3:
            cc = centers - centers.mean(axis=0, keepdims=True)
            cov_c = np.cov(cc, rowvar=False)
            _, vecs = np.linalg.eigh(cov_c)
            basis = vecs[:, -3:][:, ::-1]

    if basis is None:
        # 回退：全体数据 PCA
        cov = np.cov(x, rowvar=False)
        _, eigvecs = np.linalg.eigh(cov)
        basis = eigvecs[:, -3:][:, ::-1]

    coords = x @ basis

    # 归一化到 [-1, 1]，零方差维度退化为 0，避免除零。
    span = np.abs(coords).max(axis=0)
    span[span < 1e-12] = 1.0
    coords = coords / span
    return coords, sample_idx


def is_query_correct(q: "QueryState") -> bool | None:
    """判定单个查询的预测是否正确。

    判定准则（区别于"是否命中"）：
    - 期望命中某姓名（expected_hit_name 非空，含正例与 fuzzy 拼错查询）：
      必须命中，且命中的库内姓名等于期望姓名，才算正确。
      仅命中但命中了错误姓名 -> 错误。
    - 期望不命中（库外反例，expected_label 为 False）：未命中即正确。
    - 无标签（自定义查询）：返回 None（不计入准确率）。
    """
    if q.expected_hit_name is not None:
        return q.match_found and q.first_hit_name == q.expected_hit_name
    if q.expected_label is False:
        return not q.match_found
    return None


# =============================================================================
# 终端可视化工具
# =============================================================================

class Colors:
    """ANSI 颜色码"""
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"

    @classmethod
    def disable(cls):
        for attr in dir(cls):
            if not attr.startswith("_") and attr != "disable":
                setattr(cls, attr, "")


def _c(text: str, color: str) -> str:
    return f"{color}{text}{Colors.END}"


def banner(title: str, w: int = 80) -> None:
    print()
    print("=" * w)
    print(_c(f"  {title}", Colors.BOLD + Colors.CYAN))
    print("=" * w)


def phase(num: int, name: str, party: str, w: int = 80) -> None:
    print()
    pc = Colors.GREEN if party == "B" else Colors.BLUE
    pl = _c(f"【Party {party}】", pc + Colors.BOLD)
    nl = _c(f"Step {num}: {name}", Colors.BOLD + Colors.YELLOW)
    print("─" * w)
    print(f"  {pl} {nl}")
    print("─" * w)


def sub(name: str) -> None:
    print(_c(f"  ▸ {name}", Colors.CYAN))


def kv(label: str, value: Any, ind: int = 4) -> None:
    print(f"{' ' * ind}{_c(label, Colors.DIM)}: {value}")


def vec(label: str, v: np.ndarray, ind: int = 4) -> None:
    stats = f"shape={
        v.shape}, min={
        v.min(): .3f}, max={
        v.max(): .3f}, mean={
        v.mean(): .3f} "
    print(f"{' ' * ind}{_c(label, Colors.DIM)}: {stats}")


def ok(msg: str) -> None:
    print(_c(f"  ✓ {msg}", Colors.GREEN))


def info(msg: str) -> None:
    print(_c(f"  ℹ {msg}", Colors.CYAN))


def arrow(label: str) -> None:
    print(_c(f"  → {label}", Colors.YELLOW + Colors.DIM))


def timing(label: str, sec: float, ind: int = 4) -> None:
    c = Colors.GREEN if sec < 1.0 else Colors.YELLOW if sec < 5.0 else Colors.RED
    print(f"{' ' * ind}{_c('⏱', c)} {_c(label, Colors.DIM)}: {sec:.3f}s")


def sep(c: str = "·", w: int = 80) -> None:
    print(_c(c * w, Colors.DIM))


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class QueryState:
    """单个查询在各步骤中的状态"""
    query_name: str
    query_index: int | None
    expected_label: bool | None
    # Step 2
    query_200_std: np.ndarray | None = None
    query_50_norm: np.ndarray | None = None
    first_round_req: FirstRoundRequest | None = None
    party_a_state: PartyALocalState | None = None
    # Step 3
    encrypted_scores: list | None = None
    # Step 4-5
    sim_scores: np.ndarray | None = None
    selected_cluster: int = -1
    second_round_req: SecondRoundRequest | None = None
    # Step 6-8
    encrypted_results: list | None = None
    # Step 9
    match_found: bool = False
    checked_columns: int = 0
    first_positive_column: int | None = None
    first_hit_name: str | None = None
    total_time_ms: float = 0.0
    is_fuzzy_demo: bool = False  # 标记为人为拼错的模糊匹配演示查询
    fuzzy_note: str | None = None  # 该 fuzzy 查询的错法说明（如 "漏字母"）
    expected_hit_name: str | None = None  # 期望命中的库内正确姓名（命中正确性判定用）


# =============================================================================
# 核心演示引擎：按步骤聚合所有查询
# =============================================================================

class StreamingDemoEngineV2:
    """按步骤聚合展示所有查询的演示引擎"""

    def __init__(
        self,
        names_b: list[str],
        k_mode: str | int = "sqrt",
        random_state: int = 42,
        tau: float = SIMILARITY_THRESHOLD,
        early_stop: bool = True,
    ):
        self.names_b = list(names_b)
        self.k_mode = k_mode
        self.random_state = random_state
        self.tau = tau
        self.early_stop = early_stop

        self.artifacts: OfflineArtifacts | None = None
        self.secret_context: Any = None
        self.public_context_bytes: bytes | None = None

        self.offline_time_ms: float = 0.0
        self.context_time_ms: float = 0.0

        # 3D 点云可视化数据（Step 1 填充）
        self.points_3d: list[list[float]] = []
        self.point_clusters: list[int] = []

    # =========================================================================
    # Step 1: Party B - Clustering (所有查询共享)
    # =========================================================================
    def run_step1_clustering(self, verbose: bool = True) -> None:
        """Step 1: B 侧离线聚类（一次性，所有查询共享）"""
        if verbose:
            phase(1, "Clustering (Offline)", "B")
            info(f"Database: {len(self.names_b)} names")
            kv("Sample", self.names_b[:3])

        t0 = time.perf_counter()

        # MinHash
        signatures_200 = batch_encode(self.names_b, NUM_PERMUTATIONS_CLUSTER)
        signatures_50 = signatures_200[:, :NUM_PERMUTATIONS_MATCH]
        if verbose:
            sub("MinHash Encoding")
            vec("Signatures_200", signatures_200)
            kv("Signatures_50", f"{signatures_50.shape} (sliced from EL=200)")

        # Normalize
        normalized_200 = l2_normalize(signatures_200)
        normalized_50 = l2_normalize(signatures_50)

        # StandardScaler
        scaler, standardized_200, scaler_mean, scaler_scale = fit_scaler(
            normalized_200)

        # K-Means
        k = choose_k(len(self.names_b), mode=self.k_mode)
        centroids, cluster_assignments = run_cosine_kmeans(
            standardized_200, k=k, iterations=20,
            random_state=self.random_state)
        cluster_sizes = np.bincount(cluster_assignments, minlength=k)
        if verbose:
            sub("K-Means Clustering")
            kv("K", k)
            kv(
                "Cluster sizes", f"min={
                    cluster_sizes.min()}, max={
                    cluster_sizes.max()}, mean={
                    cluster_sizes.mean():.1f}")
            kv("Assignments (first 20)", cluster_assignments[:20])

        # Build cluster matrix
        cluster_matrix, max_size = build_cluster_matrix(
            normalized_50, cluster_assignments, k)
        if verbose:
            sub("Build Cluster Matrix")
            kv("Shape",
               f"(k={k}, max_size={max_size}, dim={NUM_PERMUTATIONS_MATCH})")
            kv("Cluster sizes (first 5)", cluster_sizes[:5])

        self.artifacts = OfflineArtifacts(
            centroids=centroids,
            cluster_matrix=cluster_matrix,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            cluster_assignments=cluster_assignments,
            max_size=max_size,
        )

        # 3D 投影点云（可视化用，不影响协议）：用簇质心判别投影降到 3 维，
        # 让同簇点聚拢、不同簇分开。
        coords_3d, sample_idx = pca_project_3d(
            standardized_200, cluster_assignments=cluster_assignments, k=k)
        self.points_3d = [
            [round(float(c[0]), 4), round(float(c[1]), 4), round(float(c[2]), 4)]
            for c in coords_3d
        ]
        self.point_clusters = [
            int(cluster_assignments[i]) for i in sample_idx
        ]

        self.offline_time_ms = (time.perf_counter() - t0) * 1000
        if verbose:
            timing("Offline complete", self.offline_time_ms / 1000)
            ok(
                f"Artifacts: centroids={
                    centroids.shape}, cluster_matrix={
                    cluster_matrix.shape}")

    # =========================================================================
    # Step 2: Party A - Query Vector Encrypt (所有查询一起)
    # =========================================================================
    def run_step2_encrypt_queries(
            self, queries: list[QueryState],
            verbose: bool = True) -> None:
        """Step 2: A 侧批量加密所有查询"""
        if verbose:
            phase(2, "Query Vector Encrypt", "A")
            info(f"Processing {len(queries)} queries")

        t0 = time.perf_counter()

        # Create CKKS context (once)
        if self.secret_context is None:
            ctx_t0 = time.perf_counter()
            self.secret_context = create_ckks_context()
            self.secret_context.generate_relin_keys()
            self.public_context_bytes = serialize_public_context(
                self.secret_context)
            self.context_time_ms = (time.perf_counter() - ctx_t0) * 1000
            if verbose:
                sub("Create CKKS Context")
                timing("Context creation", self.context_time_ms / 1000)
                info("Galois keys + Relinearization keys ready")
        else:
            if verbose:
                info("Reusing existing CKKS context")

        # Encode & encrypt each query
        if verbose:
            sub("Encode & Encrypt Queries")
            print(f"{' ' * 4}{_c('Query',
                                 Colors.DIM):<25} {'EL=200':>12} {'EL=50':>12} {'Encrypt':>12}")
            print(f"{' ' * 4}{'-' * 65}")

        for q in queries:
            q.query_200_std, q.query_50_norm = encode_query_vectors(
                q.query_name, self.artifacts.scaler_mean, self.artifacts.scaler_scale)
            q.first_round_req = FirstRoundRequest(
                public_context_bytes=self.public_context_bytes,
                encrypted_query_200=encrypt(
                    q.query_200_std, self.secret_context),)
            q.party_a_state = PartyALocalState(
                secret_context=self.secret_context,
                encrypted_query_50=encrypt(
                    q.query_50_norm, self.secret_context),)
            if verbose:
                print(
                    f"{' ' * 4}{q.query_name[:24]:<25} "
                    f"{str(q.query_200_std.shape):>12} "
                    f"{str(q.query_50_norm.shape):>12} "
                    f"{'CKKSVector':>12}"
                )

        total = (time.perf_counter() - t0) * 1000
        if verbose:
            timing(f"{len(queries)} queries encrypted", total / 1000)
            ok("FirstRoundRequests ready")
            info("Private key & encrypted_query_50 remain local")

    # =========================================================================
    # Step 3: Party B - CompareToCentroids (所有查询一起)
    # =========================================================================
    def run_step3_compare_centroids(
            self, queries: list[QueryState],
            verbose: bool = True) -> None:
        """Step 3: B 侧批量计算与质心的相似度"""
        if verbose:
            phase(3, "CompareToCentroids", "B")
            info(
                f"Computing encrypted query-to-centroid dot products for {len(queries)} queries")
            kv("Centroids", self.artifacts.centroids.shape)

        t0 = time.perf_counter()

        if verbose:
            print(
                f"{' ' * 4}{_c('Query', Colors.DIM):<25} {'Scores':>10} {'Type':>15}")
            print(f"{' ' * 4}{'-' * 52}")

        for q in queries:
            q.encrypted_scores = [
                dot_ct_pt(q.first_round_req.encrypted_query_200, centroid)
                for centroid in self.artifacts.centroids
            ]
            if verbose:
                print(
                    f"{' ' * 4}{q.query_name[:24]:<25} "
                    f"{len(q.encrypted_scores):>10} "
                    f"{'CKKSVector':>15}"
                )

        total = (time.perf_counter() - t0) * 1000
        if verbose:
            timing(f"Centroid comparison", total / 1000)
            ok(f"Encrypted similarity scores ready for all queries")

    # =========================================================================
    # Step 4/5: Party A - Decrypt & Select Cluster (所有查询一起)
    # =========================================================================
    def run_step4_5_select_clusters(
            self, queries: list[QueryState],
            verbose: bool = True) -> None:
        """Step 4-5: A 侧批量解密、选择 cluster、构建 one-hot、重新加密"""
        if verbose:
            phase(4, "Decrypt & Select Cluster", "A")
            info("Decrypting similarity scores and selecting clusters")

        t0 = time.perf_counter()

        if verbose:
            print(f"{' ' * 4}{_c('Query',
                                 Colors.DIM):<25} {'Cluster':>8} {'Score':>10} {'Selector':>12}")
            print(f"{' ' * 4}{'-' * 57}")

        for q in queries:
            # Step 4: Decrypt
            q.sim_scores = decrypt_sim_scores(
                q.encrypted_scores, q.party_a_state.secret_context)
            # Step 5: Build one-hot & encrypt
            k = self.artifacts.centroids.shape[0]
            q.selected_cluster, selector = build_selector(q.sim_scores, k)
            encrypted_selector = encrypt_selector(
                selector, q.party_a_state.secret_context)
            q.second_round_req = SecondRoundRequest(
                encrypted_query_50=q.party_a_state.encrypted_query_50,
                encrypted_selector=encrypted_selector,
            )
            if verbose:
                print(
                    f"{' ' * 4}{q.query_name[:24]:<25} "
                    f"{q.selected_cluster:>8} "
                    f"{q.sim_scores[q.selected_cluster]:>10.4f} "
                    f"{'one-hot':>12}"
                )

        total = (time.perf_counter() - t0) * 1000
        if verbose:
            timing("Cluster selection & re-encryption", total / 1000)
            ok("SecondRoundRequests ready")
            info("B cannot know which cluster was selected (selector encrypted)")

    # =========================================================================
    # Step 6/7/8: Party B - Compute Sim (所有查询一起)
    # =========================================================================
    def run_step6_7_8_compute_sim(
            self, queries: list[QueryState],
            verbose: bool = True) -> None:
        """Step 6-8: B 侧批量列式匹配"""
        if verbose:
            phase(6, "Compute Sim (Selector × Matrix + Mask + Dot)", "B")
            info(
                "Step 6: Selector×Matrix  |  Step 7: Random Mask  |  Step 8: Encrypted Similarity")
            kv("Cluster matrix", f"{self.artifacts.cluster_matrix.shape}")

        t0 = time.perf_counter()

        k, max_size, dim = self.artifacts.cluster_matrix.shape

        if verbose:
            print(f"{' ' * 4}{_c('Query',
                                 Colors.DIM):<25} {'Cluster':>8} {'Columns':>8} {'Masks':>10}")
            print(f"{' ' * 4}{'-' * 55}")

        for q in queries:
            q.encrypted_results = []
            for col_idx in range(max_size):
                column_j = self.artifacts.cluster_matrix[:, col_idx, :]
                mask = _RNG.uniform(RANDOM_MASK_MIN, RANDOM_MASK_MAX)
                masked_column_j = mask * column_j
                encrypted_selected_name = matmul_ct_pt(
                    q.second_round_req.encrypted_selector, masked_column_j
                )
                encrypted_cos_score = dot_ct_ct(
                    q.second_round_req.encrypted_query_50,
                    encrypted_selected_name)
                encrypted_final = add_plain(
                    encrypted_cos_score, -mask * self.tau)
                q.encrypted_results.append(encrypted_final)

            if verbose:
                print(
                    f"{' ' * 4}{q.query_name[:24]:<25} "
                    f"{q.selected_cluster:>8} "
                    f"{len(q.encrypted_results):>8} "
                    f"{'applied':>10}"
                )

        total = (time.perf_counter() - t0) * 1000
        if verbose:
            timing("Column-wise matching", total / 1000)
            ok("Encrypted scores ready for all queries")
            info("Random masks per column: B cannot learn raw similarity")

    # =========================================================================
    # Step 9: Party A - Check Sim (所有查询一起)
    # =========================================================================
    def run_step9_check_sim(
            self, queries: list[QueryState],
            verbose: bool = True) -> None:
        """Step 9: A 侧批量解密判断"""
        if verbose:
            phase(9, "Check Sim (Decrypt & Judge)", "A")
            info(
                f"Checking {
                    len(queries)} queries: decrypt column scores, find positive values")
            kv("Threshold τ", self.tau)
            kv("Early stop", self.early_stop)

        t0 = time.perf_counter()

        if verbose:
            print(f"{' ' * 4}{_c('Query',
                                 Colors.DIM):<25} {'Match':>6} {'Checked':>8} {'Hit Col':>8} {'Hit Name':>20}")
            print(f"{' ' * 4}{'-' * 71}")

        for q in queries:
            checked = 0
            first_positive = None

            for col_idx, enc_score in enumerate(q.encrypted_results):
                values = (
                    enc_score.decrypt()
                    if hasattr(enc_score, "decrypt")
                    else q.party_a_state.secret_context.decrypt(enc_score)
                )
                plain_score = float(np.asarray(
                    values, dtype=np.float64).reshape(-1)[0])
                checked += 1

                if plain_score > 1e-6 and first_positive is None:
                    first_positive = col_idx
                    if self.early_stop:
                        break

            q.match_found = first_positive is not None
            q.checked_columns = checked
            q.first_positive_column = first_positive

            # Find hit name
            if first_positive is not None:
                members = np.where(
                    self.artifacts.cluster_assignments == q.selected_cluster)[0]
                if first_positive < len(members):
                    db_idx = int(members[first_positive])
                    q.first_hit_name = self.names_b[db_idx]

            match_str = _c(
                "Y", Colors.GREEN) if q.match_found else _c(
                "N", Colors.RED)
            hit_name = (q.first_hit_name or "")[:18]
            if verbose:
                print(
                    f"{' ' * 4}{q.query_name[:24]:<25} "
                    f"{match_str:>6} "
                    f"{q.checked_columns:>8} "
                    f"{first_positive if first_positive is not None else '-':>8} "
                    f"{hit_name:>20}"
                )

        total = (time.perf_counter() - t0) * 1000
        if verbose:
            timing("Match checking", total / 1000)

    # =========================================================================
    # 端到端批量演示
    # =========================================================================
    def run_all_queries(
        self, queries: list[QueryState], verbose: bool = True
    ) -> list[QueryState]:
        """运行所有查询的完整 9 步演示"""
        for _ in self.run_all_queries_with_events(queries, verbose=verbose):
            pass  # consume generator
        return self._last_results

    def run_all_queries_with_events(
        self, queries: list[QueryState], verbose: bool = True
    ):
        """Generator that yields structured events at each protocol step.

        Yields dicts of the form:
            {"event": "step_complete", "step": N, "name": "...", "party": "A"|"B",
             "data": {...}, "timing_ms": float}

        The final yield is:
            {"event": "complete", "summary": {...}}
        """
        t_total = time.perf_counter()
        step_timings = {}

        # ── Step 1: Clustering ──
        t0 = time.perf_counter()
        self.run_step1_clustering(verbose=verbose)
        step_timings[1] = (time.perf_counter() - t0) * 1000

        centroids_np = self.artifacts.centroids
        k = centroids_np.shape[0]
        cluster_counts = np.bincount(
            self.artifacts.cluster_assignments, minlength=k
        ).tolist()

        yield {
            "event": "step_complete",
            "step": 1,
            "name": "Clustering (Offline)",
            "party": "B",
            "data": {
                "db_size": len(self.names_b),
                "k": k,
                "cluster_sizes": cluster_counts,
                "max_cluster_size": int(self.artifacts.max_size),
                "signature_dim": 200,
                "match_dim": 50,
                "sample_names": self.names_b[:5],
                "points_3d": self.points_3d,
                "point_clusters": self.point_clusters,
            },
            "timing_ms": round(step_timings[1], 2),
        }

        # ── Step 2: Query Encrypt ──
        t0 = time.perf_counter()
        self.run_step2_encrypt_queries(queries, verbose=verbose)
        step_timings[2] = (time.perf_counter() - t0) * 1000

        yield {
            "event": "step_complete",
            "step": 2,
            "name": "Query Vector Encrypt",
            "party": "A",
            "data": {
                "num_queries": len(queries),
                "context_time_ms": round(self.context_time_ms, 2),
                "query_names": [q.query_name for q in queries],
            },
            "timing_ms": round(step_timings[2], 2),
        }

        # ── Step 3: Compare to Centroids ──
        t0 = time.perf_counter()
        self.run_step3_compare_centroids(queries, verbose=verbose)
        step_timings[3] = (time.perf_counter() - t0) * 1000

        yield {
            "event": "step_complete",
            "step": 3,
            "name": "Compare to Centroids",
            "party": "B",
            "data": {
                "num_centroids": k,
                "num_queries": len(queries),
            },
            "timing_ms": round(step_timings[3], 2),
        }

        # ── Step 4/5: Decrypt & Select ──
        t0 = time.perf_counter()
        self.run_step4_5_select_clusters(queries, verbose=verbose)
        step_timings[4] = (time.perf_counter() - t0) * 1000

        # Collect per-query centroid similarity scores for visualization
        centroid_scores = []
        for q in queries:
            centroid_scores.append({
                "query_name": q.query_name,
                "selected_cluster": int(q.selected_cluster),
                "scores": [round(float(s), 4) for s in q.sim_scores],
                "is_fuzzy": q.is_fuzzy_demo,
                "expected_label": q.expected_label,
            })

        yield {
            "event": "step_complete",
            "step": 4,
            "name": "Decrypt & Select Cluster",
            "party": "A",
            "data": {
                "centroid_scores": centroid_scores,
                "num_centroids": k,
            },
            "timing_ms": round(step_timings[4], 2),
        }

        # ── Step 6/7/8: Column Matching ──
        t0 = time.perf_counter()
        self.run_step6_7_8_compute_sim(queries, verbose=verbose)
        step_timings[6] = (time.perf_counter() - t0) * 1000

        yield {
            "event": "step_complete",
            "step": 6,
            "name": "Column-wise Matching",
            "party": "B",
            "data": {
                "num_queries": len(queries),
                "columns_per_query": [
                    {"query_name": q.query_name, "num_columns": len(q.encrypted_results)}
                    for q in queries
                ],
            },
            "timing_ms": round(step_timings[6], 2),
        }

        # ── Step 9: Check Sim ──
        t0 = time.perf_counter()
        # We need per-column scores for the frontend visualization.
        # run_step9_check_sim decrypts and judges but doesn't store all scores.
        # We do a manual check here that also collects per-column plaintext
        # scores.
        if verbose:
            phase(9, "Check Sim (Decrypt & Judge)", "A")
            info(
                f"Checking {
                    len(queries)} queries: decrypt column scores, find positive values")
            kv("Threshold τ", self.tau)
            kv("Early stop", self.early_stop)
            print(f"{' ' * 4}{_c('Query',
                                 Colors.DIM):<25} {'Match':>6} {'Checked':>8} {'Hit Col':>8} {'Hit Name':>20}")
            print(f"{' ' * 4}{'-' * 71}")

        column_scores_data = []
        for q in queries:
            checked = 0
            first_positive = None
            col_scores = []

            for col_idx, enc_score in enumerate(q.encrypted_results):
                values = (
                    enc_score.decrypt()
                    if hasattr(enc_score, "decrypt")
                    else q.party_a_state.secret_context.decrypt(enc_score)
                )
                plain_score = float(np.asarray(
                    values, dtype=np.float64).reshape(-1)[0])
                col_scores.append(round(plain_score, 6))
                checked += 1

                if plain_score > 1e-6 and first_positive is None:
                    first_positive = col_idx
                    if self.early_stop:
                        break

            q.match_found = first_positive is not None
            q.checked_columns = checked
            q.first_positive_column = first_positive

            if first_positive is not None:
                members = np.where(
                    self.artifacts.cluster_assignments == q.selected_cluster)[0]
                if first_positive < len(members):
                    db_idx = int(members[first_positive])
                    q.first_hit_name = self.names_b[db_idx]

            column_scores_data.append(
                {"query_name": q.query_name, "match_found": q.match_found,
                 "checked_columns": checked,
                 "first_positive_column": first_positive
                 if first_positive is not None else -1,
                 "hit_name": q.first_hit_name or "",
                 "expected_hit_name": q.expected_hit_name or "",
                 "is_fuzzy": q.is_fuzzy_demo,
                 "is_correct": is_query_correct(q),
                 "selected_cluster": int(q.selected_cluster),
                 "column_scores": col_scores, })

            if verbose:
                match_str = _c(
                    "Y", Colors.GREEN) if q.match_found else _c(
                    "N", Colors.RED)
                hit_name = (q.first_hit_name or "")[:18]
                print(
                    f"{' ' * 4}{q.query_name[:24]:<25} "
                    f"{match_str:>6} "
                    f"{q.checked_columns:>8} "
                    f"{first_positive if first_positive is not None else '-':>8} "
                    f"{hit_name:>20}"
                )

        step_timings[9] = (time.perf_counter() - t0) * 1000

        yield {
            "event": "step_complete",
            "step": 9,
            "name": "Check Similarity (Decrypt & Judge)",
            "party": "A",
            "data": {
                "column_scores": column_scores_data,
                "tau": self.tau,
                "early_stop": self.early_stop,
            },
            "timing_ms": round(step_timings[9], 2),
        }

        # ── Summary ──
        total_ms = (time.perf_counter() - t_total) * 1000

        if verbose:
            print()
            sep("=", 80)
            print(_c("  FINAL RESULTS SUMMARY", Colors.BOLD + Colors.YELLOW))
            sep("=", 80)
            print(
                f"{
                    ' ' * 4}{
                    _c(
                        'Query',
                        Colors.DIM):<25} {
                    'Label':>6} {
                        'Pred':>6} {
                            'Cluster':>8} {
                                'Correct':>8} {
                                    'Time(ms)':>10}")
            print(f"{' ' * 4}{'-' * 65}")
            for q in queries:
                label = "Y" if q.expected_label else "N" if q.expected_label is not None else "?"
                pred = _c(
                    "Y", Colors.GREEN) if q.match_found else _c(
                    "N", Colors.RED)
                correct = is_query_correct(q)
                mark = _c("✓", Colors.GREEN) if correct else _c(
                    "✗", Colors.RED) if correct is False else " "
                tag = _c("  ◀ FUZZY", Colors.BOLD +
                         Colors.YELLOW) if q.is_fuzzy_demo else ""
                print(
                    f"{' ' * 4}{q.query_name[:24]:<25} "
                    f"{label:>6} "
                    f"{pred:>6} "
                    f"{q.selected_cluster:>8} "
                    f"{mark:>8} "
                    f"{q.total_time_ms:>10.1f}"
                    f"{tag}"
                )

            fuzzy_queries = [q for q in queries if q.is_fuzzy_demo]
            if fuzzy_queries:
                print()
                print(
                    _c(
                        "  Fuzzy match demo (misspelled queries still hit the correct name):",
                        Colors.BOLD +
                        Colors.CYAN))
                for q in fuzzy_queries:
                    note = f" [{q.fuzzy_note}]" if q.fuzzy_note else ""
                    if q.match_found:
                        outcome = _c(
                            "-> hit ", Colors.GREEN) + _c(f"\"{q.first_hit_name or '?'}\"", Colors.GREEN)
                    else:
                        outcome = _c("-> miss", Colors.RED)
                    print(
                        f"    {q.query_name:<26}{outcome}{_c(note, Colors.DIM)}")

            correct = sum(
                1 for q in queries if is_query_correct(q) is True)
            total_with_label = sum(
                1 for q in queries if is_query_correct(q) is not None)
            if total_with_label > 0:
                acc = correct / total_with_label * 100
                print()
                ok(f"Accuracy: {correct}/{total_with_label} = {acc:.1f}%")

            timing("Total query time", total_ms / 1000)
            timing("Offline prep time", self.offline_time_ms / 1000)

        # Build summary
        results_list = []
        for q in queries:
            results_list.append(
                {"query_name": q.query_name, "query_index": q.query_index,
                 "expected_label": q.expected_label,
                 "match_found": q.match_found,
                 "selected_cluster": int(q.selected_cluster),
                 "checked_columns": q.checked_columns,
                 "hit_name": q.first_hit_name or "",
                 "expected_hit_name": q.expected_hit_name or "",
                 "is_fuzzy": q.is_fuzzy_demo,
                 "fuzzy_note": q.fuzzy_note or "",
                 "is_correct": is_query_correct(q),
                 "first_positive_column": q.first_positive_column
                 if q.first_positive_column is not None else -1, })

        correct = sum(1 for q in queries if is_query_correct(q) is True)
        total_with_label = sum(
            1 for q in queries if is_query_correct(q) is not None)

        self._last_results = queries

        yield {
            "event": "complete",
            "summary": {
                "results": results_list,
                "accuracy": {
                    "correct": correct,
                    "total": total_with_label,
                    "percent": round(correct / total_with_label * 100, 1) if total_with_label > 0 else None,
                },
                "timing_ms": {
                    "step1_clustering": round(step_timings.get(1, 0), 2),
                    "step2_encrypt": round(step_timings.get(2, 0), 2),
                    "step3_compare_centroids": round(step_timings.get(3, 0), 2),
                    "step4_select_cluster": round(step_timings.get(4, 0), 2),
                    "step6_column_matching": round(step_timings.get(6, 0), 2),
                    "step9_check_sim": round(step_timings.get(9, 0), 2),
                    "total": round(total_ms, 2),
                    "offline": round(self.offline_time_ms, 2),
                },
                "config": {
                    "db_size": len(self.names_b),
                    "k": k,
                    "tau": self.tau,
                    "early_stop": self.early_stop,
                    "num_queries": len(queries),
                },
            },
        }


# =============================================================================
# CLI
# =============================================================================

def _parse_indices(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


# 演示用查询集（GUI 与 CLI 共享，确保两端展示一致）。
# 每个库内真实姓名后紧跟一条人为拼错的查询，正确查询与对应 fuzzy 查询一上一下相邻。
# (正确姓名, 拼错查询, 错法说明)
DEMO_PAIRS = [
    ("JAMES MICHAEL AARON", "JAMES MICHAEL ARON",
     "dropped letter (JAMES->JAMS)"),
    ("TIMOTHY DUANE AARMSTRONG", "TIMOTHY DUANE ARMSTRONG",
     "dropped double letter (AARMSTRONG->ARMSTRONG)"),
    ("NATHAN EDWARD AARON", "NATHAN EDWARD AARO",
     "dropped trailing letter (AARON->AARO)"),
    ("JOY GAYLE ABAZIED COOPER", "JOY GAYL ABAZIED COOPER",
     "dropped trailing letter (GAYLE->GAYL)"),
    ("HAROLD GRANT ABBOTT", "HAROLD GRNT ABBOTT",
     "dropped vowel (GRANT->GRNT)"),
]

# 反例：库内不存在的姓名，期望全部不命中，证明系统不会乱匹配。
DEMO_NEGATIVE_NAMES = [
    "DAVID ALLEN BLACK",
    "DEBORAH ELAINE BLACK",
    "DONNA WEST BLACK",
]


def build_demo_queries(
    names_a: list[str], labels: list
) -> list[QueryState]:
    """构建标准演示查询集：5 对 (正确名 + 拼错名) + 3 个库外反例。

    GUI 默认运行与 CLI `python demo_streaming.py`（无 --query）共用此函数，
    保证前端演示和终端演示使用完全相同的查询列表。
    """
    queries: list[QueryState] = []
    for correct_name, fuzzy_name, note in DEMO_PAIRS:
        idx = names_a.index(correct_name) if correct_name in names_a else None
        queries.append(
            QueryState(
                query_name=correct_name,
                query_index=idx,
                expected_label=bool(labels[idx]) if idx is not None else True,
                expected_hit_name=correct_name,
            )
        )
        queries.append(
            QueryState(
                query_name=fuzzy_name,
                query_index=None,
                expected_label=True,
                is_fuzzy_demo=True,
                fuzzy_note=note,
                expected_hit_name=correct_name,
            )
        )

    for neg_name in DEMO_NEGATIVE_NAMES:
        idx = names_a.index(neg_name) if neg_name in names_a else None
        queries.append(
            QueryState(
                query_name=neg_name,
                query_index=idx,
                expected_label=bool(labels[idx]) if idx is not None else False,
                expected_hit_name=None,
            )
        )

    return queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming demo v2: step-aggregated, privacy-preserving fuzzy name matching"
    )
    parser.add_argument("--data-path", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--db-limit", type=int, default=100)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument(
        "--query-indices",
        type=_parse_indices,
        default=[
            2,
            7,
            26,
            49,
            100])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--tau", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(
            PROJECT_ROOT /
            "artifacts" /
            "demo" /
            "streaming_v2"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.no_color:
        Colors.disable()

    # Load dataset
    names_a, names_b_all, labels = load_dataset("ncvr_10k", args.data_path)
    names_b = names_b_all[: args.db_limit]

    # Build queries
    if args.query:
        queries = [
            QueryState(
                query_name=args.query,
                query_index=None,
                expected_label=None)]
    else:
        queries = build_demo_queries(names_a, labels)

    if not queries:
        print("No valid queries.")
        return 1

    # Banner
    banner("PRIVACY-PRESERVING FUZZY NAME MATCHING - STREAMING DEMO v2")
    info(
        f"Database: {
            len(names_b)} records | Queries: {
            len(queries)} | K: {
                args.k} | τ: {
                    args.tau}")

    # Run
    engine = StreamingDemoEngineV2(
        names_b=names_b,
        k_mode=args.k,
        tau=args.tau,
        early_stop=not args.no_early_stop,
    )
    results = engine.run_all_queries(queries, verbose=True)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "dataset": "ncvr_10k",
            "db_limit": args.db_limit,
            "k": args.k,
            "tau": args.tau,
            "early_stop": not args.no_early_stop,
        },
        "timing_ms": {
            "offline": engine.offline_time_ms,
            "context": engine.context_time_ms,
        },
        "queries": [
            {
                "query_name": q.query_name,
                "query_index": q.query_index,
                "expected_label": q.expected_label,
                "match_found": q.match_found,
                "selected_cluster": q.selected_cluster,
                "checked_columns": q.checked_columns,
                "first_hit_name": q.first_hit_name,
            }
            for q in results
        ],
    }
    json_path = output_dir / "streaming_v2_results.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    ok(f"Results saved to: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
