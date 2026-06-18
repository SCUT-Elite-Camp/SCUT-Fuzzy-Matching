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

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.params import (
    NUM_PERMUTATIONS_CLUSTER,
    NUM_PERMUTATIONS_MATCH,
    SIMILARITY_THRESHOLD,
    choose_k,
    RANDOM_MASK_MIN,
    RANDOM_MASK_MAX,
)
from ckks.context import create_ckks_context
from ckks.keys import encrypt, serialize_public_context
from ckks.operations import dot_ct_pt, dot_ct_ct, matmul_ct_pt, add_plain
from clustering.kmeans_cosine import run_cosine_kmeans
from evaluation.dataset_loader import load_dataset
from minhash.encoder import batch_encode, _generate_shingles, _hash_shingle
from party_a.local_prep import encode_query_vectors
from party_a.online_querier import decrypt_sim_scores, build_selector, encrypt_selector
from party_b.offline_prep import fit_scaler, build_cluster_matrix
from preprocessing.normalizer import l2_normalize
from preprocessing.text_cleaner import clean_name
from protocol.types import (
    OfflineArtifacts,
    FirstRoundRequest,
    PartyALocalState,
    SecondRoundRequest,
)
from random import SystemRandom

_RNG = SystemRandom()


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
    stats = f"shape={v.shape}, min={v.min():.3f}, max={v.max():.3f}, mean={v.mean():.3f}"
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
        scaler, standardized_200, scaler_mean, scaler_scale = fit_scaler(normalized_200)

        # K-Means
        k = choose_k(len(self.names_b), mode=self.k_mode)
        centroids, cluster_assignments = run_cosine_kmeans(
            standardized_200, k=k, iterations=20, random_state=self.random_state
        )
        cluster_sizes = np.bincount(cluster_assignments, minlength=k)
        if verbose:
            sub("K-Means Clustering")
            kv("K", k)
            kv("Cluster sizes", f"min={cluster_sizes.min()}, max={cluster_sizes.max()}, mean={cluster_sizes.mean():.1f}")
            kv("Assignments (first 20)", cluster_assignments[:20])

        # Build cluster matrix
        cluster_matrix, max_size = build_cluster_matrix(normalized_50, cluster_assignments, k)
        if verbose:
            sub("Build Cluster Matrix")
            kv("Shape", f"(k={k}, max_size={max_size}, dim={NUM_PERMUTATIONS_MATCH})")
            kv("Cluster sizes (first 5)", cluster_sizes[:5])

        self.artifacts = OfflineArtifacts(
            centroids=centroids,
            cluster_matrix=cluster_matrix,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            cluster_assignments=cluster_assignments,
            max_size=max_size,
        )

        self.offline_time_ms = (time.perf_counter() - t0) * 1000
        if verbose:
            timing("Offline complete", self.offline_time_ms / 1000)
            ok(f"Artifacts: centroids={centroids.shape}, cluster_matrix={cluster_matrix.shape}")

    # =========================================================================
    # Step 2: Party A - Query Vector Encrypt (所有查询一起)
    # =========================================================================
    def run_step2_encrypt_queries(self, queries: list[QueryState], verbose: bool = True) -> None:
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
            self.public_context_bytes = serialize_public_context(self.secret_context)
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
            print(f"{' ' * 4}{_c('Query', Colors.DIM):<25} {'EL=200':>12} {'EL=50':>12} {'Encrypt':>12}")
            print(f"{' ' * 4}{'-' * 65}")

        for q in queries:
            q.query_200_std, q.query_50_norm = encode_query_vectors(
                q.query_name, self.artifacts.scaler_mean, self.artifacts.scaler_scale
            )
            q.first_round_req = FirstRoundRequest(
                public_context_bytes=self.public_context_bytes,
                encrypted_query_200=encrypt(q.query_200_std, self.secret_context),
            )
            q.party_a_state = PartyALocalState(
                secret_context=self.secret_context,
                encrypted_query_50=encrypt(q.query_50_norm, self.secret_context),
            )
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
    def run_step3_compare_centroids(self, queries: list[QueryState], verbose: bool = True) -> None:
        """Step 3: B 侧批量计算与质心的相似度"""
        if verbose:
            phase(3, "CompareToCentroids", "B")
            info(f"Computing encrypted query-to-centroid dot products for {len(queries)} queries")
            kv("Centroids", self.artifacts.centroids.shape)

        t0 = time.perf_counter()

        if verbose:
            print(f"{' ' * 4}{_c('Query', Colors.DIM):<25} {'Scores':>10} {'Type':>15}")
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
    def run_step4_5_select_clusters(self, queries: list[QueryState], verbose: bool = True) -> None:
        """Step 4-5: A 侧批量解密、选择 cluster、构建 one-hot、重新加密"""
        if verbose:
            phase(4, "Decrypt & Select Cluster", "A")
            info("Decrypting similarity scores and selecting clusters")

        t0 = time.perf_counter()

        if verbose:
            print(f"{' ' * 4}{_c('Query', Colors.DIM):<25} {'Cluster':>8} {'Score':>10} {'Selector':>12}")
            print(f"{' ' * 4}{'-' * 57}")

        for q in queries:
            # Step 4: Decrypt
            q.sim_scores = decrypt_sim_scores(q.encrypted_scores, q.party_a_state.secret_context)
            # Step 5: Build one-hot & encrypt
            k = self.artifacts.centroids.shape[0]
            q.selected_cluster, selector = build_selector(q.sim_scores, k)
            encrypted_selector = encrypt_selector(selector, q.party_a_state.secret_context)
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
    def run_step6_7_8_compute_sim(self, queries: list[QueryState], verbose: bool = True) -> None:
        """Step 6-8: B 侧批量列式匹配"""
        if verbose:
            phase(6, "Compute Sim (Selector × Matrix + Mask + Dot)", "B")
            info("Step 6: Selector×Matrix  |  Step 7: Random Mask  |  Step 8: Encrypted Similarity")
            kv("Cluster matrix", f"{self.artifacts.cluster_matrix.shape}")

        t0 = time.perf_counter()

        k, max_size, dim = self.artifacts.cluster_matrix.shape

        if verbose:
            print(f"{' ' * 4}{_c('Query', Colors.DIM):<25} {'Cluster':>8} {'Columns':>8} {'Masks':>10}")
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
                    q.second_round_req.encrypted_query_50, encrypted_selected_name
                )
                encrypted_final = add_plain(encrypted_cos_score, -mask * self.tau)
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
    def run_step9_check_sim(self, queries: list[QueryState], verbose: bool = True) -> None:
        """Step 9: A 侧批量解密判断"""
        if verbose:
            phase(9, "Check Sim (Decrypt & Judge)", "A")
            info(f"Checking {len(queries)} queries: decrypt column scores, find positive values")
            kv("Threshold τ", self.tau)
            kv("Early stop", self.early_stop)

        t0 = time.perf_counter()

        if verbose:
            print(f"{' ' * 4}{_c('Query', Colors.DIM):<25} {'Match':>6} {'Checked':>8} {'Hit Col':>8} {'Hit Name':>20}")
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
                plain_score = float(np.asarray(values, dtype=np.float64).reshape(-1)[0])
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
                members = np.where(self.artifacts.cluster_assignments == q.selected_cluster)[0]
                if first_positive < len(members):
                    db_idx = int(members[first_positive])
                    q.first_hit_name = self.names_b[db_idx]

            match_str = _c("Y", Colors.GREEN) if q.match_found else _c("N", Colors.RED)
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
        t_total = time.perf_counter()

        # Step 1 (once per database)
        self.run_step1_clustering(verbose=verbose)

        # Step 2-9 (batch all queries)
        self.run_step2_encrypt_queries(queries, verbose=verbose)
        self.run_step3_compare_centroids(queries, verbose=verbose)
        self.run_step4_5_select_clusters(queries, verbose=verbose)
        self.run_step6_7_8_compute_sim(queries, verbose=verbose)
        self.run_step9_check_sim(queries, verbose=verbose)

        total_ms = (time.perf_counter() - t_total) * 1000

        # Summary
        if verbose:
            print()
            sep("=", 80)
            print(_c("  FINAL RESULTS SUMMARY", Colors.BOLD + Colors.YELLOW))
            sep("=", 80)
            print(f"{' ' * 4}{_c('Query', Colors.DIM):<25} {'Label':>6} {'Pred':>6} {'Cluster':>8} {'Match':>6} {'Time(ms)':>10}")
            print(f"{' ' * 4}{'-' * 65}")
            for q in queries:
                label = "Y" if q.expected_label else "N" if q.expected_label is not None else "?"
                pred = _c("Y", Colors.GREEN) if q.match_found else _c("N", Colors.RED)
                correct = (q.expected_label == q.match_found) if q.expected_label is not None else None
                mark = _c("✓", Colors.GREEN) if correct else _c("✗", Colors.RED) if correct is False else " "
                print(
                    f"{' ' * 4}{q.query_name[:24]:<25} "
                    f"{label:>6} "
                    f"{pred:>6} "
                    f"{q.selected_cluster:>8} "
                    f"{mark:>6} "
                    f"{q.total_time_ms:>10.1f}"
                )

            correct = sum(
                1 for q in queries
                if q.expected_label is not None and q.match_found == q.expected_label
            )
            total_with_label = sum(1 for q in queries if q.expected_label is not None)
            if total_with_label > 0:
                acc = correct / total_with_label * 100
                print()
                ok(f"Accuracy: {correct}/{total_with_label} = {acc:.1f}%")

            timing("Total query time", total_ms / 1000)
            timing("Offline prep time", self.offline_time_ms / 1000)

        return queries


# =============================================================================
# CLI
# =============================================================================

def _parse_indices(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming demo v2: step-aggregated, privacy-preserving fuzzy name matching"
    )
    parser.add_argument("--data-path", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--db-limit", type=int, default=100)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--query-indices", type=_parse_indices, default=[2, 7, 26, 49, 100])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--tau", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "artifacts" / "demo" / "streaming_v2"))
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
        queries = [QueryState(query_name=args.query, query_index=None, expected_label=None)]
    else:
        queries = [
            QueryState(query_name=names_a[i], query_index=i, expected_label=bool(labels[i]))
            for i in args.query_indices
            if 0 <= i < len(names_a)
        ]

    if not queries:
        print("No valid queries.")
        return 1

    # Banner
    banner("PRIVACY-PRESERVING FUZZY NAME MATCHING - STREAMING DEMO v2")
    info(f"Database: {len(names_b)} records | Queries: {len(queries)} | K: {args.k} | τ: {args.tau}")

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
