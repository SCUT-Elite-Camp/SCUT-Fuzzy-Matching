from __future__ import annotations

from dataclasses import dataclass
from random import SystemRandom
from typing import Iterable

import numpy as np
import tenseal as ts
from sklearn.preprocessing import StandardScaler

from ckks.context import create_ckks_context
from ckks.keys import encrypt, serialize_public_context
from ckks.operations import add_plain, dot_ct_ct, dot_ct_pt, matmul_ct_pt, mul_plain
from clustering.kmeans_cosine import run_cosine_kmeans
from config.params import KMEANS_ITERATIONS, RANDOM_MASK_MAX, RANDOM_MASK_MIN, choose_k

from .config import MultiAttributeConfig
from .encoder import encode_record_vectors
from .model import MatchRecord

_RNG = SystemRandom()


@dataclass
class MultiOfflineArtifacts:
    centroids: np.ndarray
    cluster_matrix: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    cluster_assignments: np.ndarray
    max_size: int


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


@dataclass
class MultiProtocolRun:
    catch: bool
    selected_cluster: int
    checked_columns: int
    first_positive_column: int | None
    artifacts: MultiOfflineArtifacts


def _build_cluster_matrix(
    match_vectors: np.ndarray,
    assignments: np.ndarray,
    k: int,
) -> tuple[np.ndarray, int]:
    matrix = np.asarray(match_vectors, dtype=np.float64)
    assignments = np.asarray(assignments, dtype=np.int32)
    if matrix.ndim != 2:
        raise ValueError("match_vectors must be 2-D")
    if assignments.shape != (matrix.shape[0],):
        raise ValueError("cluster assignment count mismatch")

    sizes = np.bincount(assignments, minlength=k)
    max_size = int(sizes.max()) if len(sizes) else 0
    out = np.zeros((k, max_size, matrix.shape[1]), dtype=np.float64)
    for cluster_idx in range(k):
        members = matrix[assignments == cluster_idx]
        out[cluster_idx, : len(members), :] = members
    return out, max_size


def prepare_party_b_multi_offline(
    records_b: Iterable[MatchRecord],
    *,
    cfg: MultiAttributeConfig | None = None,
    k_mode: str | int = "sqrt",
    random_state: int = 42,
) -> MultiOfflineArtifacts:
    cfg = cfg or MultiAttributeConfig()
    records = list(records_b)
    if not records:
        raise ValueError("records_b cannot be empty")

    cluster_vectors, match_vectors = encode_record_vectors(records, cfg)

    scaler = StandardScaler()
    standardized = scaler.fit_transform(cluster_vectors)

    # StandardScaler may produce exact zero scale for a constant feature; sklearn
    # exposes scale_=1 for such features, so A-side division remains safe.
    k = choose_k(len(records), mode=k_mode)
    centroids, assignments = run_cosine_kmeans(
        standardized,
        k=k,
        iterations=KMEANS_ITERATIONS,
        random_state=random_state,
    )
    cluster_matrix, max_size = _build_cluster_matrix(
        match_vectors, assignments, centroids.shape[0]
    )

    return MultiOfflineArtifacts(
        centroids=centroids,
        cluster_matrix=cluster_matrix,
        scaler_mean=scaler.mean_.astype(np.float64),
        scaler_scale=scaler.scale_.astype(np.float64),
        cluster_assignments=assignments,
        max_size=max_size,
    )


def prepare_party_a_multi_query(
    query: MatchRecord,
    artifacts: MultiOfflineArtifacts,
    *,
    cfg: MultiAttributeConfig | None = None,
) -> tuple[MultiFirstRoundRequest, MultiPartyAState]:
    cfg = cfg or MultiAttributeConfig()
    cluster_vec, match_vec = encode_record_vectors([query], cfg)
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


def choose_multi_cluster(
    encrypted_scores: list[ts.CKKSVector | bytes],
    state: MultiPartyAState,
) -> tuple[MultiSecondRoundRequest, int]:
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

    selected = int(np.argmax(np.asarray(scores)))
    selector = np.zeros(len(scores), dtype=np.float64)
    selector[selected] = 1.0
    enc_selector = encrypt(selector, state.secret_context)

    return (
        MultiSecondRoundRequest(
            encrypted_match_query=state.encrypted_match_query.serialize(),
            encrypted_selector=enc_selector.serialize(),
        ),
        selected,
    )


def column_wise_multi_matching(
    cluster_matrix: np.ndarray,
    request: MultiSecondRoundRequest,
    public_context: bytes | ts.Context,
    *,
    tau: float,
):
    matrix = np.asarray(cluster_matrix, dtype=np.float64)
    if matrix.ndim != 3:
        raise ValueError("cluster_matrix must be 3-D")
    k, max_size, match_dim = matrix.shape
    ctx = _load_public_context(public_context)
    enc_query = _load_cipher(request.encrypted_match_query, ctx)
    enc_selector = _load_cipher(request.encrypted_selector, ctx)

    if enc_query.size() != match_dim:
        raise ValueError(
            f"encrypted match query dim {enc_query.size()} != {match_dim}"
        )
    if enc_selector.size() != k:
        raise ValueError(f"selector dim {enc_selector.size()} != k={k}")

    for column_idx in range(max_size):
        # Every row is one cluster; encrypted one-hot selector privately picks
        # the candidate vector from the selected cluster in this column.
        plain_column = matrix[:, column_idx, :]  # (k, match_dim)
        enc_candidate = matmul_ct_pt(enc_selector, plain_column)
        enc_score = dot_ct_ct(enc_candidate, enc_query)
        enc_score = add_plain(enc_score, -float(tau))
        mask = _RNG.uniform(RANDOM_MASK_MIN, RANDOM_MASK_MAX)
        yield mul_plain(enc_score, mask).serialize()


def decrypt_multi_match(
    encrypted_scores,
    state: MultiPartyAState,
    *,
    eps: float = 1e-6,
    early_stop: bool = True,
) -> tuple[bool, int, int | None]:
    checked = 0
    first_positive = None
    for column_idx, item in enumerate(encrypted_scores):
        if isinstance(item, bytes):
            item = ts.ckks_vector_from(state.secret_context, item)
        value = float(np.asarray(item.decrypt(), dtype=np.float64).reshape(-1)[0])
        checked += 1
        if value > eps and first_positive is None:
            first_positive = column_idx
            if early_stop:
                return True, checked, first_positive
    return first_positive is not None, checked, first_positive


def run_multi_attribute_protocol(
    records_b: Iterable[MatchRecord],
    query: MatchRecord,
    *,
    cfg: MultiAttributeConfig | None = None,
    k_mode: str | int = "sqrt",
    random_state: int = 42,
    early_stop: bool = True,
) -> MultiProtocolRun:
    """Run the complete two-round name+DOB privacy-preserving protocol."""

    cfg = cfg or MultiAttributeConfig()
    artifacts = prepare_party_b_multi_offline(
        records_b,
        cfg=cfg,
        k_mode=k_mode,
        random_state=random_state,
    )
    first_req, state = prepare_party_a_multi_query(query, artifacts, cfg=cfg)
    enc_centroid_scores = compare_multi_to_centroids(first_req, artifacts.centroids)
    second_req, selected_cluster = choose_multi_cluster(enc_centroid_scores, state)
    enc_match_scores = column_wise_multi_matching(
        artifacts.cluster_matrix,
        second_req,
        first_req.public_context_bytes,
        tau=cfg.similarity_threshold,
    )
    catch, checked, first_positive = decrypt_multi_match(
        enc_match_scores,
        state,
        early_stop=early_stop,
    )
    return MultiProtocolRun(
        catch=catch,
        selected_cluster=selected_cluster,
        checked_columns=checked,
        first_positive_column=first_positive,
        artifacts=artifacts,
    )
