"""End-to-end protocol orchestration for the single-query baseline."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from config.params import SIMILARITY_THRESHOLD
from party_a.local_prep import prepare_encrypted_query, prepare_tiled_query_batch
from party_a.online_querier import (
    check_encrypted_scores_debug,
    check_tiled_score_batch_debug,
    choose_cluster_and_build_request,
    choose_clusters_and_build_tiled_request,
)
from party_b.offline_prep import prepare_party_b_offline
from party_b.online_responder import (
    column_wise_matching,
    compare_tiled_batch_to_centroids,
    compare_to_centroids,
    tiled_batch_matching,
)
from protocol.transport import (
    serialize_tiled_first_round_request,
    serialize_tiled_second_round_request,
)
from protocol.types import (
    BatchClusterSelectionDebug,
    BatchMatchDebug,
    BatchMatchResult,
    ClusterSelectionDebug,
    MatchDebug,
    MatchResult,
    OfflineArtifacts,
)


@dataclass
class ProtocolRun:
    """Debug-friendly end-to-end result for tests and local experiments."""

    match_result: MatchResult
    artifacts: OfflineArtifacts
    cluster_debug: ClusterSelectionDebug
    match_debug: MatchDebug


@dataclass
class BatchProtocolRun:
    """Debug-friendly end-to-end result for batch protocol runs."""

    batch_match_result: BatchMatchResult
    artifacts: OfflineArtifacts
    batch_cluster_debug: BatchClusterSelectionDebug
    batch_match_debug: BatchMatchDebug


def run_single_query_protocol(
    names_b: Iterable[str],
    query_name: str,
    *,
    k_mode: str = "sqrt",
    random_state: int = 42,
    tau: float = SIMILARITY_THRESHOLD,
    early_stop: bool = True,
) -> ProtocolRun:
    """Run the documented baseline flow in one process."""

    artifacts = prepare_party_b_offline(
        names_b, k_mode=k_mode, random_state=random_state
    )
    first_round_request, party_a_state = prepare_encrypted_query(
        query_name,
        artifacts.scaler_mean,
        artifacts.scaler_scale,
    )
    encrypted_sim_scores = compare_to_centroids(
        first_round_request,
        artifacts.centroids,
    )
    second_round_request, cluster_debug = choose_cluster_and_build_request(
        encrypted_sim_scores,
        party_a_state,
        k=artifacts.centroids.shape[0],
    )
    encrypted_scores = column_wise_matching(
        artifacts.cluster_matrix,
        second_round_request,
        first_round_request.public_context_bytes,
        tau=tau,
    )
    match_result, match_debug = check_encrypted_scores_debug(
        encrypted_scores,
        party_a_state.secret_context,
        early_stop=early_stop,
    )
    return ProtocolRun(
        match_result=match_result,
        artifacts=artifacts,
        cluster_debug=cluster_debug,
        match_debug=match_debug,
    )


def run_batch_query_protocol(
    names_b: Iterable[str],
    query_names: Sequence[str],
    *,
    k_mode: str = "sqrt",
    random_state: int = 42,
    tau: float = SIMILARITY_THRESHOLD,
    early_stop: bool = True,
    serialize_communication: bool = True,
) -> BatchProtocolRun:
    """Run the V2 query-by-candidate tiled HE protocol end-to-end.

    The default enforces the real bytes boundary before either request reaches
    Party B.  ``False`` is retained only for local primitive comparisons.
    """
    artifacts = prepare_party_b_offline(
        names_b, k_mode=k_mode, random_state=random_state
    )
    first_round_request, party_a_state = prepare_tiled_query_batch(
        query_names,
        artifacts.scaler_mean,
        artifacts.scaler_scale,
    )
    first_round_for_b = (
        serialize_tiled_first_round_request(first_round_request)
        if serialize_communication
        else first_round_request
    )
    encrypted_sim_scores = compare_tiled_batch_to_centroids(
        first_round_for_b,
        artifacts.centroids,
        serialize_output=serialize_communication,
    )
    second_round_request, tiled_cluster_debug = choose_clusters_and_build_tiled_request(
        encrypted_sim_scores,
        party_a_state,
        k=artifacts.centroids.shape[0],
    )
    second_round_for_b = (
        serialize_tiled_second_round_request(second_round_request)
        if serialize_communication
        else second_round_request
    )
    encrypted_scores = tiled_batch_matching(
        artifacts.cluster_matrix,
        second_round_for_b,
        first_round_request.public_context_bytes,
        tau=tau,
        serialize_output=serialize_communication,
    )
    tiled_match_result, tiled_match_debug = check_tiled_score_batch_debug(
        encrypted_scores,
        party_a_state.secret_context,
        layout=party_a_state.layout,
        logical_width=artifacts.max_size,
        early_stop=early_stop,
    )
    return BatchProtocolRun(
        batch_match_result=BatchMatchResult(catches=tiled_match_result.catches),
        artifacts=artifacts,
        batch_cluster_debug=BatchClusterSelectionDebug(
            selected_clusters=tiled_cluster_debug.selected_clusters
        ),
        batch_match_debug=BatchMatchDebug(
            checked_columns=tiled_match_debug.logical_columns_checked,
            first_positive_columns=tiled_match_debug.first_positive_columns,
        ),
    )
