"""End-to-end protocol orchestration for the single-query baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from config.params import SIMILARITY_THRESHOLD
from party_a.local_prep import prepare_encrypted_query
from party_a.online_querier import (
    check_encrypted_scores_debug,
    choose_cluster_and_build_request,
)
from party_b.offline_prep import prepare_party_b_offline
from party_b.online_responder import compare_to_centroids, column_wise_matching
from protocol.types import (
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

