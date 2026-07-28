"""Batch protocol wire-boundary helpers.

Party A owns ciphertext objects bound to its private context.  Before a request
crosses to Party B, every ciphertext is serialized and later rebound only to
Party B's public context.
"""

import tenseal as ts

from ckks.batching import serialize_feature_batch
from protocol.types import (
    BatchFirstRoundRequest,
    BatchSecondRoundRequest,
    TiledFirstRoundRequest,
    TiledSecondRoundRequest,
)


def _serialize_tiled_ciphertexts(
    ciphertexts,
    *,
    expected_count: int | None,
    active_slots: int,
) -> list[bytes]:
    if expected_count is not None and len(ciphertexts) != expected_count:
        raise ValueError(
            f"ciphertext count mismatch: expected {expected_count}, "
            f"got {len(ciphertexts)}"
        )
    if expected_count is None and len(ciphertexts) <= 0:
        raise ValueError("ciphertext collection must not be empty")

    for index, ciphertext in enumerate(ciphertexts):
        if isinstance(ciphertext, ts.CKKSVector):
            if ciphertext.size() != active_slots:
                raise ValueError(
                    f"ciphertext[{index}] size {ciphertext.size()} != "
                    f"active_slots {active_slots}"
                )
        elif not isinstance(ciphertext, bytes):
            raise ValueError(
                f"ciphertext[{index}] must be CKKSVector or bytes, "
                f"got {type(ciphertext)}"
            )
    return serialize_feature_batch(ciphertexts)


def serialize_batch_first_round_request(
    request: BatchFirstRoundRequest,
) -> BatchFirstRoundRequest:
    """Return a first-round request whose ciphertext payload is bytes-only."""
    return BatchFirstRoundRequest(
        public_context_bytes=request.public_context_bytes,
        encrypted_query_200=serialize_feature_batch(request.encrypted_query_200),
        batch_size=request.batch_size,
    )


def serialize_batch_second_round_request(
    request: BatchSecondRoundRequest,
) -> BatchSecondRoundRequest:
    """Return a second-round request whose ciphertext payload is bytes-only."""
    return BatchSecondRoundRequest(
        encrypted_query_50=serialize_feature_batch(request.encrypted_query_50),
        encrypted_selectors=serialize_feature_batch(request.encrypted_selectors),
        batch_size=request.batch_size,
    )


def serialize_tiled_first_round_request(
    request: TiledFirstRoundRequest,
) -> TiledFirstRoundRequest:
    """Return a V2 first-round request whose ciphertext payload is bytes-only."""
    if not isinstance(request.public_context_bytes, bytes):
        raise ValueError("public_context_bytes must be bytes")
    return TiledFirstRoundRequest(
        public_context_bytes=request.public_context_bytes,
        encrypted_query_200=_serialize_tiled_ciphertexts(
            request.encrypted_query_200,
            expected_count=200,
            active_slots=request.layout.active_slots,
        ),
        layout=request.layout,
    )


def serialize_tiled_second_round_request(
    request: TiledSecondRoundRequest,
) -> TiledSecondRoundRequest:
    """Return a V2 second-round request whose ciphertext payload is bytes-only."""
    return TiledSecondRoundRequest(
        encrypted_query_50=_serialize_tiled_ciphertexts(
            request.encrypted_query_50,
            expected_count=50,
            active_slots=request.layout.active_slots,
        ),
        encrypted_selectors=_serialize_tiled_ciphertexts(
            request.encrypted_selectors,
            expected_count=None,
            active_slots=request.layout.active_slots,
        ),
        layout=request.layout,
    )
