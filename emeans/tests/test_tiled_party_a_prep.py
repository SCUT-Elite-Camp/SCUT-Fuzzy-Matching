"""Tests for V2 tiled Party A local prep and wire serialization."""

import numpy as np
import pytest
import tenseal as ts

from ckks.context import create_ckks_context
from ckks.tiling import make_slot_tile_layout
from party_a.local_prep import (
    encode_query_batch,
    prepare_tiled_query_batch,
)
from protocol.transport import (
    serialize_tiled_first_round_request,
)
from protocol.types import TiledFirstRoundRequest, TiledPartyALocalState


@pytest.fixture(scope="module")
def mock_scaler():
    np.random.seed(42)
    mean = np.random.uniform(-1.0, 1.0, size=200)
    scale = np.random.uniform(0.5, 2.0, size=200)
    return mean, scale


@pytest.fixture(scope="module")
def ckks_ctx():
    return create_ckks_context()


def test_prepare_tiled_query_batch_shapes(mock_scaler, ckks_ctx):
    mean, scale = mock_scaler
    names = ["JOHN SMITH", "ALICE BROWN"]

    req, state = prepare_tiled_query_batch(names, mean, scale, context=ckks_ctx)

    assert isinstance(req, TiledFirstRoundRequest)
    assert isinstance(state, TiledPartyALocalState)

    layout = make_slot_tile_layout(2)
    assert req.layout == layout
    assert state.layout == layout

    assert len(req.encrypted_query_200) == 200
    for c in req.encrypted_query_200:
        assert c.size() == layout.active_slots

    assert len(state.encrypted_query_50) == 50
    for c in state.encrypted_query_50:
        assert c.size() == layout.active_slots

    assert isinstance(req.public_context_bytes, bytes)


def test_prepare_tiled_query_batch_m200(mock_scaler, ckks_ctx):
    mean, scale = mock_scaler
    names = [f"NAME_{i}" for i in range(200)]

    req, state = prepare_tiled_query_batch(names, mean, scale, context=ckks_ctx)

    layout = make_slot_tile_layout(200)
    assert req.layout == layout
    assert req.layout.tile_width == 20
    assert req.layout.active_slots == 4000

    assert len(req.encrypted_query_200) == 200
    assert all(c.size() == 4000 for c in req.encrypted_query_200)
    assert len(state.encrypted_query_50) == 50
    assert all(c.size() == 4000 for c in state.encrypted_query_50)


def test_tiled_encryption_recovers_query_values(mock_scaler, ckks_ctx):
    mean, scale = mock_scaler
    names = ["JOHN SMITH", "ALICE BROWN", "BOB WILSON"]

    q200_plain, _ = encode_query_batch(names, mean, scale)
    req, state = prepare_tiled_query_batch(names, mean, scale, context=ckks_ctx)
    layout = state.layout

    # Decrypt Q200 and verify q-major layout: each query occupies T consecutive
    # slots, repeated identically.
    decrypted = np.stack(
        [
            np.asarray(c.decrypt(ckks_ctx.secret_key()), dtype=np.float64)
            for c in req.encrypted_query_200
        ],
        axis=1,
    )
    assert decrypted.shape == (layout.active_slots, 200)

    for q in range(layout.batch_size):
        rows = decrypted[q * layout.tile_width : (q + 1) * layout.tile_width]
        # rows shape (T, 200); each row should equal the original q200 row.
        np.testing.assert_allclose(rows[0], q200_plain[q], atol=1e-4)


def test_tiled_first_round_request_bytes_only(mock_scaler, ckks_ctx):
    mean, scale = mock_scaler
    names = ["JOHN SMITH", "ALICE BROWN"]

    req, _ = prepare_tiled_query_batch(names, mean, scale, context=ckks_ctx)
    wire_req = serialize_tiled_first_round_request(req)

    assert isinstance(wire_req.public_context_bytes, bytes)
    assert all(isinstance(c, bytes) for c in wire_req.encrypted_query_200)
    assert wire_req.layout == req.layout


def test_public_context_cannot_decrypt(mock_scaler, ckks_ctx):
    mean, scale = mock_scaler
    names = ["JOHN SMITH"]

    req, _ = prepare_tiled_query_batch(names, mean, scale, context=ckks_ctx)
    pub = ts.Context.load(req.public_context_bytes)
    ct_bytes = req.encrypted_query_200[0].serialize()
    ct = ts.ckks_vector_from(pub, ct_bytes)

    # Public context must lack secret key; decrypt raises ValueError in TenSEAL
    with pytest.raises((RuntimeError, ValueError)):
        ct.decrypt()


def test_tiled_request_does_not_contain_secret_context(mock_scaler, ckks_ctx):
    mean, scale = mock_scaler
    names = ["JOHN SMITH"]

    req, _ = prepare_tiled_query_batch(names, mean, scale, context=ckks_ctx)
    # TiledFirstRoundRequest dataclass has no secret_context field
    assert not hasattr(req, "secret_context")
    assert req.public_context_bytes == req.public_context_bytes
