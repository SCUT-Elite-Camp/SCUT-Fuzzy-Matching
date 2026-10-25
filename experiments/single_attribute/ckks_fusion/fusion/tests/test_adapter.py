"""适配层的最小测试：阻断行为、索引映射、固定形状不省计算。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fusion.adapter import CandidateEnvelope, FusionBlocked, adapt  # noqa: E402
from fusion.database_layout.build_candidate_blocks import build_blocks  # noqa: E402

N = 40
K = 4
MAX_SIZE = 10
DIM = 50
BATCH = 200


def make_layout(block_size: int = 10):
    assignments = np.repeat(np.arange(K), N // K)
    return build_blocks(n=N, block_size=block_size, strategy="cluster",
                        assignments=assignments, codes=None)


def common(layout):
    return dict(layout=layout, max_size=MAX_SIZE, clusters=K, dim=DIM, batch_size=BATCH)


def test_secret_share_is_blocked_with_reason():
    try:
        adapt(CandidateEnvelope("approx_psi", "secret_share", share_bits=1),
              **common(make_layout()))
    except FusionBlocked as exc:
        message = str(exc)
    else:
        raise AssertionError("secret_share should be blocked")
    assert "emp-ot" in message or "volePSI" in message
    assert "深度" in message


def test_plaintext_index_maps_to_blocks_and_leaks():
    layout = make_layout(block_size=10)
    survivors = [np.array([0, 1]), np.array([5]), np.array([], dtype=int)]
    batch = adapt(CandidateEnvelope("f", "plaintext_index", per_query=survivors),
                  **common(layout))
    assert batch.mode == "B_public_index"
    assert batch.leaks_candidate_index is True
    assert batch.blocks == 1          # 三条查询的候选都在第 0 个块里
    assert batch.processed_positions == 10


def test_fixed_shape_does_not_reduce_work():
    layout = make_layout(block_size=10)
    base = adapt(CandidateEnvelope("none", "fixed_shape"), **common(layout))
    cand = adapt(CandidateEnvelope("f", "fixed_shape"), **common(layout))
    assert cand.tiles == base.tiles   # 隐私约束下候选缩减不转化为计算缩减
    assert cand.leaks_candidate_index is False


def test_scattered_layout_is_worse_than_clustered():
    """候选集中在**同一个簇**内时，按簇聚集应显著减少需要的块数。

    这正是主实验的情形：幸存候选来自同一条查询选中的那个簇。
    """
    assignments = np.arange(N) % K          # 簇在原始顺序里交错
    clustered = build_blocks(n=N, block_size=2, strategy="cluster", assignments=assignments)
    stored = build_blocks(n=N, block_size=2, strategy="stored")
    # 只取簇 0 的成员（0,4,8,...,28 中的前 8 个）
    survivors = [np.array([0, 4, 8, 12, 16, 20, 24, 28])]
    env = CandidateEnvelope("f", "plaintext_index", per_query=survivors)
    b_clustered = adapt(env, **common(clustered))
    b_stored = adapt(env, **common(stored))
    assert b_stored.blocks > b_clustered.blocks, (
        f"stored={b_stored.blocks} clustered={b_clustered.blocks}"
    )


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(tests)} passed")
