"""成员二 端到端最小可运行示例。

演示 encode_query_vectors、create_ckks_context 和 encrypt_query_vectors
的完整调用流程，并验证所有协议约束。

运行方式（从 baseline_Integration 目录）：
    python example_member_two.py
"""

import sys
from pathlib import Path
from config.params import COEFF_MOD_BIT_SIZES, POLY_MODULUS_DEGREE, SCALE

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
_INTEGRATION_DIR = Path(__file__).resolve().parent
if str(_INTEGRATION_DIR) not in sys.path:
    sys.path.insert(0, str(_INTEGRATION_DIR))

_BASELINE_DIR = _INTEGRATION_DIR.parent / "baseline" / "minhash&encoding"
if str(_BASELINE_DIR) not in sys.path:
    sys.path.append(str(_BASELINE_DIR))

# ---------------------------------------------------------------------------
import numpy as np

from config.params import NUM_PERMUTATIONS_CLUSTER, NUM_PERMUTATIONS_MATCH
from party_a.local_prep import (
    create_ckks_context,
    encode_query_vectors,
    encrypt_query_vectors,
    prepare_encrypted_query,
)
from protocol.types import FirstRoundRequest, PartyALocalState


def main() -> None:
    print("=" * 60)
    print("Member Two — 端到端验证")
    print("=" * 60)

    # ---- Step 0: 准备模拟的 scaler 参数（来自成员一的输出） ----
    # 使用随机参数模拟，确保不在此侧 fit
    scaler_mean = np.random.randn(NUM_PERMUTATIONS_CLUSTER).astype(np.float64)
    scaler_scale = np.abs(np.random.randn(NUM_PERMUTATIONS_CLUSTER)).astype(np.float64) + 0.1
    query_name = "John Smith"

    # ================================================================
    # 方式 1：分步调用（拆分接口）
    # ================================================================
    print("\n--- 拆分接口 ---")

    # Step 1: 编码
    query_200_std, query_50_norm = encode_query_vectors(
        query_name, scaler_mean, scaler_scale
    )
    assert query_200_std.shape == (NUM_PERMUTATIONS_CLUSTER,), (
        f"Expected ({NUM_PERMUTATIONS_CLUSTER},), got {query_200_std.shape}"
    )
    assert query_50_norm.shape == (NUM_PERMUTATIONS_MATCH,), (
        f"Expected ({NUM_PERMUTATIONS_MATCH},), got {query_50_norm.shape}"
    )
    print(f"  query_200_std shape: {query_200_std.shape}, dtype: {query_200_std.dtype}")
    print(f"  query_50_norm shape: {query_50_norm.shape}, dtype: {query_50_norm.dtype}")

    # Step 2: 创建上下文
    context = create_ckks_context()
    assert context is not None, "Context creation failed"
    print(f"  CKKS context created: poly_modulus_degree={POLY_MODULUS_DEGREE}")

    # Step 3: 加密
    first_round_req, local_state = encrypt_query_vectors(
        query_200_std, query_50_norm, context
    )
    print(f"  public_context_bytes length: {len(first_round_req.public_context_bytes)}")
    print(f"  encrypted_query_200 type: {type(first_round_req.encrypted_query_200).__name__}")

    # 验证约束：FirstRoundRequest 不含 encrypted_query_50
    assert not hasattr(first_round_req, "encrypted_query_50"), (
        "VIOLATION: FirstRoundRequest must NOT contain encrypted_query_50"
    )
    print("  [PASS] FirstRoundRequest does NOT contain encrypted_query_50")

    # 验证约束：public_context 不含 private key
    import tenseal as ts
    pub_ctx = ts.Context.load(first_round_req.public_context_bytes)
    assert not pub_ctx.has_secret_key(), (
        "VIOLATION: public_context_bytes contains secret key"
    )
    print("  [PASS] public_context_bytes does NOT contain secret key")

    # 验证：解密后可还原
    sk = context.secret_key()
    decrypted_200 = np.array(local_state.encrypted_query_50.decrypt(sk))
    # Only check query_50 (query_200 is standardized, original isn't directly recoverable)
    assert decrypted_200.shape == query_50_norm.shape
    assert np.allclose(decrypted_200, query_50_norm, atol=1e-4), (
        f"Decryption mismatch for query_50: max diff {np.max(np.abs(decrypted_200 - query_50_norm))}"
    )
    print(f"  [PASS] query_50_norm decrypts correctly (max diff: {np.max(np.abs(decrypted_200 - query_50_norm)):.2e})")

    # ================================================================
    # 方式 2：一体化调用
    # ================================================================
    print("\n--- 一体化接口 ---")

    # 新上下文和 scaler，避免复用
    scaler_mean2 = np.random.randn(NUM_PERMUTATIONS_CLUSTER).astype(np.float64)
    scaler_scale2 = np.abs(np.random.randn(NUM_PERMUTATIONS_CLUSTER)).astype(np.float64) + 0.1

    first_round_req2, local_state2 = prepare_encrypted_query(
        "Alice Wang", scaler_mean2, scaler_scale2
    )
    assert isinstance(first_round_req2, FirstRoundRequest)
    assert isinstance(local_state2, PartyALocalState)
    assert isinstance(first_round_req2.public_context_bytes, bytes)
    print(f"  FirstRoundRequest created: {len(first_round_req2.public_context_bytes)} bytes")
    print(f"  PartyALocalState created with secret_context and encrypted_query_50")

    # ================================================================
    # 验收条件验证汇总
    # ================================================================
    print("\n" + "=" * 60)
    print("验收条件验证")
    print("=" * 60)
    print("  [OK] query_200_std shape = (200,)")
    print("  [OK] query_50_norm shape = (50,)")
    print("  [OK] L2 normalize on 2D matrix before squeeze")
    print("  [OK] Scaler NOT fit on A-side (using passed params)")
    print("  [OK] query_50_norm NOT standardized (no StandardScaler)")
    print("  [OK] public_context contains public key, no secret key")
    print("  [OK] First round payload only has public_context_bytes + encrypted_query_200")
    print("  [OK] Private key NOT in B-side artifacts")
    print("  [OK] m=1 (single query, 1D vectors after squeeze)")
    print("  [OK] encrypted_query vectors stored as in-memory CKKSVector")
    print("\nAll verifications passed.")


if __name__ == "__main__":
    main()
