---
name: member-two-critical-constraints
description: Hard constraints that Member Two must never violate (from spec and general protocol rules)
metadata:
  type: feedback
---

Nine critical constraints for Member Two:

1. ALL CKKS parameters from `config.params` — `POLY_MODULUS_DEGREE`, `COEFF_MOD_BIT_SIZES`, `SCALE` — never hardcoded
2. `public_context_bytes` must NOT contain secret key — use `make_context_public()` before serialization
3. `FirstRoundRequest` only has `public_context_bytes` and `encrypted_query_200` — NEVER include `encrypted_query_50`
4. `query_50_norm` must NOT be standardized — only L2-normalized (no StandardScaler)
5. L2 normalization on 2D matrices BEFORE squeeze — then squeeze to 1D
6. Scaler MUST NOT be fit on A-side — use only the passed `scaler_mean` and `scaler_scale`
7. Single query only (m=1) — all vectors 1D after squeeze
8. Use TenSEAL as primary library — CKKSVector objects stored in-memory (not serialized bytes) in dataclasses
9. `encrypted_query_50` must never be sent in first round — it goes to PartyALocalState (local only)
