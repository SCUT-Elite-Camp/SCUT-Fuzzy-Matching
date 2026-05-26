---
name: member-two-boundaries
description: Member Two's exact scope per spec lines 365-459 — what it owns vs what it must never touch
metadata:
  type: reference
---

Member Two (Step 2) owns these files in `baseline_Integration/`:
- `protocol/types.py` — shared type aliases and dataclasses (CipherBytes, EncryptedVector200, FirstRoundRequest, PartyALocalState, etc.)
- `ckks/context.py` — `create_ckks_context()` using params from `config.params`
- `ckks/keys.py` — `encrypt`, `decrypt`, `serialize_ct`, `deserialize_ct`, `serialize_public_context`
- `ckks/operations.py` — `dot_ct_pt`, `dot_ct_ct`, `add_plain`, `mul_plain`
- `party_a/local_prep.py` — `encode_query_vectors`, `create_ckks_context` (re-export), `encrypt_query_vectors`, `prepare_encrypted_query`

Never modify: Member One (offline_artifacts.py), Member Three, Member Four, Member Five code.

The baseline reference implementation at `baseline/ckks/` provides implementation patterns (context.py, keys.py) but hardcodes params — Member Two must import from `config.params` instead.
