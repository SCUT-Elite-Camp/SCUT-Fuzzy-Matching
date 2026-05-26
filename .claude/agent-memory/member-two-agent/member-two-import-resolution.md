---
name: member-two-import-resolution
description: How import path resolution works when party_a/ code needs to reach both baseline_Integration/ and baseline/minhash&encoding/
metadata:
  type: reference
---

The path setup in `party_a/local_prep.py`:
```python
_INTEGRATION_DIR = Path(__file__).resolve().parent.parent  # baseline_Integration/
sys.path.insert(0, str(_INTEGRATION_DIR))  # priority for config, ckks, protocol

_BASELINE_DIR = _INTEGRATION_DIR.parent / "baseline" / "minhash&encoding"
sys.path.append(str(_BASELINE_DIR))  # for minhash.encoder, preprocessing.normalizer
```

`_INTEGRATION_DIR` is inserted at position 0, giving it priority over `_BASELINE_DIR` (appended).
This means `config` resolves to `baseline_Integration/config/` (not `baseline/minhash&encoding/config/`).
Both config/params.py files have identical CKKS params, so this shadowing is intentional and safe.

The `preprocessing` module only exists in `baseline/minhash&encoding/`, so it resolves correctly.
The `minhash` package also lives only in `baseline/minhash&encoding/`, so it resolves correctly.
