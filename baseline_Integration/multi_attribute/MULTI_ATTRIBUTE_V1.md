# Multi-Attribute Matching V1: Name + DOB

This patch adds a first runnable multi-attribute prototype without changing the existing name-only batch/tiled production path.

## Core score

For each record:

- name -> existing MinHash -> L2 normalize
- DOB -> normalized exact-category hashed one-hot vector
- concatenate `sqrt(weight) * attribute_vector`

Therefore the CKKS dot product computes approximately:

`S = w_name * S_name + w_dob * S_dob`

Default prototype configuration:

- `w_name = 0.70`
- `w_dob = 0.30`
- `tau = 0.80`
- cluster vector = 200 MinHash dims + 256 DOB dims = 456 dims
- match vector = 50 MinHash dims + 256 DOB dims = 306 dims

Missing DOB is encoded as all-zero and contributes no score.

## Added files

- `multi_attribute/model.py`
- `multi_attribute/config.py`
- `multi_attribute/encoder.py`
- `multi_attribute/protocol.py`
- `scripts/demo_multi_attribute.py`
- `tests/test_multi_attribute.py`

## Run

```powershell
python -m pip install -r requirements.txt
python scripts/demo_multi_attribute.py
python -m pytest tests/test_multi_attribute.py -q
```

## Expected plaintext reference behavior

With the default config:

- `Jon Smith + 2001/05/17` vs `John Smith + 2001-05-17`: about `0.889`, above tau -> match
- exact `John Smith` but wrong DOB: `0.700`, below tau -> reject

The plaintext score is only printed in the local demo for validation. The protocol path still sends ciphertext across the A/B boundary and returns only threshold-sign information for the final catch/no-catch decision.

## Next step

After the two-attribute version is validated on labeled data, extend the same encoder registry to gender/address/country and then replace the fixed 200/50 assumptions in tiled batching with dynamic feature counts.
