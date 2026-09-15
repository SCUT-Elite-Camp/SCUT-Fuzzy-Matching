# Multi-Attribute Matching V2: declarative attribute schema

V2 generalizes the hard-coded name+DOB prototype of [V1](MULTI_ATTRIBUTE_V1.md) into a
declarative schema. **The two-round protocol structure is unchanged** — round 1 still
sends one cluster-dim ciphertext to pick a cluster, round 2 still sends one match-dim
ciphertext plus a one-hot selector and judges column by column. Only the *width* of those
vectors and the way they are assembled become configurable.

V1's closing note ("extend the same encoder registry to gender/address/country and then
replace the fixed 200/50 assumptions") is what this implements.

## Core score (unchanged from V1)

```
S = Σ wᵢ · simᵢ(attributeᵢ)
```

Each attribute block is L2-normalized, scaled by `sqrt(wᵢ)`, then concatenated. The CKKS
dot product expands into exactly the weighted sum above, because blocks don't overlap.

## The schema

```python
from multi_attribute import AttributeSchema

schema = AttributeSchema.from_dict({
    "similarity_threshold": 0.6,
    "name_attribute": "name",
    "standardize_cluster": True,
    "attributes": [
        {"name": "name",  "kind": "fuzzy_text", "weight": 0.30,
         "cluster_dim": 200, "match_dim": 50},
        {"name": "dob",   "kind": "date",       "weight": 0.20,
         "blocks": 2, "buckets_per_block": 128, "seed": 20260904,
         "day_first": True, "on_invalid": "missing"},
        {"name": "phone", "kind": "exact",      "weight": 0.20,
         "blocks": 2, "buckets_per_block": 128, "normalize": "digits"},
    ],
})
```

`cluster_dim` and `match_dim` are now **derived** from the attribute list rather than being
constants. JSON carries attributes as a **list**, because layout order is part of the
payload and `sort_keys=True` would destroy dict key order.

`AttributeSpec.params` holds kind-specific parameters in a dict, so registering a new kind
with new parameter names never requires touching the schema class.

### Compatibility

Every public function still takes `cfg`, with the same `None` default. `cfg` now accepts
`None` / `AttributeSchema` / `MultiAttributeConfig` / a plain mapping, resolved by
`resolve_schema()`. So `cfg=MultiAttributeConfig(...)` keeps working, and V1's documented
demo values are reproduced bit-for-bit (`tests/test_multi_attribute_schema.py -k legacy`
asserts `assert_array_equal` against the original implementation).

## Built-in kinds

| kind | params | (cluster, match) dims | implementation |
|---|---|---|---|
| `fuzzy_text` | `cluster_dim` (≤200), `match_dim` | `(cd, md)` | `minhash.encoder.batch_encode` + `l2_normalize` |
| `exact` | `blocks`, `buckets_per_block`, `seed`, `normalize` | `(D, D)`, `D = blocks·buckets` | `hashing.exact_hash_vector` |
| `date` | as `exact`, plus `day_first`, `on_invalid` | `(D, D)` | `normalize_dob` + `exact_hash_vector` |

Aliases: `text`/`name` → `fuzzy_text`, `categorical`/`category` → `exact`, `dob` → `date`.
Kind names are canonicalized on construction, so JSON round-trips are stable.

`normalize` modes for `exact`: `strip` | `casefold` | `digits`. **`digits`** strips all
non-digit characters, so `+1 (555) 010-0199` and `15550100199` land in the same bucket —
that is what makes phone numbers usable as an exact attribute.

## Two deliberate policy knobs on `date`

Both default to the safe behavior; the permissive behavior must be declared.

- **`day_first`** (default `False`). `05/03/2001` is genuinely ambiguous, and guessing
  wrong yields a syntactically valid but semantically wrong date — worse than an error. So
  `DD/MM/YYYY` / `DD-MM-YYYY` / `DD.MM.YYYY` are only parsed when the caller declares
  `day_first: true`.
- **`on_invalid`** (default `"raise"`). Unparseable values fail closed by default. Real
  dirty data needs the escape hatch: **FEBRL 4b contains 64/5000 (1.3%) invalid dates**
  such as `19450493`, so `config/examples/febrl_multi_attribute.json` sets
  `"on_invalid": "missing"`, which treats them as missing (all-zero block, zero
  contribution). `demo_multi_attribute_dataset.py` audits and prints the count so the
  zeroing is never silent.

## One intentional behavior change vs V1

`fuzzy_text` treats `None` or blank as **missing → all-zero block**. V1 routed an empty
name through MinHash's `"<empty>"` sentinel and got a unit vector back, i.e. an empty name
collected its full weight. The new behavior matches the existing DOB convention. This is
pinned by a test so it can't regress into an accident.

## Registering your own kind

`registry.register_encoder` is the extension point. `encode_attribute` asserts the returned
blocks match the declared dims, which is what makes "declare a kind, get a correct
protocol" trustworthy:

```python
from multi_attribute import AttributeBlock, register_encoder

def _dims(spec):  return int(spec.params.get("dim", 8)), int(spec.params.get("dim", 8))
def _validate(spec): ...
def _encode(values, spec):  # -> AttributeBlock with (n, cluster_dim) / (n, match_dim)
    ...

register_encoder("my_kind", dims=_dims, encode=_encode, validate=_validate)
```

If a kind declares dims that don't match what it encodes, you get a `RegistryError` naming
the attribute — not a silently misaligned vector.

## Fail-closed schema fingerprint

With configurability comes a new silent-error risk: two schemas with the *same total
dimensions* but a different layout produce plausible-looking wrong scores.
`prepare_party_a_multi_query` therefore compares fingerprints and raises on mismatch,
printing both sides. `run_multi_attribute_protocol` does the same when you pass in
pre-built `artifacts`.

## Validation

`AttributeSchema.__post_init__` rejects: empty attribute list; non-unique or
non-identifier names; unregistered kinds; unknown `params` keys (typo protection);
non-finite or negative weights; weights not summing to 1.0; all-zero weights; a
`similarity_threshold` outside `[0, 1]`; a `name_attribute` that isn't a declared
`fuzzy_text` attribute; and `cluster_dim`/`match_dim` exceeding `CKKS_SLOT_LIMIT` (4096).
`from_dict` also rejects unknown top-level keys.

## Data layer

`multi_attribute/dataset.py` is a plain CSV → `AttributeRecord` bridge. It deliberately does
**not** touch `data_pipeline/`: that contract (`NameRecord` / `PreparedName`) is name-only
by design, and V1 states the multi-attribute module doesn't change the production path.

- `load_attribute_records` / `load_attribute_queries` — project `{attribute: column}` (or
  `{attribute: [columns...]}`), join multi-column attributes with a space, **omit the key
  entirely when all its columns are empty** (→ zero block), and raise on unknown column
  names. CSV headers are whitespace-stripped, which FEBRL needs.
- `apply_labels` — attach a separate `labels.csv`.
- `load_febrl_pair` — derive ground truth from `rec_id` (`rec-<N>-org` ↔ `rec-<N>-dup-<M>`).
  `limit` bounds **queries**, `database_limit` bounds the database: 4a's `rec_id` is not
  ascending, so applying one limit to both pushes nearly every duplicate's original out of
  the database. Orphan duplicates are **dropped**, not labelled negative — they are false
  negatives.

## Run

```powershell
python scripts/fetch_dataset.py --dataset all
python scripts/generate_synthetic_attributes.py --records 5000 --seed 42

python scripts/demo_multi_attribute_dataset.py --config config/examples/febrl_multi_attribute.json --db-limit 0 --limit 20
python scripts/demo_multi_attribute_dataset.py --config config/examples/multi_attribute_schema.json --db-limit 500 --limit 20

python -m pytest tests/test_multi_attribute_schema.py tests/test_multi_attribute_dataset.py -q
```

Both demo runs also write a JSON + CSV report to
`artifacts/demo/multi_attribute_dataset/` (override with `--output-dir`), the same layout
as `demo_ncvr_matches.py` and `demo_sage_cross_script.py`. One CSV row per query, carrying
the true-match score, the impostor score, the per-attribute breakdown (`sim_<attribute>`
columns), and the encrypted verdict. That CSV is the file to plot when calibrating tau —
the terminal report scrolls away, the CSV does not. `--no-encrypted` skips the CKKS rounds
and leaves the encrypted columns blank, which makes the plaintext calibration sweep cheap.

`fetch_dataset.py` and `generate_synthetic_attributes.py` are the **only** way to populate
`dataset/`, because that directory is gitignored — a fresh clone has nothing in it. Deleting
them would permanently break this demo and the three existence-gated real-data tests, so
they stay visible and documented rather than hidden.

## What the protocol can and cannot report

The protocol returns only three things: the **sign** of the threshold test, the **selected
cluster index**, and the **column index** reached. It never returns the id of the matched
record, and a CATCH means "some record in the database scores above tau" — *not* "the right
record was found". So the encrypted path can only yield a catch-vs-label agreement rate;
precision and recall are not computable from protocol output.

Two limits on that agreement rate, both measured on FEBRL (40 queries):

- **Cluster recall 34/40 = 85%.** Round 2 examines exactly one cluster, so a true match
  sitting in another cluster is never checked. This is a hard ceiling, not a bug.
- **Threshold overlap.** FEBRL's most heavily damaged duplicates score 0.542 while the
  best impostor scores 0.579 — they overlap, so no tau separates them. That is an
  information problem to fix with more attributes or better weights, not a tuning problem.

The demo prints the observed calibration window (lowest true-match / highest impostor) and
says whether the configured `similarity_threshold` falls inside it. **Recalibrate on your
own data**; the values in `config/examples/` are calibration results for those two datasets,
not defaults to copy.

## Added files

- `multi_attribute/schema.py`, `registry.py`, `kinds.py`, `hashing.py`, `dataset.py`
- `scripts/fetch_dataset.py`, `scripts/generate_synthetic_attributes.py`,
  `scripts/demo_multi_attribute_dataset.py`
- `config/examples/febrl_multi_attribute.json`, `config/examples/multi_attribute_schema.json`
- `tests/test_multi_attribute_schema.py`, `tests/test_multi_attribute_dataset.py`
