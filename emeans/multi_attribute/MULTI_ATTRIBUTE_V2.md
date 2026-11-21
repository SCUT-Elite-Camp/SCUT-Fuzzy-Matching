# Multi-Attribute Matching V2: declarative attribute schema

V2 generalizes the hard-coded name+DOB prototype of [V1](MULTI_ATTRIBUTE_V1.md) into a
declarative schema. **The two-round protocol structure is unchanged** — round 1 still
sends one cluster-dim ciphertext to pick a cluster, round 2 still sends one match-dim
ciphertext plus a one-hot selector and judges column by column. Only the *width* of those
vectors and the way they are assembled become configurable.

V1's closing note ("extend the same encoder registry to gender/address/country and then
replace the fixed 200/50 assumptions") is what this implements.

Three follow-ups on top of the schema layer are covered at the end of this document, all
of them **measured rather than asserted**: [weights and tau are derived from labelled
data](#calibrating-weights-and-tau-from-labelled-data) instead of hand-written,
[round 2 probes](#round-2-probe-policy-and-k-selection) more than one cluster, and the
[k policy](#round-2-probe-policy-and-k-selection) is picked from a measured cost curve.
Read the numbers before changing any of the three defaults.

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
| `exact` | `blocks`, `buckets_per_block`, `seed`, `normalize`, `tolerate` | `(D, D)`, `D = blocks·buckets` | `hashing.exact_hash_vector` |
| `date` | as `exact`, plus `day_first`, `on_invalid` | `(D, D)` | `normalize_dob` + `exact_hash_vector` |

Aliases: `text`/`name` → `fuzzy_text`, `categorical`/`category` → `exact`, `dob` → `date`.
Kind names are canonicalized on construction, so JSON round-trips are stable.

`normalize` modes for `exact`: `strip` | `casefold` | `digits`. **`digits`** strips all
non-digit characters, so `+1 (555) 010-0199` and `15550100199` land in the same bucket —
that is what makes phone numbers usable as an exact attribute.

### `tolerate`: aligned-character encoding (opt-in, measured net-negative)

`exact` and `date` default to a **cliff**: the whole string goes into one hash, so one
changed character takes the similarity from 1.0 straight to 0.0. `tolerate` replaces that
with an **aligned per-position** encoding — similarity becomes the fraction of positions
carrying the same character, so `5304218` vs `5304219` is 6/7 instead of 0.

```json
{"name": "ssn", "kind": "exact", "weight": 0.15, "normalize": "digits",
 "tolerate": {"max_length": 8, "buckets": 64}}
```

`max_length` and `buckets` are **mandatory and explicit**: the dim is
`max_length · buckets`, and it must not depend on the batch (Party A encodes one query,
Party B encodes the whole database — a batch-derived width would make the two sides
disagree). `blocks`/`buckets_per_block` are rejected alongside `tolerate` rather than
silently ignored. Values longer than `max_length` are truncated, which is lossy: set
`max_length` to the longest value present in your data.

**It is off by default because it measured worse, not because it is untested.** FEBRL
500 queries / 5000 database rows, V1's hand-written weights:

| encoding | combined AUC | weakest true match | median impostor | false-positive rate at recall 1.0 |
|---|---|---|---|---|
| cliff (default) | 0.9994 | 0.5391 | 0.5036 | **14.6%** |
| `tolerate` L7/B64 | 0.9934 | 0.5605 | 0.6339 | 95.0% |
| `tolerate` L10/B32 | 0.9945 | 0.5605 | 0.6270 | 94.8% |
| `tolerate` L16/B16 | 0.9923 | 0.5605 | 0.6410 | 94.8% |

All four rows go through `calibration.attribute_score_samples` + `roc_auc`, so the cliff row
is the same 0.9994 as the calibration table above; the false-positive rate is measured at
each variant's own `recall_first` threshold.

The tell is the **median impostor** column, not the max. Partial credit raises the weakest
true match by 0.02 — and raises the *typical* impostor by 0.13, pushing the bulk of the
impostor distribution up into the positive range. Attributes like SSN and postcode are
supposed to be "all of it or none of it as evidence"; awarding partial credit for
`5304218` vs `5304219` hands that credit to unrelated records too. A one-character error
in an SSN is far more likely to be a different person than a typo.

It remains available because the measurement is dataset-specific — for a fixed-width
identifier corrupted by a known single-character OCR error channel, the tradeoff could
reverse. Measure the median impostor before enabling it.

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

## Calibrating weights and tau from labelled data

`similarity_threshold` and the per-attribute weights used to be numbers someone wrote down
(`0.3/0.2/0.2/0.15/0.1/0.05` and `tau=0.6` for FEBRL, with a comment saying they came from
measurement). `multi_attribute/calibration.py` derives both from labelled pairs instead:

```powershell
python scripts/calibrate_multi_attribute.py --config config/examples/febrl_multi_attribute.json --output config/examples/febrl_multi_attribute.calibrated.json
python scripts/demo_multi_attribute_dataset.py --config config/examples/febrl_multi_attribute.calibrated.json --db-limit 0 --limit 20
```

The output is a **drop-in job file** — same keys, calibrated `schema` block, plus a
`_calibration` report — so a calibrated run is reproducible without re-running the fit.

### Why the obvious rule is wrong

Weighting each attribute by its own `AUC − 0.5` is the first thing anyone tries and it is
**measurably worse**. FEBRL, 500 queries:

| rule | combined AUC | weakest true match | strongest impostor |
|---|---|---|---|
| hand-written (V1) | 0.9994 | 0.539 | 0.618 |
| proportional to per-attribute AUC | 0.9824 | **0.214** | 0.609 |
| separability (default) | 0.9997 | 0.513 | 0.598 |

`ssn` has the highest standalone AUC (0.911), so the proportional rule puts 44.7% of the
weight on it — but `ssn` is nearly binary. The moment FEBRL damages one duplicate's SSN,
that query's true-match score collapses to zero and the weakest true match falls from
0.539 to 0.214. **A high AUC can just mean "usually redundant".** The rule is kept as
`--weight-policy auc` so the claim stays falsifiable, but it is not the default.

### What the default does

`--weight-policy separability` runs coordinate ascent directly on the *combined* score
against a **soft-AUC / pairwise-sigmoid** objective:

```
objective(w) = mean_i  sigmoid( (score_i(true match) − score_i(hardest impostor)) / T )
```

That is exactly the quantity the threshold test needs — the probability that a true match
outranks its own strongest impostor — and small `T` concentrates on the hardest pairs, so
weights are chosen *jointly* and complementary attributes earn their share. Candidates are
restricted to a per-attribute top-K union (`_CandidateSet`) so it never materialises an
`(n_query, n_database, n_attribute)` tensor.

Negatives are each query's **strongest impostor**, not random database records: those are
what tau actually has to reject, and on FEBRL random records are beaten by essentially
every true match, which would report an uninformative AUC of 1.0.

Threshold criteria: `recall_first` (default — sits just below the weakest true match),
`youden`, `target_recall`.

### The fit generalizes (and the temperature is load-bearing)

A weight vector fitted on the same queries it is scored on proves nothing, so the same
objective was fitted on half the labelled queries and scored on the other half (250/250,
seed 0):

| weights | held-out AUC | held-out weakest true match | held-out strongest impostor |
|---|---|---|---|
| hand-written (V1) | 0.9994 | 0.5417 | 0.6182 |
| fitted, `T=0.05` | 0.9999 | 0.5323 | **0.5724** |
| fitted, `T=0.15` | 0.9922 | 0.2784 | 0.4839 |

The fitted weights **narrow the interleave** out of sample: the strongest impostor drops
0.618 → 0.572 while the weakest true match moves only 0.542 → 0.532. That is the trade the
threshold actually wants.

`T=0.15` is in the table to show the temperature is not a free knob. A large `T` makes the
sigmoid nearly linear, so the objective rewards average separation instead of the hard
pairs, and it collapses the fit onto four attributes — worst held-out weakest-true-match
0.278. Small `T` is what keeps the optimizer on the queries that decide recall.

### Reading the separation verdict

Three states, and the middle one is the one that gets misread:

- **separable** — `highest_impostor < lowest_true_match`; one tau covers everything.
- **overlap** — some query loses to *its own* strongest impostor. No tau fixes those;
  more attributes or better field quality is the fix.
- **ranges interleave** — every query beats its own impostors, but the weakest true match
  still sits below the strongest impostor. FEBRL is exactly this (0.5132 vs 0.5982). No
  global tau gets both recall and a clean boundary; `recall_first` spends false positives
  to buy recall, and the report prints the exchange rate rather than hiding it.

## Round-2 probe policy and k selection

### Multi-cluster probing

Round 2 used to examine exactly one cluster. The true match sitting in another cluster was
therefore never checked — and since round 1's job is a k-way classification over centroids,
that happens for ~12% of FEBRL queries. The fix is not a smarter single choice; it is
`probes=`:

```powershell
python scripts/demo_multi_attribute_dataset.py --config ...                # default: 3
python scripts/demo_multi_attribute_dataset.py --config ... --probes 1     # V1 behavior
python scripts/demo_multi_attribute_dataset.py --config ... --probes all   # exhaustive
```

A sends one one-hot selector per probed cluster, ordered by **descending centroid score**.
That ordering is what makes probing cheap: the true cluster's mean rank is only ~0.15, so
early stop fires in the first cluster or two for most queries.

**The default is 3, not exhaustive.** Exhaustive reaches cluster recall 1.0000, but its
saving is built entirely on early stop, and early stop needs a *hitting column* to stop at.
A query whose record is absent from the database never hits one, so exhaustive degenerates
into a full linear scan for exactly that query class (measured: with 500 database rows, all
20 blank queries scanned all 500 columns). A fixed probe count gives that class a hard cost
ceiling — at most the columns of 3 clusters. The trade is cluster recall 1.0000 → 0.96,
made deliberately: without a cost ceiling, exhaustive is not deployable.

Measured on FEBRL, 100 queries / 5000 database rows / k=71 / calibrated tau=0.513, one
encrypted run per row, only `--probes` changed:

| policy | cluster recall | encrypted agreement | mean columns | worst-case columns |
|---|---|---|---|---|
| top-1 | 0.87 | 0.87 | 41.5 | 105 |
| **top-3 (default)** | **0.96** | **0.96** | **53.3** | **207** |
| exhaustive | 1.0000 | — | — | **5000 (= full DB)** |

Cost is round-2 scan columns (ciphertext-ciphertext dots — the expensive resource); a full
linear scan is 5000. An earlier 500-query sweep at other k (seeds 7/13/42 averaged,
**deployment** accounting: B computes every column of every probed cluster up to and
including the true one) agrees in direction:

| k | policy | cluster recall | mean columns | p95 | selectors |
|---|---|---|---|---|---|
| 100 | top-1 | 0.887 | 53 | — | 1 |
| 100 | top-8 | 0.988 | 406 | — | 8 |
| 100 | exhaustive | 1.0000 | 72 | 163 | 100 |
| 150 | top-1 | 0.885 | 36 | — | 1 |
| 150 | top-8 | 0.987 | 271 | — | 8 |
| 150 | exhaustive | 1.0000 | 49 | 114 | 150 |

The deployment rule pays more than the demo's lazy generator, which stops mid-cluster when
A's early stop fires — a 40-query encrypted run at `--probes all --k-mode auto` (k=99)
reports mean 34.2 columns where the k=100 row above says 72. The gap is the tail of the true
cluster after the match, which the lazy rule never builds.

**Top-m can never reach 1.0** — it stops at 0.987/0.988 and the residue is a genuine tail of
the true-cluster rank, because top-m always pays for m full clusters while exhaustive stops
at the true one. But exhaustive buys that last 1.2% with an unbounded worst case, which is
the wrong trade.

**This policy is opt-in per call site.** `probes=1` (pure argmax) remains the library
default in `multi_attribute/protocol.py` so existing callers and tests are unchanged.
`DEFAULT_MULTI_ATTRIBUTE_PROBES = 3` is the demo's default, and the name-only path
(`party_a` / `party_b`) has no probing logic at all — it stays top-1.

**The honest limit:** probing can only early-stop on a query that *hits*. The demo reports
the mean scan columns for hit and blank queries separately for exactly this reason. The
sublinear cost is a property of matching queries, not of the protocol; for the rest, top-3
is a ceiling rather than a speedup.

### k policy

`--k-mode` accepts `sqrt` (k = ⌊√n⌋, the V1 baseline), `log2` (k = ⌊log₂n⌋), `auto`
(k = ⌊1.4·√n⌋), an integer, or `fixed:<k>`. `auto` is the default in the tables above.

`log2` loses decisively and it is not close: at n=5000 it gives k=12, i.e. ~417 records per
cluster, so every probed cluster is enormous (621 mean columns against sqrt's 105). Its only
advantage — 12 selectors instead of 70 — is dwarfed by the column blowup.

Above sqrt the columns keep falling roughly as 1/k while the selector payload grows as k,
which puts the optimum near 1.4·√n across a wide range of network-vs-compute weightings.
The curve is flat enough between k=70 and k=150 that anything in that band is defensible.

Cluster recall does **not** distinguish these k's — top-1 recall is 0.84–0.89 in every row.
What k changes is how much a miss costs once probing is exhaustive.

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

# derive weights + tau from labelled pairs, then run the calibrated job
python scripts/calibrate_multi_attribute.py --config config/examples/febrl_multi_attribute.json --output config/examples/febrl_multi_attribute.calibrated.json
python scripts/demo_multi_attribute_dataset.py --config config/examples/febrl_multi_attribute.calibrated.json --db-limit 0 --limit 20 --k-mode auto

python scripts/demo_multi_attribute_dataset.py --config config/examples/multi_attribute_schema.json --db-limit 500 --limit 20

python -m pytest tests/test_multi_attribute_schema.py tests/test_multi_attribute_dataset.py tests/test_multi_attribute_probing.py -q
```

Both demo runs also write a JSON + CSV report to
`artifacts/demo/multi_attribute_dataset/` (override with `--output-dir`), the same layout
as `demo_ncvr_matches.py` and `demo_sage_cross_script.py`. One CSV row per query, carrying the
**true-match score `expected_score`** (looked up by the true id — *not* `top1_score`, which is
the plaintext argmax and silently becomes the impostor's score if argmax lands on one), `tau`,
`should_catch`, `expected_score_margin`, the impostor score, the per-attribute breakdown
(`sim_<attribute>` columns), the encrypted verdict, `enc_miss_cause` attributing each miss
(`round1_cluster` / `boundary` / `unexplained`), and the probe bookkeeping —
`enc_probed_clusters`, `enc_true_match_cluster_rank`, `enc_cluster_hit_top1_only`. That CSV is the file to plot when calibrating
tau — the terminal report scrolls away, the CSV does not. `--no-encrypted` skips the CKKS rounds
and leaves the encrypted columns blank, which makes the plaintext calibration sweep cheap.

`fetch_dataset.py` and `generate_synthetic_attributes.py` are the **only** way to populate
`dataset/`, because that directory is gitignored — a fresh clone has nothing in it. Deleting
them would permanently break this demo and the three existence-gated real-data tests, so
they stay visible and documented rather than hidden.

## What the protocol can and cannot report

The protocol returns only three things: the **sign** of the threshold test, the **selected
cluster index**, and the **column index** reached. It never returns the id of the matched
record, and a CATCH means "some record in the selected cluster scores above tau" — *not* "the
right record was found". So what the encrypted path yields is a **detection rate**, not
identification precision/recall.

The demo reports that rate as

```
should_catch = plaintext score of the TRUE MATCH > tau      (strict >, same as round 2)
recall       = #{catch and should_catch} / #{should_catch}
```

The denominator is `should_catch`, **not** "the label is true". A duplicate perturbed below
tau is not supposed to catch under the protocol's own criterion, so counting it in the
denominator charges the recall rate with a debt the protocol never took on. Under this
definition "an unperturbed query (plaintext score exactly 1.0) must catch" becomes an
assertable invariant, and the demo asserts it.

Three limits, all measured on FEBRL:

- **Round 1 is a k-way classification, and it used to be the whole gap.** A single probe
  scans one cluster, so a true match sitting anywhere else is never checked: on the original
  100-query run 13 positives were lost that way, with `boundary` 0 / `unexplained` 0 — the
  *entire* gap (the 500-query top-1 miss rate is 11%, same story). This
  is not a bug, and it is *independent of tau*, so no threshold can move it. It was fixed
  where it lives, in the probe policy: top-3 recovers 9 of the 13, and the residual 4 are
  still the same failure mode. What remains, and what the report leads with, is
  **top-1-only cluster recall** (0.84–0.89 on FEBRL) — the quality of round 1's own decision,
  as opposed to how much round 2 pays to insure against it. Neither number is tau-dependent.
  Raising `probes` saturates the first by construction (exhaustive reaches 1.0000), so the
  report says so and points at the second — and caps `probes` at 3 anyway, because exhaustive's
  worst case is a full-database scan for every query whose record is absent.
- **The calibration window interleaves.** FEBRL's weakest true match (0.5132 over 500
  queries) sits below the strongest impostor (0.5982). Every query beats *its own* impostors,
  but no single global tau gets both full recall and a clean boundary — that is an information
  problem to fix with more attributes or better weights, not a tuning problem. The report
  separates this from the sharper failure (`overlap`: a query that loses to its own strongest
  impostor, which no tau can save) because the two look similar in a histogram and have
  different fixes.
- **The denominator itself contains tau.** This is the one to watch. Raising tau deletes the
  *hardest* queries from the denominator and pushes the rate toward 100% **without fixing
  anything**: on FEBRL, tau 0.60 → 0.90 takes the denominator from 99 to 50 and the printed
  rate to 100.0%, while the cluster counts do not move at all. The miss count is
  tau-dependent too — it fell 12 → 0 in that same move, because its denominator is the
  should-catch set, not because anything was fixed — so the report labels it `tau-DEPENDENT`
  and prints the tau-free cluster counts next to it. A recall figure whose denominator you can
  shrink is not a target. So every ratio is printed with its raw `n/d`, the old all-positives
  recall is kept alongside for contrast, and a tau guard fires when the threshold sits above
  the impostor ceiling while still costing positives.

`top-1-only cluster recall` and the calibration window (lowest true-match / highest impostor)
are the numbers in the report that tau cannot move. Judge a run by those.

The demo prints whether the configured `similarity_threshold` falls inside the observed
calibration window. **Recalibrate on your own data**; the values in `config/examples/` are
calibration results for those two datasets, not defaults to copy.

## Added files

- `multi_attribute/schema.py`, `registry.py`, `kinds.py`, `hashing.py`, `dataset.py`,
  `jobfile.py`, `calibration.py`
- `scripts/fetch_dataset.py`, `scripts/generate_synthetic_attributes.py`,
  `scripts/demo_multi_attribute_dataset.py`, `scripts/calibrate_multi_attribute.py`
- `config/examples/febrl_multi_attribute.json`, `config/examples/multi_attribute_schema.json`
- `tests/test_multi_attribute_schema.py`, `tests/test_multi_attribute_dataset.py`,
  `tests/test_multi_attribute_probing.py`

`jobfile.py` holds only the job/data loading the demo and the calibrator share. It was split
out so the calibrator does not import the demo (and therefore `tenseal`): plaintext
calibration runs in an environment without TenSEAL installed, which is also why
`tests/test_multi_attribute_probing.py` guards its two encrypted cases with
`pytest.importorskip`.
