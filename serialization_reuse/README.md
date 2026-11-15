# CKKS Fuzzy Matching with Serialization Reuse

Clean, minimal runtime copy of the current CKKS fuzzy-matching model.

## What is included

- Character-trigram MinHash representation (EL=200 / EL=50 prefix)
- NumPy/scikit-learn cosine K-Means
- Tiled two-round TenSEAL CKKS protocol
- NCVR 10K data used by the evaluation
- Single CLI entry point: `scripts/evaluate_ncvr_10k.py`

Candidate depth profiles:

| Profile | probes | status |
|---|---:|---|
| `fast` | 1 | optional |
| `balanced` | 3 | **default** |

`high_recall=8` is no longer a formal runtime profile. Historical Top-8
experiments are archived outside this folder.

## Requirements

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The code was validated with Python 3.10, NumPy 1.26, scikit-learn 1.5,
TenSEAL 0.3.16.

## Quick start

```bash
# Balanced default: Top-3, tiled CKKS batch
python scripts/evaluate_ncvr_10k.py \
  --batch-size 100 \
  --depth-profile balanced \
  --output-dir artifacts/ncvr_balanced

# Fast: Top-1
python scripts/evaluate_ncvr_10k.py \
  --batch-size 100 \
  --depth-profile fast \
  --output-dir artifacts/ncvr_fast
```

Useful options:

```text
--query-limit N      limit queries
--db-limit N         limit database rows
--k N                fixed cluster count (default 50)
--fuzzy-ratio 0.3    positive-query perturbation
--probes N           explicit depth, formal range 1..3
```

## Result files

`--output-dir` contains:

- `ncvr_10k_result.json`
- `ncvr_10k_metrics.csv`
- `ncvr_10k_confusion.csv`
- `ncvr_10k_metrics.png`
- `ncvr_10k_confusion.png`

## Layout

```text
ckks/            CKKS context, batching, tiling
clustering/      cosine K-Means
config/          parameters and candidate-depth profiles
data/            NCVR 10K database/queries
data_pipeline/   dataset loading/adapters
evaluation/      benchmark, metrics, reporting
minhash/         trigram MinHash encoder
party_a/         query-side encoding/encryption
party_b/         database-side offline/online computation
preprocessing/   name normalization
protocol/        tiled two-round protocol
scripts/         evaluate_ncvr_10k.py
```

## Notes

- This folder intentionally contains no historical research reports, tests,
  docs, demos, GUI, or experimental variants.
- The original README and all historical material are preserved in `../other/`.
- Top-3 is the formal default. Top-1 remains available via `--depth-profile fast`
  or `--probes 1`.
