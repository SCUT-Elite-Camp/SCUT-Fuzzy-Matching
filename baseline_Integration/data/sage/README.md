# SAGE multilingual name dataset

`sage_names_all.csv` is the extracted source table with two columns:
`candidate_name,country`. It contains 123,479 source rows from 23 countries.

The unified pipeline reads it with the `sage_names` adapter, removes duplicate
`(candidate_name, country)` pairs, assigns stable SHA-256-derived record IDs,
and normalizes names with `unicode_v1`, emitting 43,206 database records. The
difference from the 43,218 byte-exact source pairs comes from 12 additional
NFKC/case-fold-equivalent duplicates.

`sage_multilingual_queries.csv` is a small validation set, not an official SAGE
benchmark. It contains positive formatting/diacritic variants and explicit
negative queries across Latin, Arabic, Han, Han+Latin, and Myanmar scripts.

Generate the auditable prepared dataset from the project root:

```powershell
python scripts/prepare_dataset.py `
  --config config/examples/sage_pipeline.json `
  --output-dir data/sage/prepared
```

The generated directory contains `database.csv`, `queries.csv`, and
`manifest.json`. `manifest.json` records the original row count, duplicate
count, country distribution, normalization configuration, output sizes, and
orphan positive-query count. The 43,206 records currently expand to 56,705
deduplicated matching entries after native and Latin variants are indexed.

The validation set includes small cross-script checks such as
`廖學廣` matching `LiaoXueGuang` and an Arabic source name matching its generated
Latin variant. Transliteration is context-free and is not a language-specific
name romanization model.

Run the small real-HE validation:

```powershell
python scripts/validate_sage_multilingual.py
```

Run the presentation-oriented terminal demo:

```powershell
python scripts/demo_sage_cross_script.py
```
