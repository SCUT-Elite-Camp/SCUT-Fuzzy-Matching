# NCVR 10K Test Subset

Source: `dataset/ncvoter_Statewide.txt` (local NCVR statewide voter-level file; ignored by git)

Generated files:
- `ncvr_10k_database.csv`: 10000 B-side database records with `ncid,full_name`
- `ncvr_10k_queries.csv`: 200 A-side queries with `query_ncid,query_name,label`

Construction:
- B database: first 10000 unique usable NCID records with non-empty full names
- Positive queries: first 100 records from B database (`label=True`)
- Negative queries: next 100 unique records after the B database (`label=False`)
- Full name format: `first_name middle_name last_name name_suffix_lbl`, skipping blanks
- Input encoding used for extraction: latin1

Purpose: small, deterministic local subset for testing current model behavior on real NCVR names.
