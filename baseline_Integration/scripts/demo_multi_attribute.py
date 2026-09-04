"""Runnable name + DOB privacy-preserving matching demo.

Run from baseline_Integration/:
    python scripts/demo_multi_attribute.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_attribute import (
    MatchRecord,
    MultiAttributeConfig,
    plaintext_similarity,
)
from multi_attribute.protocol import run_multi_attribute_protocol



def main() -> None:
    cfg = MultiAttributeConfig(
        name_weight=0.70,
        dob_weight=0.30,
        similarity_threshold=0.80,
    )

    records_b = [
        MatchRecord("John Smith", "2001-05-17", "B001"),
        MatchRecord("Jane Doe", "1999-08-02", "B002"),
        MatchRecord("Jon Smythe", "1988-11-23", "B003"),
        MatchRecord("Mary Johnson", "2003-01-10", "B004"),
        MatchRecord("Zhang San", "2005-03-18", "B005"),
    ]

    cases = [
        (
            "fuzzy-name + same DOB: should match",
            MatchRecord("Jon Smith", "2001/05/17", "Q1"),
        ),
        (
            "exact name + different DOB: should reject",
            MatchRecord("John Smith", "1990-01-01", "Q2"),
        ),
        (
            "different person: should reject",
            MatchRecord("Alice Brown", "1995-07-19", "Q3"),
        ),
    ]

    print("=== Multi-attribute private matching demo (name + DOB) ===")
    print(
        f"weights: name={cfg.name_weight:.2f}, dob={cfg.dob_weight:.2f}, "
        f"tau={cfg.similarity_threshold:.2f}"
    )
    print(
        f"vector dims: cluster={cfg.cluster_dim}, match={cfg.match_dim} "
        f"(name {cfg.name_cluster_dim}/{cfg.name_match_dim} + DOB {cfg.dob_dim})"
    )

    for title, query in cases:
        print("\n---", title, "---")
        # Plaintext scores are shown only for local validation/demo. They are not
        # part of the production protocol output.
        plain_scores = [plaintext_similarity(query, item, cfg) for item in records_b]
        best_idx = max(range(len(plain_scores)), key=plain_scores.__getitem__)
        print(
            "local reference best:",
            records_b[best_idx].record_id,
            records_b[best_idx].name,
            f"score={plain_scores[best_idx]:.4f}",
        )

        result = run_multi_attribute_protocol(
            records_b,
            query,
            cfg=cfg,
            k_mode="sqrt",
            random_state=42,
        )
        print(
            "encrypted protocol result:",
            "CATCH" if result.catch else "NO-CATCH",
            f"selected_cluster={result.selected_cluster}",
            f"checked_columns={result.checked_columns}",
        )


if __name__ == "__main__":
    main()
