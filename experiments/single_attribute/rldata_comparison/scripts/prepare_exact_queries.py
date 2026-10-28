"""Build the exact-name query set used by the "no fuzzy name" experiments.

``data/queries.csv`` mixes exact, edited and native-umlaut positives with
recombination negatives.  The Top-k scaling experiment also needs a query set
without any fuzzy perturbation:

* 100 exact positives  -- query name is the database record name verbatim;
* 100 recombination negatives -- first name of one record + last name of
  another, verified absent from the database.

Records already used by ``queries.csv`` / ``calibration_queries.csv`` are
excluded so the new set is disjoint.  The DOB of each positive is copied from
its source record; a negative takes the DOB of its first-name source.  This
keeps the protocol input identical in shape to ``queries.csv``.

Run from the repository root::

    .venv\\Scripts\\python.exe experiments\\single_attribute\\rldata_comparison\\scripts\\prepare_exact_queries.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from names import normalize_name  # noqa: E402

DEFAULT_DATA = EXPERIMENT / "data"
OUTPUT_NAME = "queries_exact.csv"
FIELDS = [
    "query_id",
    "name",
    "normalized_name",
    "dob",
    "label",
    "expected_record_id",
    "expected_identity",
    "perturbation",
    "variant_form",
    "source_record_id",
]


def _used_record_ids(data_dir: Path) -> set[str]:
    used: set[str] = set()
    for filename in ("queries.csv", "calibration_queries.csv"):
        frame = pd.read_csv(data_dir / filename)
        for column in ("expected_record_id", "source_record_id"):
            if column in frame:
                used.update(
                    str(value)
                    for value in frame[column].fillna("")
                    if str(value)
                )
    return used


def _first_last(name: str) -> tuple[str, str]:
    parts = [part for part in name.split() if part]
    if len(parts) <= 1:
        return (parts[0] if parts else ""), ""
    return parts[0], parts[-1]


def build_exact_queries(
    database: pd.DataFrame, *, seed: int, data_dir: Path = DEFAULT_DATA
) -> list[dict]:
    rng = random.Random(seed)
    used_ids = _used_record_ids(data_dir)

    names_in_db = Counter(database["normalized_name"].tolist())
    records = database.to_dict("records")

    candidates = [
        record
        for record in records
        if str(record["record_id"]) not in used_ids
        and str(record["dob"] or "")
        and names_in_db[record["normalized_name"]] == 1
    ]
    rng.shuffle(candidates)
    if len(candidates) < 100:
        raise RuntimeError(f"only {len(candidates)} unused exact positive records")

    positives = candidates[:100]
    used_ids.update(str(record["record_id"]) for record in positives)

    negatives: list[dict] = []
    seen_names: set[str] = set()
    attempts = 0
    while len(negatives) < 100 and attempts < 200000:
        attempts += 1
        left, right = rng.sample(records, 2)
        if left["identity"] == right["identity"]:
            continue
        first, _ = _first_last(str(left["name"]))
        _, last = _first_last(str(right["name"]))
        if not first or not last:
            continue
        name = f"{first} {last}"
        canonical = normalize_name(name)
        if canonical in names_in_db or canonical in seen_names:
            continue
        seen_names.add(canonical)
        negatives.append(
            {
                "name": name,
                "normalized_name": canonical,
                "dob": str(left["dob"] or ""),
            }
        )
    if len(negatives) < 100:
        raise RuntimeError(f"only built {len(negatives)} recombination negatives")

    queries: list[dict] = []
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"E{counter:04d}"

    for record in positives:
        queries.append(
            {
                "query_id": next_id(),
                "name": record["name"],
                "normalized_name": record["normalized_name"],
                "dob": record["dob"],
                "label": "true",
                "expected_record_id": record["record_id"],
                "expected_identity": str(int(record["identity"])),
                "perturbation": "exact",
                "variant_form": "ascii",
                "source_record_id": record["record_id"],
            }
        )
    for negative in negatives:
        queries.append(
            {
                "query_id": next_id(),
                "name": negative["name"],
                "normalized_name": negative["normalized_name"],
                "dob": negative["dob"],
                "label": "false",
                "expected_record_id": "",
                "expected_identity": "-1",
                "perturbation": "recombination",
                "variant_form": "ascii",
                "source_record_id": "",
            }
        )
    return queries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=20261100)
    args = parser.parse_args()

    database = pd.read_csv(args.data / "database.csv")
    database["normalized_name"] = database["normalized_name"].fillna("")
    queries = build_exact_queries(database, seed=args.seed, data_dir=args.data)

    output = args.data / OUTPUT_NAME
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(queries)

    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["exact"] = {
        "total": len(queries),
        "positive": sum(query["label"] == "true" for query in queries),
        "negative": sum(query["label"] == "false" for query in queries),
        "perturbations": dict(sorted(Counter(q["perturbation"] for q in queries).items())),
        "seed": args.seed,
        "file": OUTPUT_NAME,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest["exact"], indent=2, ensure_ascii=False))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
