"""Build the RLdata10000 B-side and the 200-query A-side used by both schemes.

Source: ``RLdata10000`` from the R package ``RecordLinkage``
(10,000 records, 9,000 identities, 1,000 identities with two records).
The ``.rda`` file is parsed with the pure-Python ``rdata`` package, so no R
installation is required.

Outputs (directly under ``data/`` next to this script):

* ``database.csv``            -- 10,000 B-side records
* ``queries.csv``             -- 200 evaluation queries (100 pos / 100 neg)
* ``calibration_queries.csv`` -- 200 disjoint queries for threshold calibration
* ``manifest.json``           -- composition and generation parameters

Query composition (per split):

* 50 exact positives          -- query name == a B record name
* 40 fuzzy positives          -- one edit (substitute/delete/insert/transpose)
* 10 multilingual positives   -- native German umlaut form of an ASCII B name
* 50 recombination negatives  -- first name of one record + last name of another
* 50 fuzzy negatives          -- recombination negative + one edit

Both positive classes keep the source record's date of birth, so the
multi-attribute ablation is not confounded by a second perturbation.
"""

from __future__ import annotations

import argparse
import csv
import sys
import datetime as dt
import json
import random
import string
from collections import Counter
from pathlib import Path
from typing import Any

import rdata

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
ROOT = EXPERIMENT.parents[2]
for entry in (HERE, ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from names import name_variants, normalize_name  # noqa: E402

DEFAULT_RDA = ROOT / "baseline_Integration" / "dataset" / "rldata10000" / "RLdata10000.rda"
DEFAULT_OUT = EXPERIMENT / "data"

LETTERS = string.ascii_uppercase


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"<na>", "na", "nan", "none"}:
        return ""
    return text


def _as_int(value: Any) -> int | None:
    text = _as_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _valid_dob(year: Any, month: Any, day: Any) -> str:
    y, m, d = _as_int(year), _as_int(month), _as_int(day)
    if y is None or m is None or d is None:
        return ""
    if not (1900 <= y <= 2010):
        return ""
    try:
        return dt.date(y, m, d).isoformat()
    except ValueError:
        return ""


def _full_name(row: dict[str, Any]) -> str:
    parts = [
        _as_text(row.get("fname_c1")),
        _as_text(row.get("fname_c2")),
        _as_text(row.get("lname_c1")),
        _as_text(row.get("lname_c2")),
    ]
    return " ".join(part for part in parts if part)


def load_records(rda_path: Path) -> list[dict[str, Any]]:
    parsed = rdata.read_rda(rda_path)
    frame = parsed["RLdata10000"]
    identities = parsed["identity.RLdata10000"]

    records: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(frame.iterrows()):
        name = _full_name(row)
        if not name:
            continue
        identity = _as_int(identities[index])
        if identity is None:
            continue
        record = {
            "record_id": f"RL{index + 1:05d}",
            "name": name,
            "normalized_name": normalize_name(name),
            "dob": _valid_dob(row.get("by"), row.get("bm"), row.get("bd")),
            "identity": identity,
            "fname_c1": _as_text(row.get("fname_c1")),
            "fname_c2": _as_text(row.get("fname_c2")),
            "lname_c1": _as_text(row.get("lname_c1")),
            "lname_c2": _as_text(row.get("lname_c2")),
        }
        if record["normalized_name"]:
            records.append(record)
    return records


def _first_last(name: str) -> tuple[str, str]:
    parts = [part for part in name.split() if part]
    if len(parts) <= 1:
        return (parts[0] if parts else ""), ""
    return parts[0], parts[-1]


def _mutate(name: str, rng: random.Random) -> tuple[str, str]:
    """Return one single-edit mutation and its label."""
    positions = [i for i, ch in enumerate(name) if ch.isalpha()]
    if not positions:
        return name, "none"
    operation = rng.choice(["substitute", "delete", "insert", "transpose"])

    if operation == "substitute":
        pos = rng.choice(positions)
        replacement = rng.choice([c for c in LETTERS if c != name[pos]])
        return name[:pos] + replacement + name[pos + 1 :], "substitution"

    if operation == "delete" and len(positions) > 3:
        pos = rng.choice(positions)
        return name[:pos] + name[pos + 1 :], "deletion"

    if operation == "insert":
        pos = rng.choice(positions)
        return name[:pos] + rng.choice(LETTERS) + name[pos:], "insertion"

    # transpose: swap two adjacent letters
    adjacent = [
        (name[i], name[i + 1], i)
        for i in range(len(name) - 1)
        if name[i].isalpha() and name[i + 1].isalpha() and name[i] != name[i + 1]
    ]
    if adjacent:
        _, _, pos = rng.choice(adjacent)
        return name[:pos] + name[pos + 1] + name[pos] + name[pos + 2 :], "transposition"

    pos = rng.choice(positions)
    replacement = rng.choice([c for c in LETTERS if c != name[pos]])
    return name[:pos] + replacement + name[pos + 1 :], "substitution"


def _native_umlaut(name: str) -> str | None:
    """Turn one German transliterated umlaut back into its native letter.

    Only a digraph preceded by a consonant is an umlaut (``MUELLER``); the same
    letters in ``BAUER``/``SAUER`` are a syllable boundary and must be left
    alone, otherwise the "native" query is not a real German spelling.
    """
    for ascii_form, native in (("UE", "Ü"), ("OE", "Ö"), ("AE", "Ä")):
        start = 0
        while True:
            index = name.find(ascii_form, start)
            if index < 0:
                break
            previous = name[index - 1] if index > 0 else ""
            if previous and previous.isalpha() and previous not in "AEIOU":
                return name[:index] + native + name[index + 2 :]
            start = index + 1
    return None


def _unique_positives(
    records: list[dict[str, Any]],
    forbidden_names: set[str],
    used_ids: set[str],
    count: int,
    *,
    require_umlaut: bool,
    rng: random.Random,
) -> list[dict[str, Any]]:
    candidates = []
    for record in records:
        if record["record_id"] in used_ids:
            continue
        if not record["dob"]:
            continue
        if forbidden_names[record["normalized_name"]] != 1:
            continue
        if require_umlaut and _native_umlaut(record["name"]) is None:
            continue
        candidates.append(record)
    rng.shuffle(candidates)
    picked: list[dict[str, Any]] = []
    for record in candidates:
        if len(picked) >= count:
            break
        if record["record_id"] in used_ids:
            continue
        picked.append(record)
        used_ids.add(record["record_id"])
    return picked


def _safe_mutation(
    name: str,
    forbidden: set[str],
    rng: random.Random,
    attempts: int = 80,
) -> tuple[str, str]:
    best = (name, "none")
    for _ in range(attempts):
        candidate, label = _mutate(name, rng)
        if normalize_name(candidate) not in forbidden:
            return candidate, label
        best = (candidate, label)
    return best


def _recombination(
    records: list[dict[str, Any]],
    forbidden: set[str],
    rng: random.Random,
    attempts: int = 200,
) -> tuple[str, str, str] | None:
    for _ in range(attempts):
        a, b = rng.sample(records, 2)
        if a["identity"] == b["identity"]:
            continue
        first, _ = _first_last(a["name"])
        _, last = _first_last(b["name"])
        if not first or not last:
            continue
        candidate = f"{first} {last}"
        if normalize_name(candidate) in forbidden:
            continue
        return candidate, a["record_id"], b["record_id"]
    return None


def build_queries(
    records: list[dict[str, Any]],
    *,
    seed: int,
    split: str,
    used_source_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    used_source_ids = used_source_ids if used_source_ids is not None else set()
    forbidden_names = Counter(record["normalized_name"] for record in records)
    names_in_db = set(forbidden_names)

    exact_sources = _unique_positives(
        records, forbidden_names, used_source_ids, 50, require_umlaut=False, rng=rng
    )
    fuzzy_sources = _unique_positives(
        records, forbidden_names, used_source_ids, 40, require_umlaut=False, rng=rng
    )
    multilingual_sources = _unique_positives(
        records, forbidden_names, used_source_ids, 10, require_umlaut=True, rng=rng
    )
    if len(exact_sources) < 50 or len(fuzzy_sources) < 40 or len(multilingual_sources) < 10:
        raise RuntimeError(
            f"{split}: not enough unique positive sources "
            f"({len(exact_sources)}/{len(fuzzy_sources)}/{len(multilingual_sources)})"
        )

    queries: list[dict[str, Any]] = []
    query_counter = 0

    def next_id() -> str:
        nonlocal query_counter
        query_counter += 1
        return f"{split[0].upper()}{query_counter:04d}"

    for record in exact_sources:
        queries.append(
            {
                "query_id": next_id(),
                "name": record["name"],
                "normalized_name": record["normalized_name"],
                "dob": record["dob"],
                "label": "true",
                "expected_record_id": record["record_id"],
                "expected_identity": str(record["identity"]),
                "perturbation": "exact",
                "variant_form": "ascii",
                "source_record_id": record["record_id"],
            }
        )

    for record in fuzzy_sources:
        mutation, label = _safe_mutation(
            record["name"], names_in_db - {record["normalized_name"]}, rng
        )
        queries.append(
            {
                "query_id": next_id(),
                "name": mutation,
                "normalized_name": normalize_name(mutation),
                "dob": record["dob"],
                "label": "true",
                "expected_record_id": record["record_id"],
                "expected_identity": str(record["identity"]),
                "perturbation": label,
                "variant_form": "ascii",
                "source_record_id": record["record_id"],
            }
        )

    for record in multilingual_sources:
        native = _native_umlaut(record["name"])
        assert native is not None
        queries.append(
            {
                "query_id": next_id(),
                "name": native,
                "normalized_name": normalize_name(native),
                "dob": record["dob"],
                "label": "true",
                "expected_record_id": record["record_id"],
                "expected_identity": str(record["identity"]),
                "perturbation": "umlaut_native",
                "variant_form": "umlaut_native",
                "source_record_id": record["record_id"],
            }
        )

    negative_sources = [r for r in records if r["record_id"] not in used_source_ids]
    for index in range(50):
        combination = _recombination(negative_sources, names_in_db, rng)
        if combination is None:
            raise RuntimeError(f"{split}: failed to build recombination negative")
        name, left_id, right_id = combination
        dob_source = next(r for r in records if r["record_id"] == left_id)
        queries.append(
            {
                "query_id": next_id(),
                "name": name,
                "normalized_name": normalize_name(name),
                "dob": dob_source["dob"] or "",
                "label": "false",
                "expected_record_id": "",
                "expected_identity": "-1",
                "perturbation": "recombination",
                "variant_form": "ascii",
                "source_record_id": "",
            }
        )

    for index in range(50):
        combination = _recombination(negative_sources, names_in_db, rng)
        if combination is None:
            raise RuntimeError(f"{split}: failed to build fuzzy negative")
        name, left_id, right_id = combination
        mutation, label = _safe_mutation(name, names_in_db, rng)
        dob_source = next(r for r in records if r["record_id"] == right_id)
        queries.append(
            {
                "query_id": next_id(),
                "name": mutation,
                "normalized_name": normalize_name(mutation),
                "dob": dob_source["dob"] or "",
                "label": "false",
                "expected_record_id": "",
                "expected_identity": "-1",
                "perturbation": f"recombination_{label}",
                "variant_form": "ascii",
                "source_record_id": "",
            }
        )

    # Sanity: positives must point at an existing record, negatives must not
    # normalise to a database name.
    record_ids = {record["record_id"] for record in records}
    for query in queries:
        if query["label"] == "true":
            assert query["expected_record_id"] in record_ids
        else:
            assert query["normalized_name"] not in names_in_db
    return queries


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rda", type=Path, default=DEFAULT_RDA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()

    records = load_records(args.rda)
    if len(records) < 10000:
        raise SystemExit(f"expected >=10000 usable records, got {len(records)}")

    used_source_ids: set[str] = set()
    test_queries = build_queries(
        records, seed=args.seed, split="test", used_source_ids=used_source_ids
    )
    calibration_queries = build_queries(
        records,
        seed=args.seed + 1,
        split="calibration",
        used_source_ids=used_source_ids,
    )

    database_fields = [
        "record_id",
        "name",
        "normalized_name",
        "dob",
        "identity",
        "fname_c1",
        "fname_c2",
        "lname_c1",
        "lname_c2",
    ]
    query_fields = [
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
    write_csv(args.out / "database.csv", records, database_fields)
    write_csv(args.out / "queries.csv", test_queries, query_fields)
    write_csv(args.out / "calibration_queries.csv", calibration_queries, query_fields)

    def split_stats(queries: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "total": len(queries),
            "positive": sum(q["label"] == "true" for q in queries),
            "negative": sum(q["label"] == "false" for q in queries),
            "perturbations": dict(sorted(Counter(q["perturbation"] for q in queries).items())),
            "variants": dict(sorted(Counter(q["variant_form"] for q in queries).items())),
        }

    identities = Counter(record["identity"] for record in records)
    manifest = {
        "source": str(args.rda),
        "seed": args.seed,
        "database_records": len(records),
        "database_unique_names": len({r["normalized_name"] for r in records}),
        "database_identities": len(identities),
        "identities_with_duplicate_records": sum(1 for c in identities.values() if c > 1),
        "records_with_valid_dob": sum(bool(r["dob"]) for r in records),
        "test": split_stats(test_queries),
        "calibration": split_stats(calibration_queries),
        "name_variant_counts": dict(
            Counter(len(name_variants(r["name"])) for r in records)
        ),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
