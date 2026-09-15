"""下载带出生日期/地址/SSN 等属性的公开测试数据集。

默认落到 ``dataset/``（该目录已在 .gitignore 中，因此不会让仓库体积膨胀）。

用法（在 baseline_Integration/ 下执行）::

    python scripts/fetch_dataset.py --dataset febrl
    python scripts/fetch_dataset.py --dataset all --output dataset

关于镜像：网上教程常给的 ``pages.turi.com`` / ``static.turi.com`` 两个地址
**已经失效**（DNS 无法解析），``raw.githubusercontent.com`` 在部分网络下不稳定。
所以这里按顺序尝试多个镜像，jsdelivr CDN 实测最稳。
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_MIRRORS = (
    "https://cdn.jsdelivr.net/gh/moj-analytical-services/splink_datasets@main/data",
    "https://gcore.jsdelivr.net/gh/moj-analytical-services/splink_datasets@main/data",
    "https://raw.githubusercontent.com/moj-analytical-services/splink_datasets/main/data",
)

# name -> (repo-relative source path, expected header columns, minimum rows)
_CATALOGUE = {
    "febrl": {
        "febrl/dataset4a.csv": (
            "febrl/dataset4a.csv",
            ["rec_id", "given_name", "surname", "date_of_birth", "soc_sec_id"],
            5000,
        ),
        "febrl/dataset4b.csv": (
            "febrl/dataset4b.csv",
            ["rec_id", "given_name", "surname", "date_of_birth", "soc_sec_id"],
            5000,
        ),
        "febrl/dataset3.csv": (
            "febrl/dataset3.csv",
            ["rec_id", "given_name", "surname", "date_of_birth"],
            5000,
        ),
    },
}


def _download(relative_source: str, timeout: int = 120) -> bytes:
    errors: list[str] = []
    for mirror in _MIRRORS:
        url = f"{mirror}/{relative_source}"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append(f"  {url} -> {exc}")
            continue
    raise RuntimeError(
        f"all mirrors failed for {relative_source}:\n" + "\n".join(errors)
    )


def _verify(payload: bytes, expected_columns: list[str], min_rows: int, label: str) -> int:
    text = payload.decode("utf-8", errors="replace")
    reader = csv.reader(text.splitlines())
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError(f"{label}: downloaded file is empty") from None
    header = [h.strip() for h in header]

    missing = [c for c in expected_columns if c not in header]
    if missing:
        raise ValueError(f"{label}: downloaded header is missing {missing}; got {header}")

    rows = sum(1 for _ in reader)
    if rows < min_rows:
        raise ValueError(f"{label}: expected >= {min_rows} rows, got {rows}")
    return rows


def fetch(dataset: str, output_dir: Path) -> list[tuple[Path, int]]:
    if dataset not in _CATALOGUE:
        raise SystemExit(
            f"unknown dataset {dataset!r}; available: {sorted(_CATALOGUE)} or 'all'"
        )
    written: list[tuple[Path, int]] = []
    for relative_dest, (relative_source, columns, min_rows) in _CATALOGUE[dataset].items():
        destination = output_dir / relative_dest
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {relative_source} ...", flush=True)
        payload = _download(relative_source)
        rows = _verify(payload, columns, min_rows, relative_source)
        destination.write_bytes(payload)
        written.append((destination, rows))
        print(f"    -> {destination}  ({rows} rows, {len(payload)} bytes)", flush=True)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=[*sorted(_CATALOGUE), "all"],
        default="febrl",
        help="which dataset to fetch (default: febrl)",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "dataset"),
        help="output root (default: baseline_Integration/dataset, gitignored)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    datasets = sorted(_CATALOGUE) if args.dataset == "all" else [args.dataset]

    print(f"fetching {', '.join(datasets)} -> {output_dir}")
    total = 0
    for name in datasets:
        print(f"[{name}]")
        total += len(fetch(name, output_dir))
    print(f"\ndone: {total} file(s) written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
