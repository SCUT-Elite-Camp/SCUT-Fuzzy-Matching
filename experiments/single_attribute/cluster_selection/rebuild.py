"""Rebuild one historical experiment without overwriting the active baseline."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]


def rebuild(variant: str, output: Path) -> Path:
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["variants"][variant]
    patch = HERE / entry["patch"]
    if hashlib.sha256(patch.read_bytes()).hexdigest() != entry["patch_sha256"]:
        raise ValueError("Patch checksum mismatch")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output must be a new directory: {output}")
    archive = subprocess.check_output(
        ["git", "archive", manifest["base_commit"], "baseline_Integration"],
        cwd=REPOSITORY,
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        members = tree.getmembers()
        for member in members:
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported archive entry: {member.name}")
            if not (output / member.name).resolve().is_relative_to(output):
                raise ValueError(f"Unsafe archive path: {member.name}")
        output.mkdir(parents=True)
        # Explicit extraction also works on Python 3.10 (no filter= API needed).
        for member in members:
            target = output / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = tree.extractfile(member)
                if stream is None:
                    raise ValueError(f"Cannot read archive entry: {member.name}")
                with stream:
                    target.write_bytes(stream.read())
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=output, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=output, check=True)
    return output / "baseline_Integration"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", required=True,
        choices=("bad_event", "fixed_threshold", "topk", "hyperplane"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(rebuild(args.variant, args.output))


if __name__ == "__main__":
    main()
