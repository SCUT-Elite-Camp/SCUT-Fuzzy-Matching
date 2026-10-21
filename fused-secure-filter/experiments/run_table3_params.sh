#!/usr/bin/env bash
set -euo pipefail
BIN=${1:-./build/approx_psi_plain}
for N in 256 1024 4096; do
  case "$N" in
    256) T=8 ;;
    1024) T=10 ;;
    4096) T=12 ;;
  esac
  echo "===== n=${N}, t=${T} ====="
  # Avoid generating huge datasets: print the derived k then stop early with an intentionally impossible match count? Better use --help? Instead main generates data.
  # This script is informational; use test suite for the exact 96/89/86 check.
  python3 - <<PY
import math
n=$N; t=$T; lam=40
q=1/(n*t)**(1/(t-1))
a=1-q+n*q**t
k=math.ceil((lam+math.log2(n))/math.log2(1/a))
print(f"k={k}")
PY
done
