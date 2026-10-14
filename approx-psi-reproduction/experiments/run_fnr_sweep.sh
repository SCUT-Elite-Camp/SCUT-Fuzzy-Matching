#!/usr/bin/env bash
set -euo pipefail
BIN=${1:-./build/approx_psi_plain}
for K in 5 10 20 40 80; do
  echo "===== k=${K} ====="
  "$BIN" --n 16 --ell 128 --d 4 --t 4 --lambda 10 --matches 4 --trials 100 --seed 42 --k "$K" | grep -E '^(k=|FNR=|avg_Hamming_checks=|false_positives=)'
done
