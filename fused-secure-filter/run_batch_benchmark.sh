#!/usr/bin/env bash
set -euo pipefail
# Usage: ./run_batch_benchmark.sh [build-dir] [n] [ell] [d] [t] [lambda]
BUILD="${1:-build_full}"; N="${2:-16}"; ELL="${3:-128}"; D="${4:-4}"; T="${5:-4}"; L="${6:-10}"
EMP_PORT="${EMP_PORT:-12447}"; PSI_PORT="${PSI_PORT:-23558}"
EXE="$BUILD/approx_psi_full"
if [[ ! -x "$EXE" ]]; then echo "Missing $EXE. Build first." >&2; exit 1; fi
mkdir -p benchmark_logs
BASE="benchmark_logs/batch_n${N}_ell${ELL}_d${D}_t${T}_lambda${L}"
"$EXE" 1 "$EMP_PORT" "$PSI_PORT" 127.0.0.1 "$N" "$ELL" "$D" "$T" "$L" >"${BASE}.alice.log" 2>&1 &
ALICE_PID=$!
trap 'kill "$ALICE_PID" 2>/dev/null || true' EXIT
sleep 1
"$EXE" 2 "$EMP_PORT" "$PSI_PORT" 127.0.0.1 "$N" "$ELL" "$D" "$T" "$L" >"${BASE}.bob.log" 2>&1
wait "$ALICE_PID"
trap - EXIT
printf '=== matched counts ===\n'
grep -H 'matched_local_elements=' "${BASE}.alice.log" "${BASE}.bob.log"
printf '=== per-round stage timing (Alice) ===\n'
grep 'STAGE_TIMING' "${BASE}.alice.log" || true
printf '=== end-to-end network stats (Alice) ===\n'
tail -18 "${BASE}.alice.log"
echo "Logs: ${BASE}.{alice,bob}.log"
