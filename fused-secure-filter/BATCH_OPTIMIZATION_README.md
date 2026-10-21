# Secure Approx-PSI: round-batched performance optimization

> **2026-10-18 update:** the Hamming predicate was replaced by a small-threshold
> saturating tree and projection coordinates are skipped (both public), cutting
> round-batched EMP traffic by 7.9x-19x. See `PERFORMANCE_ANALYSIS.md` for the
> measured before/after table against the paper's Table 3.

## Changed code

- `include/secure/hamming_threshold.h` (new): 4-bit-leaf + 3-bit-saturating merge tree, ~605 AND gates for 128 bits instead of ~3987; generic saturating counter fallback for d > 6.
- `src/secure/secure_approx_psi.cpp`: `secureFilterBatch` uses that circuit and accepts a public `compare_mask`, so the round's projection coordinates (provably 0 for every PSI hit) are not compared. `APPROX_PSI_COMPARE_ALL=1` disables the skip for A/B checks.
- `src/secure/secure_approx_psi.cpp`: a complete RsCpsi round now feeds **all bins** to `secureFilterBatch`; no plaintext PSI hit or distance is used for branching. EMP inputs, circuit evaluation and authorized final outputs are processed in separate phases. Each round emits `STAGE_TIMING` with projection, RsCpsi and EMP-batch durations.
- `include/secure/secure_approx_psi.h`: batch API added. Legacy per-bin API retained for compatibility.
- `CMakeLists.txt`: removes unconditional requirements for EMP when only building plain tests; secure builds still require EMP and volePSI.
- `run_batch_benchmark.sh`: launches Alice and Bob concurrently, saves logs, prints match counts and stage timings.

## Important clarification

The input already had **one EMPContext/SH2PCSession in main** and reused it across all rounds. This revision preserves that long-lived session; it does not claim that per-bin construction originally reopened TCP connections.

Only final `keep` membership bits are revealed. Masked *own* payloads previously revealed by the per-bin path are omitted from the circuit because each party already knows their original vector and can reconstruct its own masked output from `(keep, own_vector)` locally. Do not substitute this for a functionality that specifically requires secret-shared masked vectors to remain unopened.

This change batches per-round logical operations and reduces revelations; whether it reduces transport round trips as expected depends on EMP's internals, and must be measured on the user's original EMP runtime.

## Build on the original Ubuntu server

```bash
cmake -S . -B build_batch \
  -DAPPROX_PSI_BUILD_SECURE=ON \
  -DAPPROX_PSI_HAS_EMP=ON \
  -DAPPROX_PSI_HAS_VOLEPSI=ON \
  -DAPPROX_PSI_BUILD_FULL_DEMO=ON \
  -DCMAKE_PREFIX_PATH="$HOME/wsq/emp_new/install;$HOME/volepsi/out/install/linux"
cmake --build build_batch --target approx_psi_full -j4
./run_batch_benchmark.sh build_batch 16 128 4 4 10
./run_batch_benchmark.sh build_batch 256 128 4 8 40
./run_batch_benchmark.sh build_batch 1024 128 4 10 40
./run_batch_benchmark.sh build_batch 4096 128 4 12 40
```

The `RsCpsiSender::mMapping` correction described in the original report must remain applied to the **installed volePSI implementation**. This package does not contain that upstream dependency or apply the upstream patch.

## Stage communication measurement: what is and is not measurable

`STAGE_TIMING` provides actual per-round duration measurements for projection, PSI and fused EMP. EMP's `get_statistics_string()` is printed at program termination, giving its actual transport totals. **Do not interpret EMP-only counters as total PSI+EMP bytes.** `coproto`/volePSI traffic is on a separate socket. This version does not instrument coproto's socket, so per-stage **byte counts for PSI and EMP are not yet complete**. Accurately splitting Hamming vs VMult/network would require instrumenting the underlying socket/garbling runtime or running separately controlled instrumentation; fused secure filtering cannot be divided into independent network measurements by timing alone.

## Validation and comparability

1. Compare both processes' `matched_local_elements` against `4` on the planted dataset.
2. Check repeated runs and compare the **sum of both parties' complete transport traffic**, with each byte counted once at the sending endpoint, against a similarly measured baseline. Distinguish theoretical asymptotic complexity from measured LAN throughput.
3. Also compare `flushes`, `network rounds`, EMP total bytes and elapsed time. The n=4096 report baseline is **113.55 GiB** and **117,490,363 network rounds** (as reported by its old statistics); a new comparable total has not yet been measured.
4. The attached paper's near-linear bound is an asymptotic theorem for its specified gap/security setting, not an empirical byte target for this particular volePSI + EMP implementation.
5. Real, end-to-end secure build and execution **could not be verified in the current environment**, where the original server-side EMP/volePSI installations are unavailable. Plain test target was compiled and its tests passed.
