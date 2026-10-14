# Approx-PSI Full Merged Reproduction

This project merges the two uploaded codebases into one tree:

- **Plain Algorithm-1 reproduction**: parameter derivation, synthetic gap-respecting data, random projection, random representative selection, Monte-Carlo/FNR and scaling experiments.
- **Secure-subprotocol layer**: XOR secret-sharing helpers plus interfaces for SS-PSI, SS-Hamming, SS-VMult, EMP context, and a volepsi adapter.

## Important status

The project contains **secure interfaces and fallback simulations**, but it is **not yet a complete cryptographic reproduction of the paper**.

- Plain mode is fully usable for the experiments already run.
- Without `APPROX_PSI_HAS_EMP`, SS-Hamming/SS-VMult reconstruct values locally in fallback code for testing.
- Without `APPROX_PSI_HAS_VOLEPSI`, SS-PSI uses a plaintext equality fallback.
- `src/secure/volepsi_adapter.cpp` is still a stub and returns no real volepsi output.
- `SecureApproxPSI::run()` is currently a secure-integration skeleton; it does not yet complete the full candidate verification/output path.

Therefore, use the current secure tests only to validate module plumbing, **not** to claim end-to-end privacy/security.

## Build

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

If your system `ctest` is old, use:

```bash
cd build
ctest --output-on-failure
```

## Plain experiment

```bash
./build/approx_psi_plain \
  --n 16 \
  --ell 128 \
  --d 4 \
  --t 4 \
  --lambda 10 \
  --matches 4 \
  --trials 500 \
  --seed 42
```

## Existing experiment scripts

```bash
bash experiments/run_fnr_sweep.sh
bash experiments/run_table3_params.sh
```

## Secure-layer tests

Built by default (`APPROX_PSI_BUILD_SECURE=ON`):

```bash
./build/test_secret_share
./build/test_ss_hamming
./build/test_ss_vmult
./build/test_ss_psi
./build/test_secure_approxpsi
```

These currently exercise fallback/skeleton paths unless real EMP/volepsi backends are integrated.

## Directory structure

```text
include/                 Plain headers
include/secure/          Secure-subprotocol headers
src/                     Plain implementation
src/secure/              Secure/fallback implementation
secure_demo/             Two-party demo scaffolding
experiments/             FNR and Table-3 parameter scripts
tests/                   Plain + secure-layer tests
```

## Next cryptographic integration work

1. Pin and link the paper-compatible EMP-Toolkit version.
2. Replace the fallback branches in `ss_hamming.cpp` and `ss_vmult.cpp` with two-party EMP execution that never reconstructs inputs locally.
3. Pin and integrate volepsi/OPPRF into `volepsi_adapter.cpp` / `ss_psi.cpp`.
4. Complete `SecureApproxPSI::run()` so PSI payload shares flow into secure Hamming verification and secure vector multiplication, and only final permitted outputs are opened.
5. Measure real communication bytes and secure runtime for Figure 5 / Table 3 reproduction.
