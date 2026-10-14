# Secure pipeline patch: RsCpsi -> EMP SS-Hamming -> SS-VMult

This patch keeps the existing Plain Projection path and changes only the two-process secure path.

## What changed

1. `run_as_sender()` / `run_as_receiver()` no longer exchange `mFlagBits`.
2. No hit-bin list is opened.
3. No peer element id is exchanged.
4. No peer vector is serialized or sent in cleartext.
5. Both parties iterate over **all RsCpsi output bins** in the same order.
6. Each party feeds its own `mFlagBits[k]` share directly to EMP.
7. EMP reconstructs `hit = flag_A XOR flag_B` inside the circuit.
8. Each party feeds only its own local vector to EMP.
9. EMP computes:

   `keep = hit AND valid_A AND valid_B AND [H(a,b) <= d]`

10. The same EMP fragment performs vector masking:

   `out_A = keep * a`

   `out_B = keep * b`

11. `out_A` is opened only to Alice and `out_B` only to Bob.
12. Secure output records only the party's own matched element id. Peer id and exact distance stay hidden.

## Important output semantics

For the secure two-process path, `ExperimentResult.matches` is used as a compatibility container:

- Alice: `match.a_id` is the local matched id.
- Bob: `match.b_id` is the local matched id.
- the peer-id field and `distance` are set to `SIZE_MAX` because they are intentionally not revealed.

`approx_psi_full` prints only `local_id`.

## Build on the existing server

From the project root:

```bash
rm -rf build_full

CC=$HOME/local/gcc-11/bin/gcc \
CXX=$HOME/local/gcc-11/bin/g++ \
cmake -S . -B build_full \
  -DCMAKE_BUILD_TYPE=Release \
  -DAPPROX_PSI_BUILD_SECURE=ON \
  -DAPPROX_PSI_HAS_EMP=ON \
  -DAPPROX_PSI_HAS_VOLEPSI=ON \
  -DAPPROX_PSI_BUILD_FULL_DEMO=ON \
  -DCMAKE_PREFIX_PATH=$HOME/wsq/emp/install \
  -DOPENSSL_ROOT_DIR=$HOME/miniconda3 \
  -DOPENSSL_INCLUDE_DIR=$HOME/miniconda3/include \
  -DOPENSSL_CRYPTO_LIBRARY=$HOME/miniconda3/lib/libcrypto.so \
  -DOPENSSL_SSL_LIBRARY=$HOME/miniconda3/lib/libssl.so

cmake --build build_full --target approx_psi_full -j4
```

The existing CMake expects volepsi at:

```text
$HOME/volepsi/out/install/linux
```

If the actual installation is elsewhere, change the two volepsi prefix lines in `CMakeLists.txt`.

## First smoke test

Terminal 1 (Alice):

```bash
./build_full/approx_psi_full 1 12345 23456 127.0.0.1 16 128 4 4 10
```

Terminal 2 (Bob):

```bash
./build_full/approx_psi_full 2 12345 23456 127.0.0.1 16 128 4 4 10
```

Expected shape of output:

```text
party=ALICE
matched_local_elements=4
local_id=...
...
```

and similarly for Bob. The exact first-match rounds can vary with the projection schedule, but both sides should recover the planted local elements without printing peer ids or distances.

## Local validation performed here

The dependency-free/fallback build was compiled and its 6 existing CTest cases passed. The full EMP + volepsi target still needs to be compiled and executed on the server where those libraries are installed.
