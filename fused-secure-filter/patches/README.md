# Patches and build notes for the secure backend

The end-to-end name pipeline (`name_secure_pipeline`) needs two dependencies that
are not vendored here:

- **EMP** (`emp-tool`, `emp-ot`, `emp-sh2pc`) installed in `/usr/local`
  (`SH2PCSession` explicit-session API, v0.3.x headers).
- **volePSI** installed from source (the local build used commit `ec76012`).

## 1. volePSI: sender row → bin mapping

`volepsi-mmapping.patch` adds the sender-side `Sharing::mMapping` that upstream
leaves empty, so callers can tell which local input row owns a PSI bin:

```bash
git -C ~/volepsi apply /path/to/fused-secure-filter/patches/volepsi-mmapping.patch
```

Rebuild and reinstall volePSI after applying it.

## 2. volePSI: generated `config.h`

`RsCpsi.h` includes `volePSI/config.h`, which only exists in the build tree.
After building, copy it into the install prefix before configuring this project:

```bash
cp ~/volepsi/out/build/linux/volePSI/config.h \
   ~/volepsi/out/install/linux/include/volePSI/config.h
```

## 3. Configure and build

```bash
export VOLEPSI_ROOT=$HOME/volepsi/out/install/linux
cmake -S fused-secure-filter -B ~/build/fused-secure -G Ninja \
      -DAPPROX_PSI_BUILD_NAME_PIPELINE=ON
cmake --build ~/build/fused-secure -j16 \
      --target name_secure_pipeline test_name_top1_clear test_name_slot_keys
```

`APPROX_PSI_BUILD_NAME_PIPELINE=ON` is independent of the plain build; the
default configuration (`APPROX_PSI_HAS_EMP=OFF`) still builds and passes its six
fallback tests without EMP or volePSI.

## 4. Run

```bash
python experiments/single_attribute/rldata_comparison/scripts/secure_runner.py \
    --seeds 20261023,20261040,20261057
```

The runner exports both party inputs, starts the server first and then the
client through `wsl.exe`, checks every prediction against the plaintext
reference, and writes
`experiments/single_attribute/rldata_comparison/results/secure_e2e*.json`.

`NPSI_DEBUG=1` on both processes additionally opens the circuit's internal
`(best_distance, best_identity, accepted)` to PUBLIC for debugging; it is off by
default and adds no traffic otherwise.
