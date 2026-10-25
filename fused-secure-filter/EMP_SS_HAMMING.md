# Real EMP two-party SS-Hamming demo

This demo is a real two-process semi-honest garbled-circuit computation using
EMP `v0.3.x`. It is intentionally separate from the existing fallback secure
layer.

## What it computes

Alice owns binary vector `a`; Bob owns binary vector `b`. Inside EMP, the two
parties compute

\[
  c = [H(a,b) \le d].
\]

The bit `c` is **not opened**. Instead the program returns XOR shares:

\[
  c_A \oplus c_B = c.
\]

Alice samples a fresh random bit `r` and keeps `c_A=r`. The circuit computes
`c_B=c XOR r` and reveals only that share to Bob.

## EMP version

Use the stable `v0.3.x` line for this reproduction:

```bash
curl https://raw.githubusercontent.com/emp-toolkit/emp-readme/main/scripts/install.py -O
python3 install.py --deps --tool=v0.3.x --ot=v0.3.x --sh2pc=v0.3.x
```

## Build

From the project root:

```bash
rm -rf build_emp
cmake -S . -B build_emp \
  -DCMAKE_BUILD_TYPE=Release \
  -DAPPROX_PSI_BUILD_EMP_HAMMING_DEMO=ON
cmake --build build_emp -j
```

If CMake cannot locate EMP, pass the directory containing
`emp-sh2pc-config.cmake`, commonly `/usr/local/cmake`:

```bash
cmake -S . -B build_emp \
  -DCMAKE_BUILD_TYPE=Release \
  -DAPPROX_PSI_BUILD_EMP_HAMMING_DEMO=ON \
  -Demp-sh2pc_DIR=/usr/local/cmake
```

## Run on one machine, two terminals

Example 1: distance is 1 and threshold is 1, so the reconstructed result should
be `1`.

Terminal 1 (Alice):

```bash
./build_emp/ss_hamming_emp 1 12345 101101 1
```

Terminal 2 (Bob):

```bash
./build_emp/ss_hamming_emp 2 12345 101001 1 127.0.0.1
```

Read `local_output_share` from both terminals. Check manually:

```text
alice_share XOR bob_share = 1
```

Example 2: keep the same vectors but threshold 0. Their Hamming distance is 1,
so the reconstructed result should be `0`.

Alice:

```bash
./build_emp/ss_hamming_emp 1 12346 101101 0
```

Bob:

```bash
./build_emp/ss_hamming_emp 2 12346 101001 0 127.0.0.1
```

Now:

```text
alice_share XOR bob_share = 0
```

## Run on two machines

Alice machine (server):

```bash
./build_emp/ss_hamming_emp 1 12345 101101 1
```

Bob machine, replacing the IP with Alice's reachable IP:

```bash
./build_emp/ss_hamming_emp 2 12345 101001 1 192.168.1.10
```

Open TCP port `12345` between the hosts if a firewall is enabled.

## Security boundary

This demo is genuinely two-party for SS-Hamming: Alice never receives Bob's
clear vector; Bob never receives Alice's clear vector; the comparison result is
kept as XOR shares. The vector length and threshold are public protocol
parameters.

This does **not** yet make the full Approx-PSI secure. The next steps are real
SS-VMult and then volepsi Circuit PSI, followed by wiring all three modules into
Algorithm 1.
