#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
rm -rf build && mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH="$HOME/volepsi/out/install/linux" .. > /tmp/cmake.log 2>&1
make -j$(nproc) 2>&1 | tail -5
echo "===== run ====="
timeout 300 ./alice_bob "$@"
