#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WARMUP="${WARMUP:-100}"
ITERATIONS="${ITERATIONS:-1000}"

for size in 1024 2048; do
    "$SCRIPT_DIR/build/gemm" benchmark all "$size" "$size" "$size" "$WARMUP" "$ITERATIONS"
done

for kernel in cublas rowmajor swizzle async; do
    "$SCRIPT_DIR/build/gemm" benchmark "$kernel" 4096 4096 4096 "$WARMUP" "$ITERATIONS"
done
