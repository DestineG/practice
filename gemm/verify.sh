#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/build/gemm" verify all 1024 1024 1024
"$SCRIPT_DIR/build/gemm" verify rowmajor 2048 2048 2048
"$SCRIPT_DIR/build/gemm" verify swizzle 2048 2048 2048
"$SCRIPT_DIR/build/gemm" verify async 2048 2048 2048
