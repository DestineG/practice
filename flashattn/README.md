# FlashAttention Benchmark

`benchmark.cu` compares the naive attention kernel
`kernel_attn_prefill` with the hand-written FA1 kernel in `flashattn1.cu`.
The GPU benchmark is non-causal. The CPU reference implementation keeps a
causal switch for correctness experiments, but the command-line benchmark
does not launch the causal GPU kernel.

## Build

```bash
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_89 \
  -lineinfo -Xptxas=-v \
  flashattn/benchmark.cu -o flashattn/build/benchmark
```

`benchmark.cu` includes `naive-attention.cu` and `flashattn1.cu`, so compiling
it as a single translation unit is sufficient.

## Usage

```text
benchmark <verify|benchmark> <all|naive|fa1> N d [warmup] [iterations]
```

Examples:

```bash
# Correctness only
./flashattn/build/benchmark verify all 128 64
./flashattn/build/benchmark verify fa1 256 128

# Correctness first, then CUDA-event benchmark
./flashattn/build/benchmark benchmark all 256 64 10 100
./flashattn/build/benchmark benchmark naive 1024 128 10 100
```

In `benchmark` mode, both selected kernels are compared with the CPU FP32
reference before timing starts. A failed correctness check is reported and
the timing is still printed; the process exits with a non-zero status.

The reported time is the average kernel time over `iterations` launches after
`warmup` launches. Host initialization, allocation, and correctness checking
are outside the timed region.

## Supported Shapes

The naive kernel accepts positive `N` and `d`. The current FA1 benchmark
requires:

- `N` is a multiple of 64, because the current K/V tile padding path does not
  mask an incomplete KV tile;
- `d=64` uses `fa1<64, 64, 64>`;
- `d=128` uses `fa1<128, 16, 64>` to stay within the SM89 static shared-memory
  limit.

The FA1 algorithm is organized around `HEAD_DIM` values that are multiples of
64. Additional dimensions require a matching compile-time instantiation and a
tile shape that fits the available shared memory.

## Correctness

Inputs are deterministic BF16 Q/K/V tensors. The CPU reference converts them
to FP32 and computes:

```text
softmax((Q @ K^T) / sqrt(d)) @ V
```

The output reports maximum absolute error, mean absolute error, and maximum
relative error. The pass condition uses maximum absolute error `<= 5e-2` and
mean absolute error `<= 5e-3`; relative error is retained as a diagnostic
because values close to zero can make it misleading.

## Example Results

Measured on the current RTX 4090 setup with `warmup=3` and `iterations=20`:

| N | d | naive (ms) | FA1 (ms) | FA1 max abs |
|---:|---:|---:|---:|---:|
| 64 | 64 | 0.345600 | 0.008294 | 7.19e-3 |
| 128 | 64 | 1.288605 | 0.012970 | 7.76e-3 |
| 256 | 64 | 4.581325 | 0.022016 | 6.53e-3 |
| 64 | 128 | 0.658976 | 0.006346 | 5.49e-3 |
| 128 | 128 | 3.429888 | 0.010650 | 7.83e-3 |
| 256 | 128 | 6.961511 | 0.018637 | 5.93e-3 |

These values are device- and workload-dependent. Re-run the commands above
on the target GPU when updating the results.
