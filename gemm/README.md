# CUDA BF16 Tensor Core GEMM

本目录现在只保留 custom GEMM 实现。输入 `A`、`B` 为 row-major BF16，输出 `C` 为 row-major FP32，累加器为 FP32：

```text
C[M, N] = A[M, K] * B[K, N]
```

当前 aligned kernel 限制：

```text
M % 128 == 0
N % 128 == 0
K % 64 == 0
```

核心计算不使用 WMMA、CUTLASS 或 cuBLAS；cuBLAS 只用于 correctness 和性能对比。

## 实现结构

正式测试使用 `helper` kernel。`inline` kernel 保留在 `gemm_custom.cu` 中，用于复现一个已经定位到的编译结果问题，详见 [custom_issues.md](custom_issues.md)。

配置为：

```text
BM=128, BN=128, BK=64
WM=32,  WN=64
256 threads/block
2 shared-memory stages
65536 bytes dynamic shared memory/block
```

每个 warp 计算 `32x64` 的 C tile，每个 K tile 展开为 16 次：

```text
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
```

数据流为：

```text
Global -> cp.async 16B -> swizzled Shared
       -> ldmatrix -> BF16 MMA -> FP32 accumulator -> Global FP32
```

Shared layout 以每个 128B segment 为单位做 8 个 16B chunk 的 cyclic shift：

```text
physical_chunk = (logical_chunk + (row % 8)) % 8
```

B tile 的 256B 行会拆成两个独立 128B segment，不会跨 segment 或跨行移动。

## 正确性

cuBLAS 使用 BF16 输入和 FP32 accumulate 作为 reference。验证条件为：

```text
max_absolute <= 1e-3
relative_l2 <= 1e-5
```

helper kernel 已通过：

```text
128x128x64
1024x1024x1024
2048x2048x2048
```

并通过 `compute-sanitizer --tool memcheck`，无内存错误。

## Benchmark

测试使用 CUDA events，默认 warmup 100 次；1024/2048 和矩形尺寸执行 1000 次，4096 执行 100 次。完整测试矩阵、结果和分析直接记录在本文档中。

| M | N | K | Kernel | Time (ms) | TFLOPS |
| ---: | ---: | ---: | --- | ---: | ---: |
| 1024 | 1024 | 1024 | cuBLAS | 0.0176 | 122.326 |
| 1024 | 1024 | 1024 | naive | 0.4153 | 5.171 |
| 1024 | 1024 | 1024 | helper | 0.0356 | 60.318 |
| 2048 | 2048 | 2048 | cuBLAS | 0.1013 | 169.566 |
| 2048 | 2048 | 2048 | naive | 3.2156 | 5.343 |
| 2048 | 2048 | 2048 | helper | 0.1364 | 125.970 |
| 4096 | 4096 | 4096 | cuBLAS | 0.9167 | 149.924 |
| 4096 | 4096 | 4096 | helper | 1.0421 | 131.891 |
| 2048 | 1024 | 2048 | cuBLAS | 0.0530 | 162.215 |
| 2048 | 1024 | 2048 | helper | 0.0692 | 124.190 |
| 1024 | 2048 | 2048 | cuBLAS | 0.0529 | 162.287 |
| 1024 | 2048 | 2048 | helper | 0.0693 | 124.017 |

helper 相对 cuBLAS 的吞吐率约为：

```text
1024^3:             49.3%
2048^3:             74.3%
4096^3:             87.9%
2048x1024x2048:     76.6%
1024x2048x2048:     76.4%
```

随着矩阵增大，固定的 launch、shared-memory 和流水线开销被摊薄，helper kernel 的吞吐率逐渐接近 cuBLAS。当前主要差距来自寄存器占用较高、每个 CTA 使用 64KB shared memory 导致 occupancy 受限，以及尚未针对 Ada 架构做更激进的访存和 MMA pipeline 调度。

编译时 ptxas 报告：

```text
helper registers/thread: 162
register spills: 0
dynamic shared memory/block: 65536 bytes
```

## 复现

编译：

```bash
bash gemm/compile_custom.sh
```

correctness：

```bash
bash gemm/verify_custom.sh
```

benchmark：

```bash
bash gemm/benchmark_custom.sh
```

自定义测试参数：

```bash
WARMUP=100 ITERATIONS=1000 LARGE_ITERATIONS=100 bash gemm/benchmark_custom.sh
```

单独运行某个 kernel：

```bash
./gemm/build/gemm_custom verify helper 2048 2048 2048
./gemm/build/gemm_custom benchmark helper 2048 2048 2048 100 1000
./gemm/build/gemm_custom profile helper 2048 2048 2048
```
