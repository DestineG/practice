# CUDA BF16 Tensor Core GEMM

本目录实现 aligned BF16 GEMM：A、B 为 BF16 row-major，C 为 FP32 row-major，累加使用 FP32。

```text
C[M, N] = A[M, K] * B[K, N]
```

第一版只支持：

```text
M % 128 == 0
N % 128 == 0
K % 64 == 0
```

核心 kernel 没有使用 WMMA、CUTLASS 或 cuBLAS。cuBLAS 只作为正确性和性能 baseline。

## Kernel 配置

```text
BM=128, BN=128, BK=64
WM=32,  WN=64
256 threads, 8 warps per CTA
```

每个 warp 计算 `32x64` 的 C tile。一个 K tile 包含 4 个 `m16n8k16` K fragment；每个 `mma_k` 先加载 2 个 A fragment和 8 个 B fragment，再展开执行 16 次 MMA：

```text
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
```

## 数据流

### 1. Global 到 Shared

同步版本使用 16B `uint4` 完成：

```text
Global -> Register -> Shared
```

连续线程读取连续 global chunk，A、B 始终保持 row-major，B 不在 global-to-shared 阶段转置。

异步版本在正确性和 swizzle 验证完成后改为 16B `cp.async`，使用两个 32KB stage：

```text
write_stage: Global -> Shared
read_stage:  Shared -> ldmatrix -> MMA
```

每轮依次执行 `commit_group`、当前 stage 的 MMA、`wait_group 0` 和 `__syncthreads`，再交换 `read_stage/write_stage`。

### 2. 128B Cyclic Swizzle

shared swizzle 以 128B 为独立区域。每个区域包含 8 个 16B chunk：

```text
segment = logical_col / 64
chunk = (logical_col % 64) / 8
physical_chunk = (chunk + row % 8) % 8
```

A 的一行正好是 128B。B 的一行是 256B，因此拆成两个独立的 128B segment，循环移动不会从前半行进入后半行，也不会跨行。

一个 16B chunk 占 4 个 bank。固定 logical chunk 时，连续 8 行的物理起始 bank 为：

```text
0, 4, 8, 12, 16, 20, 24, 28
```

### 3. Shared 到 Registers

A 使用：

```text
ldmatrix.sync.aligned.m8n8.x4.shared.b16
```

四个 8x8 子矩阵的顺序为 top-left、bottom-left、top-right、bottom-right，对应 MMA 的四个 A registers。

B 使用：

```text
ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16
```

两个 8x8 子矩阵沿 K 排列，`.trans` 生成 `B.col` 需要的两个 registers。

### 4. Accumulator 写回

对于 `m16n8k16`，定义：

```text
group = lane / 4
thread = lane % 4
```

每个 lane 的四个 FP32 accumulator 对应：

```text
C0: (group,     2*thread)
C1: (group,     2*thread+1)
C2: (group + 8, 2*thread)
C3: (group + 8, 2*thread+1)
```

当前直接使用 `float2` 写入 row-major global C，没有增加 shared C staging。

## 正确性结果

测试环境：RTX 4090、SM89、CUDA 13.0、cuBLAS，编译参数为 `-O3 -arch=sm_89`。参考结果使用 cuBLAS BF16 GEMM 和 FP32 accumulate。

PASS 条件为 `max_absolute <= 1e-3` 且 `relative_l2 <= 1e-5`，实际 Tensor Core kernel 的误差均为 0，没有通过放宽 tolerance 掩盖 layout 问题。

| 尺寸 M=N=K | Kernel | Max absolute | Mean absolute | Relative L2 | 结果 |
| ---: | --- | ---: | ---: | ---: | --- |
| 1024 | row-major | 0 | 0 | 0 | PASS |
| 1024 | cyclic swizzle | 0 | 0 | 0 | PASS |
| 1024 | cp.async | 0 | 0 | 0 | PASS |
| 2048 | row-major | 0 | 0 | 0 | PASS |
| 2048 | cyclic swizzle | 0 | 0 | 0 | PASS |
| 2048 | cp.async | 0 | 0 | 0 | PASS |
| 4096 | cp.async | 0 | 0 | 0 | PASS |

naive kernel 在 1024 尺寸上的结果为：

```text
max_abs=7.152557e-07
mean_abs=9.470443e-08
relative_l2=1.773324e-07
```

Compute Sanitizer 对 swizzle 和 cp.async 版本报告 `0 errors`。

## Bank Conflict 验证

使用 NCU 2025.3.1，在 `1024x1024x1024` 上比较同步 row-major 与 cyclic swizzle。指标均为 kernel 总计数：

| Shared layout | LDSM conflicts | LDSM wavefronts | Shared-store conflicts | Store wavefronts |
| --- | ---: | ---: | ---: | ---: |
| row-major | 5,505,024 | 6,291,456 | 0 | 262,144 |
| cyclic swizzle | 0 | 786,432 | 0 | 262,144 |

cyclic swizzle 将 LDSM bank conflict 清零，LDSM wavefront 减少到 row-major 的 `1/8`，同时没有给 Register-to-Shared 写入引入 bank conflict。

异步版本在 `2048x2048x2048` 上的 LDSM conflict 和 LDGSTS conflict 也均为 0。

## 性能结果

CUDA Event 计时，先 warmup 100 次，再连续执行 1000 次取平均：

| M=N=K | Kernel | Time (ms) | TFLOPS | 相对 cuBLAS |
| ---: | --- | ---: | ---: | ---: |
| 1024 | cuBLAS | 0.0176 | 121.949 | `1.000x` |
| 1024 | naive | 0.4217 | 5.092 | `0.042x` |
| 1024 | row-major | 0.0619 | 34.670 | `0.284x` |
| 1024 | cyclic swizzle | 0.0461 | 46.579 | `0.382x` |
| 1024 | cp.async | 0.0323 | 66.507 | `0.545x` |
| 2048 | cuBLAS | 0.1019 | 168.634 | `1.000x` |
| 2048 | naive | 3.2297 | 5.319 | `0.032x` |
| 2048 | row-major | 0.1884 | 91.212 | `0.541x` |
| 2048 | cyclic swizzle | 0.1403 | 122.412 | `0.726x` |
| 2048 | cp.async | 0.1268 | 135.443 | `0.803x` |
| 4096 | cuBLAS | 0.9249 | 148.592 | `1.000x` |
| 4096 | row-major | 1.4552 | 94.450 | `0.636x` |
| 4096 | cyclic swizzle | 1.0052 | 136.724 | `0.920x` |
| 4096 | cp.async | 0.9838 | 139.698 | `0.940x` |

swizzle 相对 row-major 在 1024、2048、4096 上分别提升约 `1.34x`、`1.34x`、`1.45x`。cp.async 在三个尺寸上继续提升约 `1.43x`、`1.11x` 和 `1.02x`；4096 已接近当前 kernel 的计算吞吐上限，异步搬运的额外收益较小。

## 资源占用

ptxas 和 NCU 对最终 cp.async kernel 的结果：

```text
registers/thread:              128
register spills:               0
dynamic shared memory/block:   65536 bytes
theoretical occupancy:         16.67%
achieved occupancy:            16.54%
active warps/SM:               7.94
```

RTX 4090 runtime 查询结果为默认 48KB、opt-in 101376B shared memory/block。64KB kernel 通过 `cudaFuncSetAttribute` 请求 opt-in shared memory。当前 occupancy 由 shared memory 限制为一个 CTA/SM。

## 复现

```bash
bash gemm/compile.sh
bash gemm/verify.sh
bash gemm/benchmark.sh
```

单独测试：

```bash
./gemm/build/gemm verify all 1024 1024 1024
./gemm/build/gemm benchmark async 2048 2048 2048 10 100
./gemm/build/gemm profile swizzle 1024 1024 1024
```

查询本机 NCU 指标后，可以使用以下指标复现 bank-conflict 对比：

```text
sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.sum
sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldsm.sum
sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_st.sum
```
