# Custom GEMM 问题记录

## inline 与 helper 的结果差异

`gemm_custom.cu` 中保留了两个独立 kernel 实例：

```text
inline: 直接在 kernel 循环中计算 ldmatrix 起始地址
helper: 将同一套计算逐句放入 __forceinline__ helper
```

两条路径的地址计算逻辑一致。

A 路径都执行：

```text
matrix = lane / 8
matrix_row = lane % 8
row = warp_row + warp_mma_m * 16 + (matrix % 2) * 8 + matrix_row
col = mma_k * 16 + (matrix / 2) * 8
byte16_idx = (row * BK + col) / 8
swizzle_byte16_offset(byte16_idx, BK / 8)
```

B 路径都执行：

```text
matrix = (lane / 8) % 2
matrix_row = lane % 8
row = mma_k * 16 + matrix * 8 + matrix_row
col = warp_col + warp_mma_n * 8
byte16_idx = (row * BN + col) / 8
swizzle_byte16_offset(byte16_idx, BN / 8)
```

两者使用相同的 `As_read`、`Bs_read`、tile 参数、swizzle 函数、`ldmatrix` 和 MMA 路径。`if constexpr` 保证两个模板实例中只保留各自的一条分支。

## 可复现实验

编译两个 kernel：

```bash
bash gemm/compile_custom.sh
```

分别运行：

```bash
./gemm/build/gemm_custom verify inline 128 128 64
./gemm/build/gemm_custom verify helper 128 128 64
./gemm/build/gemm_custom verify inline 2048 2048 2048
./gemm/build/gemm_custom verify helper 2048 2048 2048
```

当前 RTX 4090、SM89、CUDA 13.0 的结果：

```text
128x128x64
inline  FAIL
helper  PASS

2048x2048x2048
inline  FAIL
helper  PASS
```

典型误差：

```text
inline 128x128x64:
max_abs=8.834214e-01
mean_abs=2.701213e-01
rel_l2=9.742146e-01

inline 2048x2048x2048:
max_abs=4.995650e+00
mean_abs=1.282092e+00
rel_l2=1.042697e+00
```

两个 kernel 都由同一次编译生成，ptxas 资源报告均为：

```text
162 registers/thread
0 spill
```

## 当前结论

已经排除以下原因：

- 16B global-to-shared chunk 分配错误；
- A/B tile 的 `byte16_idx` 计算错误；
- cyclic swizzle 公式错误；
- helper 传入的 warp/tile 参数错误；
- A/B 的 `ldmatrix` 形式不同；
- stage、MMA 或 accumulator store 路径不同。

当前能确认的是：源码地址计算逻辑相同，但两个模板实例的实际运行结果不同。helper 实例正确，inline 实例错误。还没有通过 PTX/SASS 对比定位 inline 实例的具体错误指令、寄存器依赖或编译器优化行为，因此不能把问题进一步确定为某一个已证实的 nvcc bug。
