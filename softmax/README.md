# CUDA Softmax 优化

本目录实现并比较了一组 FP32 row-wise softmax kernel。输入被视为形状为
`[batch, num_classes]` 的连续二维张量，每一行独立计算：

```text
m_i = max_j(x_ij)
y_ij = exp(x_ij - m_i) / sum_k(exp(x_ik - m_i))
```

减去行最大值用于避免指数溢出。cuDNN 的
`CUDNN_SOFTMAX_ACCURATE + CUDNN_SOFTMAX_MODE_CHANNEL` 作为性能参考。

## 结论摘要

- 从一个线程串行处理一行的 naive kernel 出发，最终固定使用
  `block_register<128>` 时，跨 30 个测试形状的几何平均性能达到 naive 的
  `28.32x`，达到 cuDNN 的 `1.135x`，并在 26/30 个形状上快于 cuDNN。
- `num_classes` 决定了主要优化策略。窄行上 cuDNN 的启动和调度效率更好；
  `num_classes >= 512` 后，一个 block 处理一行的方案整体更有优势。
- `256` 线程并不总比 `128` 线程快。`block_register<256>` 在宽行上更好，
  但跨全部形状反而比 `block_register<128>` 慢约 `1.6%`。
- 若仅根据 `num_classes` 在 cuDNN、`block_register<128>` 和
  `block_register<256>` 之间选择，现有数据上的几何平均可达到 cuDNN 的
  `1.164x`。这只是基于测量结果的 dispatch 建议，当前代码尚未实现该分发器。

完整的精简数据见 [results/benchmark_summary.md](./results/benchmark_summary.md)。

## 优化路线

```text
naive：一个线程处理一行
├── vectorized：保持线程映射，使用 float4 访存
└── warp：一个 warp 处理一行
    └── warp + vectorized：合并行内并行与向量化访存
        ├── online max/sum：合并 max 和 sum 的输入遍历
        └── block + shared：一个 block 处理宽行并缓存中间结果
            └── block + register：将行数据和 exp 结果保存在寄存器中
```

online warp 和 block kernel 是两条面向不同工作量的策略。后者不是在前者上
简单追加一个优化，而是用更高的行内并行度和片上缓存换取额外同步成本。

## 优化过程

### 0. Naive：一个线程处理一行

[`softmax_naive`](./softmax.cu#L93) 让一个线程依次完成 max、sum 和输出三个
循环。实现简单且访存连续，但一行内部完全串行：

- 每个输入元素从 global memory 读取三次；
- `expf` 在求和和写回阶段重复计算；
- block 中不同线程访问不同的行，warp 内同一条 load 指令的地址跨度为
  `num_classes`，不利于合并访存；
- 当 batch 较小时，可供调度的线程数量也不足。

它是算法正确结构的起点，但跨测试集只有 cuDNN 约 `4.0%` 的性能。

### 1. Vectorized：使用 `float4`

[`softmax_vectorized`](./softmax.cu#L153) 每次加载和写回四个 float，在不改变
“一个线程处理一行”的前提下减少内存指令和循环控制开销。

跨 30 个形状，它相对 naive 的几何平均加速为 `2.87x`。但行内仍然串行，
而且 warp 内线程仍跨行访问，因此只达到 cuDNN 的 `0.115x`。这说明向量化
缓解了指令开销，却没有解决主要的并行度问题。

该实现要求 `num_classes % 4 == 0`，输入和输出地址满足 `float4` 对齐。

### 2. Warp：一个 warp 处理一行

[`softmax_warp`](./softmax.cu#L252) 将一行分给 32 个 lane。每个 lane 以
32 为步长处理元素，warp 内通过 shuffle 完成 max 和 sum 规约。

这样既增加了行内并行度，也让相邻 lane 访问相邻元素。相对 naive 的几何
平均加速达到 `19.80x`，远高于单独向量化的收益，说明线程映射和合并访存
是 naive kernel 的首要瓶颈。

### 3. Warp + Vectorized

[`softmax_warp_vectorized`](./softmax.cu#L316) 在 warp-per-row 的基础上使用
`float4`。每个 lane 一次处理四个连续元素，继续减少 load/store 和循环指令。

该版本相对普通 warp kernel 再提升 `1.10x`，相对 naive 达到 `21.72x`。
收益明显小于第一阶段的 `float4`，因为 warp 映射已经改善了合并访存，向量化
此时主要减少指令数。

### 4. Online Softmax：合并 max 和 sum

[`softmax_warp_vectorized_online`](./softmax.cu#L429) 使用可结合的
`(max, sum)` 状态：

```text
m = max(m1, m2)
s = s1 * exp(m1 - m) + s2 * exp(m2 - m)
```

这样可在一次输入遍历中同时得到行最大值和指数和，将整个 kernel 的 global
memory 输入遍历从三次减少到两次。代价是状态合并需要额外指数计算。

该版本相对 warp + vectorized 提升 `1.06x`，相对 naive 达到 `22.93x`。
在较窄的行上，减少一次遍历带来的收益容易被额外计算和固定开销抵消；随着
`num_classes` 增大，收益更明显。

### 5. Block + Shared Memory：适配宽行

[`softmax_block_vectorized_shared`](./softmax.cu#L527) 改为一个 block 处理一行：

1. 以 `float4` 将整行从 global memory 加载到 shared memory；
2. 先在 warp 内规约，再通过少量 shared memory 完成 warp 间规约；
3. 将 `exp(x - max)` 写回 shared memory，归一化时直接复用；
4. 最后只进行一次 global-memory 写回。

它比 warp + vectorized 的几何平均性能高 `1.27x`，相对 naive 达到
`27.54x`，并在 23/30 个形状上超过 cuDNN。`num_classes` 越大，一个 warp
独自处理一行的循环次数越多，block 级并行的优势越明显。

代价是多次 `__syncthreads()` 以及随 `num_classes` 增长的动态 shared memory。
当前 launcher 因此将支持范围限制为 `num_classes <= 4096`。

### 6. Block + Register Cache

[`softmax_block_vectorized_register`](./softmax.cu#L687) 将每个线程负责的
`float4` 保存在寄存器数组中，shared memory 只用于交换各 warp 的局部归约
结果。循环由模板参数给出上界并展开。

相对 shared-memory 版本，`block_register<128>` 的几何平均收益为 `1.03x`；
相对 naive 为 `28.32x`，相对 cuDNN 为 `1.135x`。提升已经较小，原因是此时
主要 global-memory 往返已经消除，继续优化面对的是同步、指数计算和资源占用。

寄存器缓存也不是免费的。模板上界随 `num_classes` 增大，可能提高每线程寄存器
数量并限制 occupancy。因此实际使用时需要同时选择类别区间和 block size。

### 7. Block Size：128 与 256

`128` 线程跨全部形状更稳健；`256` 线程能给宽行提供更多并行度，却会在窄行
产生空闲线程和额外规约开销。现有数据建议：

| `num_classes` | 当前数据上的选择 | 相对 cuDNN 的几何平均性能 |
| ---: | --- | ---: |
| 256 | cuDNN | 1.000x |
| 512 | `block_register<128>` | 1.081x |
| 1024 | `block_register<256>` | 1.235x |
| 2048 | `block_register<256>` | 1.235x |
| 4096 | `block_register<256>` | 1.298x |

这张表是在每个 `num_classes` 下对六个 batch 取几何平均后选择固定实现，不代表
所有未测试形状都应使用相同阈值。

## 总体结果

下表的“相对父版本”按优化关系选择比较对象，而不是机械地比较源码中的前一个
kernel。所有汇总均为 30 个形状上的几何平均。

| Kernel | 比较对象 | 相对父版本 | 相对 naive | 相对 cuDNN | 快于 cuDNN |
| --- | --- | ---: | ---: | ---: | ---: |
| `naive` | - | - | 1.000x | 0.040x | 0/30 |
| `vectorized` | naive | 2.866x | 2.866x | 0.115x | 0/30 |
| `warp` | naive | 19.798x | 19.798x | 0.793x | 1/30 |
| `warp_vectorized` | warp | 1.097x | 21.724x | 0.871x | 6/30 |
| `warp_vectorized_online` | warp vectorized | 1.055x | 22.930x | 0.919x | 6/30 |
| `block_shared<128>` | warp vectorized | 1.268x | 27.545x | 1.104x | 23/30 |
| `block_register<128>` | block shared 128 | 1.028x | 28.318x | 1.135x | 26/30 |
| `block_register<256>` | register 128 | 0.984x | 27.855x | 1.116x | 22/30 |

以 `(batch, num_classes) = (4096, 1024)` 为例，naive、online warp、
`block_register<256>` 的 NCU kernel duration 分别为 `515.904 us`、
`26.944 us` 和 `19.680 us`；cuDNN 为 `23.616 us`。同一形状上，NCU 报告的
Memory Throughput 从 naive 的 `27.13%` 提升到 register kernel 的
`84.95%`，说明线程映射、合并访存和片上复用确实提高了内存系统利用率。

不要单独把 occupancy 当成优化目标。它适合解释资源约束，最终仍应以 kernel
duration 和目标输入分布为准。

## 实验环境与口径

本次报告生成于 2026-09-13，环境如下：

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090, compute capability 8.9, 24 GiB；主机有两张同型号 GPU，单次报告使用一张 |
| Driver | 595.84 |
| CUDA Toolkit / nvcc | 13.0 / 13.0.88 |
| cuDNN | 9.26.0 |
| Nsight Compute | 2025.3.1.0 |
| 编译选项 | `-O3 -std=c++17 -arch=sm_89 -lineinfo` |
| 数据类型 | FP32 |
| 输入分布 | 固定随机种子 42，均匀分布 `[-5, 5]` |
| Batch | 512, 1024, 2048, 4096, 8192, 16384 |
| Num classes | 256, 512, 1024, 2048, 4096 |

每个形状通过 `ncu --set full` 采集，各表中的 Duration 是 NCU 记录的 kernel
duration，不包含分配、主机与设备间拷贝等端到端开销。加速比定义为：

```text
speedup = reference_duration / candidate_duration
```

跨形状结果使用几何平均，使不同量级输入获得相同权重。原始 `.ncu-rep` 和
300 行详细表格保存在本地 `build/` 下，该目录被 Git 忽略；仓库只保留
[精简结果](./results/benchmark_summary.md)。

## 编译与复现

在仓库根目录执行：

```bash
bash softmax/compile.sh
bash softmax/benchmark.sh
bash softmax/summary.sh brief
bash softmax/summary.sh detail
```

可以通过环境变量覆盖编译器、架构和报告路径：

```bash
NVCC_BIN=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_89 bash softmax/compile.sh
REPORT_DIR=/path/to/ncu_reports bash softmax/summary.sh detail
```

`benchmark.sh` 使用 `sudo ncu`，运行前需要具备性能计数器权限。完整测试会为
30 个形状采集 full metric set，耗时和报告体积都明显大于普通 latency 测试。

## 正确性与适用范围

当前实现和结果有以下边界：

- 向量化 kernel 仅处理 `num_classes % 4 == 0` 且满足 `float4` 对齐的输入；
- block kernel 仅处理 `num_classes <= 4096`；
- 不支持的输入会直接返回，尚未实现通用 fallback；
- benchmark 当前没有将自定义 kernel 输出与 cuDNN/CPU reference 做误差比较；
- 每个进程只显式发起一次各版本 kernel，没有 warmup、多次独立计时和方差统计；
- NCU 适合分析 kernel 指标，但正式 latency 结论还应补充 CUDA Event 的多次测量；
- 当前汇总脚本没有统一转换 NCU 的 DRAM byte metric 单位，因此本文不引用详细
  表中的 `DRAM Read/Write` 数值。

因此，这份结果支持当前测试范围内的性能比较，但还不能视为生产级 softmax
实现的完整正确性与性能证明。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| [`softmax.cu`](./softmax.cu) | cuDNN 基线、所有自定义 kernel 与测试入口 |
| [`compile.sh`](./compile.sh) | 编译 CUDA 程序 |
| [`benchmark.sh`](./benchmark.sh) | 对 30 个输入形状采集 NCU 报告 |
| [`summary.sh`](./summary.sh) | 将 `.ncu-rep` 转为 Markdown 表格 |
| [`results/benchmark_summary.md`](./results/benchmark_summary.md) | 可提交的精简实验结果 |

