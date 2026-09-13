# CUDA Softmax 优化

本目录实现 FP32 row-wise softmax。输入是连续的 `[batch, num_classes]` 二维张量，每一行独立计算：

```text
max_value = max(x)
sum = sum(exp(x - max_value))
y = exp(x - max_value) / sum
```

减去最大值可以避免 `exp` 溢出。代码以 cuDNN 为基线，从简单的串行实现逐步优化线程映射、访存和中间结果复用。

## Kernel 优化过程

### 1. Naive

`softmax_naive` 使用一个线程处理一整行，依次完成 max、sum 和归一化。

这个版本结构简单，但存在三个主要问题：一行内部完全串行；同一个输入最多读取三次；一个 warp 中的线程分别访问不同行，访存地址间隔为 `num_classes`，无法形成理想的合并访存。

### 2. Vectorized

`softmax_vectorized` 保持一个线程处理一行，但将标量读写改为 `float4`。每条访存指令处理四个 float，减少了访存指令和循环次数。

它没有解决行内串行和跨行访存问题，所以相比 naive 有提升，但仍不是合适的线程映射方式。该实现要求 `num_classes` 是 4 的倍数。

### 3. Warp

`softmax_warp` 改为一个 warp 处理一行。每个 lane 负责一部分元素，warp 内使用 `__shfl_down_sync` 完成 max 和 sum 规约。

这样一行的计算由 32 个线程并行完成，相邻 lane 也会访问相邻元素。相比单纯使用 `float4`，改变线程映射带来的提升更明显。

### 4. Warp + Vectorized

`softmax_warp_vectorized` 在一个 warp 处理一行的基础上使用 `float4`。每个 lane 一次读取四个连续元素，同时保留 warp 内合并访存和 shuffle 规约。

这个版本同时利用行内并行和向量化，进一步减少 load、store 和循环控制指令。

### 5. Online Softmax

`softmax_warp_vectorized_online` 使用 `(max, sum)` 状态合并 max 和 sum 的计算：

```text
m = max(m1, m2)
s = s1 * exp(m1 - m) + s2 * exp(m2 - m)
```

因此只需要一次输入遍历就能得到最大值和指数和，整个 kernel 对输入的读取由三次减少为两次。它会增加一些指数运算，所以收益会随 `num_classes` 和硬件特征变化。

### 6. Block + Shared Memory

`softmax_block_vectorized_shared` 使用一个 block 处理一行，适合类别数较大的输入：

1. 使用 `float4` 将输入加载到 shared memory。
2. 先进行 warp 内规约，再通过 shared memory 完成 warp 间规约。
3. 将 `exp(x - max)` 暂存在 shared memory，归一化时直接复用。
4. 最后将结果写回 global memory。

相比一个 warp 处理一行，它提供了更多行内并行度，并避免重复读取输入和重复计算指数。代价是需要 block 同步以及随 `num_classes` 增长的 shared memory。

### 7. Block + Register Cache

`softmax_block_vectorized_register` 将每个线程负责的 `float4` 保存在寄存器数组中，shared memory 只保存各 warp 的局部规约结果。

这样可以去掉用于缓存整行数据的 shared memory，并减少 shared-memory 读写。模板参数给出类别数上界，使循环可以在编译期展开。类别数增大时，寄存器占用也会上升，因此代码同时保留 128 和 256 线程版本用于比较。

### 8. Latest Register Version

`softmax_block_vectorized_register_latest` 在 register 版本上继续做了几项小优化：加载输入时同时计算线程内最大值；使用树形方式规约 `float4`；使用 `__expf`；提前计算 `1 / sum`，归一化时使用乘法。

这些修改不改变整体算法，主要减少局部指令和除法开销。

## 复现

在仓库根目录编译：

```bash
bash softmax/compile.sh
```

先使用 CPU double 结果验证所有 kernel：

```bash
./softmax/build/softmax verify all 512 1024
```

使用 CUDA Event 测量所有 kernel。默认执行 10 次 warmup，每个样本运行 100 次 kernel，共采集 10 个样本，输出 median、min 和 P90：

```bash
./softmax/build/softmax benchmark all 512 1024
```

也可以只测一个 kernel，并在末尾指定 warmup 次数、每个样本的迭代次数和样本数：

```bash
./softmax/build/softmax benchmark block_register_latest_256 512 1024 20 200 20
```

使用 Nsight Compute 采集单个 kernel：

```bash
sudo /usr/local/cuda/bin/ncu --set full ./softmax/build/softmax profile block_register_latest_256 512 1024
```

批量采集预设的 batch 和 num_classes，并生成汇总表：

```bash
bash softmax/benchmark.sh
bash softmax/summary.sh brief
bash softmax/summary.sh detail
```

## 结果示例

下面是 RTX 4090 上对 `[512, 1024]` 输入运行 benchmark 的一组结果，单位为微秒：

```bash
./softmax/build/softmax benchmark all 512 1024 2 5 3
```

结果只表示本机的一次运行，其他 GPU 或系统状态下会有差异。

| Kernel | Median | Min | P90 |
| --- | ---: | ---: | ---: |
| cuDNN | 4.883 | 4.710 | 5.120 |
| naive | 455.066 | 454.586 | 455.302 |
| vectorized | 129.638 | 129.434 | 130.662 |
| warp | 5.293 | 5.120 | 5.325 |
| warp vectorized | 4.506 | 4.301 | 4.506 |
| warp vectorized online | 4.448 | 4.301 | 4.499 |
| block shared 128 | 3.840 | 3.686 | 3.840 |
| block shared 256 | 3.667 | 3.610 | 3.866 |
| block register 128 | 3.482 | 3.475 | 3.654 |
| block register 256 | 3.482 | 3.462 | 3.482 |
| block register latest 128 | 3.302 | 3.277 | 3.443 |
| block register latest 256 | 3.469 | 3.277 | 3.482 |

这组结果使用 `warmup=2`、`iterations=5`、`samples=3`，用于快速展示输出格式。正式比较建议增加 warmup、迭代次数和样本数。
