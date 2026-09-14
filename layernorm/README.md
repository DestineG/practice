# CUDA LayerNorm 优化

本目录实现 FP32 row-wise LayerNorm。输入是连续的 `[N, M]` 二维张量，每一行独立计算：

```text
mean = sum(x) / M
variance = sum((x - mean)^2) / M
y = (x - mean) / sqrt(variance + epsilon) * gamma + beta
```

代码使用 cuDNN LayerNorm 作为 baseline，并比较将行数据保存在 shared memory 和寄存器中的实现。

## Kernel 优化过程

### 1. Block + Shared Memory

`layernorm_shared` 使用一个 block 处理一行，每个线程以 `blockDim.x` 为步长处理若干元素。

输入第一次从 global memory 读取时被保存到 shared memory，同时计算线程局部和。均值规约分为两级：先使用 `__shfl_down_sync` 完成 warp 内规约，再由第一个 warp 合并各 warp 的结果。

得到均值后，各线程从 shared memory 读取输入，计算中心化结果和平方和。方差使用相同的两级规约，最后计算 `rsqrtf(variance + epsilon)`，应用 gamma 和 beta 后写回 global memory。

这个版本只读取一次 global input，也不会重复计算中心化结果，但需要保存整行数据。动态 shared memory 大小约为：

```text
(M + block_size / 32) * sizeof(float)
```

### 2. Block + Register Cache

`layernorm_register` 保留一个 block 处理一行的线程映射，但将每个线程负责的输入保存在寄存器数组中。shared memory 只保存各 warp 的局部规约结果。

归一化维度通过 `128、256、512、1024、2048、4096` 六个 bucket 在编译期实例化。固定的数组长度使循环可以展开，也避免了整行 shared-memory 读写。

register 版本仍然需要两次 block 规约和同步。随着 `M` 增大，每线程寄存器数量会上升，因此同时测试 128、256 和 512 三种 block size：较小 block 的调度开销低，但每个线程保存的数据更多；较大 block 能减少单线程工作量，却可能产生更多空闲线程和规约开销。

### 3. cuDNN Baseline

cuDNN baseline 使用 cuDNN 9 Backend Graph 的 `CUDNN_LAYER_NORM`。使用 training phase，让 cuDNN 根据当前输入计算每行均值和方差；inference phase 需要调用方提前提供统计量，不等价于这里的 LayerNorm kernel。

cuDNN training LayerNorm 同时输出 mean 和 inverse variance，这是接口要求。Graph 创建、执行计划选择和 workspace 分配都在正式计时前完成，CUDA Event 只测量执行阶段。

## 正确性测试

正确性使用 CPU double reference。输入、gamma 和 beta 由固定序列生成，每次运行一致。判定条件为：

```text
abs_error <= 2e-5 + 2e-4 * abs(reference)
```

测试矩阵覆盖小尺寸、非整齐尺寸以及每个 register bucket 的前后边界：

| 参数 | 测试值 |
| --- | --- |
| `N` | 1, 31, 512 |
| `M` | 1, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096 |

共 63 个形状，cuDNN、shared 128/256/512 和 register 128/256/512 全部通过。运行完整正确性矩阵：

```bash
bash layernorm/compile.sh
bash layernorm/verify.sh
```

也可以只验证一个形状或一个 kernel：

```bash
./layernorm/build/layernorm verify all 512 1024
./layernorm/build/layernorm verify register_256 512 1024
```

## 性能复现

性能矩阵覆盖 36 个常用形状：

| 参数 | 测试值 |
| --- | --- |
| `N` | 512, 1024, 2048, 4096, 8192, 16384 |
| `M` | 128, 256, 512, 1024, 2048, 4096 |

benchmark 默认执行 10 次 warmup，每个样本连续运行 100 次 kernel，共采集 10 个样本，输出单次 kernel latency 的 median、min 和 P90。时间不包含输入生成、内存分配、H2D/D2H 以及 cuDNN Graph 创建。

运行完整性能矩阵：

```bash
bash layernorm/benchmark.sh
```

运行单个形状，或者覆盖默认计时参数：

```bash
./layernorm/build/layernorm benchmark all 4096 1024
./layernorm/build/layernorm benchmark register_256 4096 1024 20 200 20
```

使用 Nsight Compute 采集单个 kernel：

```bash
sudo /usr/local/cuda/bin/ncu --set full ./layernorm/build/layernorm profile register_256 4096 1024
```

## 结果示例

测试环境为 NVIDIA GeForce RTX 4090、CUDA 13.0、cuDNN 9.26。下面是 `[N, M] = [4096, 1024]` 的一组结果，单位为微秒：

| Kernel | Median | Min | P90 |
| --- | ---: | ---: | ---: |
| cuDNN | 12.195 | 12.164 | 12.196 |
| shared 128 | 15.702 | 15.690 | 15.710 |
| shared 256 | 16.364 | 16.343 | 16.384 |
| shared 512 | 19.743 | 19.722 | 19.746 |
| register 128 | 12.327 | 12.301 | 12.347 |
| register 256 | 11.551 | 11.530 | 11.551 |
| register 512 | 16.027 | 16.015 | 16.036 |

这个形状上，register 256 相对 shared 256 提升约 `1.42x`，相对 cuDNN 提升约 `1.06x`。

下表对每个 `M` 下的六个 `N` 取几何平均，并选择表现最好的固定 block 配置：

| `M` | 最佳固定配置 | 相对 cuDNN |
| ---: | --- | ---: |
| 128 | register 128 | 1.063x |
| 256 | register 128 | 1.365x |
| 512 | register 128 | 1.212x |
| 1024 | register 128 | 0.961x |
| 2048 | register 256 | 0.961x |
| 4096 | register 256 | 0.991x |

register 128 跨全部 36 个形状的几何平均为 cuDNN 的 `1.059x`，在 20/36 个形状上更快。如果按 `M` 选择上表中的固定配置，整个矩阵的几何平均为 cuDNN 的 `1.082x`。这些结果来自重复访问同一组 device buffer 的 kernel benchmark，工作集能否保留在 L2 cache 会明显影响不同形状的结果。
