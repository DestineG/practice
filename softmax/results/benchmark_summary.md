# Softmax Benchmark Summary

本文档是从本地 `build/ncu_summary_detail.md` 提炼出的可提交结果。原始数据包含
30 个输入形状、cuDNN 和 9 个自定义 kernel，共 300 条 kernel 记录；原始
`.ncu-rep` 与详细表格继续保存在被 Git 忽略的 `build/` 目录中。

## 数据口径

- 报告时间：2026-09-13
- 输入形状：6 个 batch × 5 个 num_classes，共 30 个形状
- 指标：Nsight Compute `gpu__time_duration.sum`，统一换算为微秒
- 汇总方法：先在每个形状上计算 duration 比值，再取几何平均
- cuDNN：`CUDNN_SOFTMAX_ACCURATE`、`CUDNN_SOFTMAX_MODE_CHANNEL`
- 所有实现使用固定种子 42 生成的相同 FP32 输入

环境和测量限制见 [README](../README.md#实验环境与口径)。

## 优化阶段汇总

“相对父版本”中的父版本按优化关系定义：`warp` 与 `vectorized` 分别从 naive
验证行内并行和向量化；block shared 是 warp vectorized 面向宽行的替代方案，
不是 online warp 的直接后继。

| Kernel | 父版本 | 相对父版本 | 相对 naive | 相对 cuDNN | 快于 cuDNN |
| --- | --- | ---: | ---: | ---: | ---: |
| `naive` | - | - | 1.0000x | 0.0401x | 0/30 |
| `vectorized` | naive | 2.8657x | 2.8657x | 0.1148x | 0/30 |
| `warp` | naive | 19.7976x | 19.7976x | 0.7933x | 1/30 |
| `warp_vectorized` | warp | 1.0973x | 21.7242x | 0.8705x | 6/30 |
| `warp_vectorized_online` | warp vectorized | 1.0555x | 22.9298x | 0.9188x | 6/30 |
| `block_shared<128>` | warp vectorized | 1.2679x | 27.5447x | 1.1037x | 23/30 |
| `block_register<128>` | block shared 128 | 1.0281x | 28.3184x | 1.1347x | 26/30 |
| `block_register<256>` | register 128 | 0.9836x | 27.8548x | 1.1162x | 22/30 |

固定使用单个自定义实现时，`block_register<128>` 在整个测试集上最稳健。
`block_register<256>` 的总体回退说明 block size 应按形状选择。

## 按 Num Classes 汇总

下表对每个 `num_classes` 下的六个 batch 取几何平均，并选择一个固定实现。
`C=256` 时最快的自定义候选仍略慢于 cuDNN，因此建议保留 cuDNN。

| Num classes | 最佳固定选择 | 相对 cuDNN |
| ---: | --- | ---: |
| 256 | cuDNN | 1.0000x |
| 512 | `block_register<128>` | 1.0810x |
| 1024 | `block_register<256>` | 1.2354x |
| 2048 | `block_register<256>` | 1.2347x |
| 4096 | `block_register<256>` | 1.2979x |

按这组规则进行静态选择，在当前 30 个形状上的几何平均为 cuDNN 的
`1.1644x`。这是根据现有数据得到的分析结果，代码中尚未实现该 dispatch。

## 代表形状

以下是 `(batch, num_classes) = (4096, 1024)` 的完整 kernel 对比。Compute、
Memory 和 Occupancy 均为 Nsight Compute 报告的峰值持续吞吐或活跃度百分比。

| Kernel | Duration (us) | 相对 cuDNN | Compute (%) | Memory (%) | Registers/thread | Occupancy (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cuDNN | 23.616 | 1.000x | 72.08 | 73.25 | 32 | 91.48 |
| `naive` | 515.904 | 0.046x | 0.91 | 27.13 | 40 | 16.63 |
| `vectorized` | 159.392 | 0.148x | 2.38 | 32.57 | 38 | 16.41 |
| `warp` | 30.432 | 0.776x | 16.15 | 54.82 | 39 | 58.20 |
| `warp_vectorized` | 28.448 | 0.830x | 14.79 | 58.65 | 38 | 59.78 |
| `warp_vectorized_online` | 26.944 | 0.876x | 17.07 | 62.07 | 36 | 61.68 |
| `block_shared<128>` | 20.192 | 1.170x | 21.98 | 82.76 | 28 | 87.44 |
| `block_shared<256>` | 20.032 | 1.179x | 28.27 | 83.43 | 28 | 89.77 |
| `block_register<128>` | 20.512 | 1.151x | 18.51 | 81.35 | 30 | 88.55 |
| `block_register<256>` | 19.680 | 1.200x | 23.99 | 84.95 | 23 | 90.22 |

这个形状展示了整体趋势：只做 `float4` 无法解决低并行度；warp 映射带来最大
单步提升；宽行进一步受益于 block 级并行和片上缓存。它也说明单个形状上的
register 版本不一定同时优于 shared 版本的所有 block size，参数选择必须基于
目标输入分布。

## 解释限制

- 这些 duration 来自 NCU profiler，不是多次 CUDA Event latency 的统计值。
- 当前 benchmark 没有 warmup、重复测量、误差条或输出正确性比较。
- `DRAM Read/Write` 未进入精简结果，因为当前汇总脚本保留了 NCU 数值却丢失
  每行原始单位，固定标注为 B 会造成 Kbyte/Mbyte 数据被误读。
- 所有结论只覆盖 FP32、`num_classes % 4 == 0` 且 `num_classes <= 4096` 的
  当前测试范围。
