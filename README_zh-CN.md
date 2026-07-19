# ASCEND-RSFA

面向昇腾 NPU、基于 TritonAscend 实现的高性能 FlashAttention 前向算子。

[English](README.md) | [复现实验](#复现论文结果) | [引用](#引用)

ASCEND-RSFA 针对昇腾执行模型，对 FlashAttention v2 的软件流水、任务调度、编译流水和因果区域进行联合优化。仓库包含最终 kernel、官方基线、完整消融版本、正确性验证、分块搜索、性能剖析工具和论文原始数据。

> 当前版本属于科研发布：仅包含前向计算，主要面向 Ascend 910B3 和 FP16，不是所有 PyTorch Attention 接口的即插即用替代品。

## 核心优化

| 优化 | 作用 |
| --- | --- |
| K/V load-advance | 提前暴露地址生成与数据搬运 |
| 动态持久化调度 | 根据 AI Core 数量和任务数动态选择 grid |
| 编译流水配置 | 启用 multibuffer 和 Cube/Vector 循环混合 |
| 因果区域特化 | 将 Prefix 与 Diagonal 区域拆为专用 kernel |

论文测试集上，相对 TritonAscend FlashAttention v2 基线，ASCEND-RSFA 的几何平均加速比为：

- 因果注意力：**1.74x**
- 非因果注意力：**2.12x**

详细结果见 [性能报告](benchmarks/benchmark_results.txt)、[原始数据](benchmarks/final_benchmark_raw.json) 和 [正确性报告](benchmarks/correctness_test_results.txt)。

## 环境

| 组件 | 版本 |
| --- | --- |
| NPU | Ascend 910B3 |
| OS | Ubuntu 22.04 / AArch64 |
| CANN | 8.5.0 |
| PyTorch | 2.7.1 |
| torch_npu | 2.7.1 |
| TritonAscend | 3.2.1 |
| Python | 3.10+ |

请先按昇腾平台要求安装 CANN、PyTorch/torch_npu 和 TritonAscend。由于安装包与 CANN、架构强相关，本项目不会从普通 PyPI 自动安装这些运行时依赖。

```bash
git clone https://github.com/asdfghjkl509/ascend-rsfa.git
cd ascend-rsfa
python -m pip install -e .
```

## 使用

```python
import torch

from ascend_rsfa.flash_attention_forward import attention, get_tiling

Z, H, N, D = 128, 8, 2048, 128
causal = True
q = torch.randn(Z, H, N, D, device="npu", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

BM, BN = get_tiling(Z, H, N, D, causal)
out = attention(q, k, v, causal, 0.5, BM, BN)
```

`get_tiling` 只收录本版本已经验证的配置。对于新的 shape，请显式传入已验证的 `(BLOCK_M, BLOCK_N)`，或先运行分块搜索。

## 目录

```text
ascend_rsfa/   最终 ASCEND-RSFA kernel
baseline/      TritonAscend FlashAttention v2 基线
ablation/      6 个逐级叠加的消融版本及运行脚本
benchmarks/    正确性、性能、原始数据和报告
tiling/        分块搜索与复测脚本
profiling/     torch_npu profiler 与 msprof 工具
```

## 复现论文结果

在仓库根目录执行：

```bash
python benchmarks/run_correctness_test.py
python benchmarks/run_final_benchmark.py
python ablation/run_ablation.py --mode timing

python tiling/search_optimal_tiling.py \
  --kernel ascend_rsfa/flash_attention_forward.py \
  --kernel-name ascend-rsfa \
  --output tiling/tiling_search_optimized.json
```

硬件 profiling 还需要安装 `msprof`。生成的大体积 profiling 目录默认不会提交，论文汇总数据保留在 `profiling/`。

## 消融版本

| 版本 | 文件 | 新增优化 |
| --- | --- | --- |
| Official | `ablation/01_official.py` | 官方基线 |
| V1 | `ablation/02_v1_kv_advance.py` | K/V 地址提前推进 |
| V2 | `ablation/03_v2_numcores.py` | 动态 AI Core 调度 |
| V3 | `ablation/04_v3_pipeline_optimized.py` | 编译流水配置 |
| ASCEND-RSFA | `ablation/05_ascend_rsfa.py` | 因果 Prefix/Diagonal 拆分 |

## 引用

论文正式出版信息确定后，应更新以下条目；当前可引用手稿与软件：

```bibtex
@misc{zhao2026ascendrsfa,
  title  = {{ASCEND-RSFA}: Execution-Pipeline and Causal-Region Specialization
            for {FlashAttention} on {Ascend NPUs}},
  author = {Jingyan Zhao and Yiqi Liu and Ni Zhang and Xiang Li and XiaoAo Zhu
            and Zhiqiang Li and Feng Tian and Jie Zheng and Rui Cao and Jie Ren},
  year   = {2026},
  note   = {Manuscript and open-source software}
}
```

## 许可证与联系方式

项目使用 [BSD 3-Clause License](LICENSE)。`baseline/06-fused-attention.py` 保留其原始 MIT 许可证声明，第三方信息见 [NOTICE](NOTICE)。

- 通讯作者：Jie Ren，`renjie@snnu.edu.cn`
- 共同一作：Jingyan Zhao、Yiqi Liu
