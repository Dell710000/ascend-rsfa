# ASCEND-RSFA

High-performance FlashAttention forward kernels for Ascend NPUs, built with TritonAscend.

[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![Platform: Ascend 910B](https://img.shields.io/badge/platform-Ascend%20910B-cd2027)](https://www.hiascend.com/)
[![TritonAscend](https://img.shields.io/badge/TritonAscend-3.2.1-4c8bf5)](https://gitee.com/ascend/triton-ascend)
[![Repository quality](https://github.com/asdfghjkl509/ascend-rsfa/actions/workflows/quality.yml/badge.svg)](https://github.com/asdfghjkl509/ascend-rsfa/actions/workflows/quality.yml)

[中文说明](README_zh-CN.md) | [Reproducibility](#reproducing-the-paper-results) | [Citation](#citation)

ASCEND-RSFA specializes FlashAttention v2 for the Ascend execution model. It combines software pipelining, persistent scheduling, compiler-pipeline tuning, and causal-region specialization while preserving the standard online-softmax algorithm.

The current release contains the forward kernel, the official TritonAscend-derived baseline, all ablation variants, correctness checks, tiling search, profiling utilities, and the raw data used by the paper.

> Status: research release. The implementation is forward-only and targets Ascend 910B3 with FP16 inputs. It is not a drop-in replacement for every PyTorch attention API.

## Highlights

| Optimization | Effect |
| --- | --- |
| K/V load-advance pipeline | Exposes address generation and data movement earlier |
| Persistent dynamic scheduling | Adapts the grid to available AI Cores and task count |
| Ascend compiler-pipeline tuning | Enables multibuffering and Cube/Vector loop mixing |
| Causal region specialization | Splits prefix and diagonal work into specialized kernels |

On the paper benchmark set, ASCEND-RSFA reaches a geometric-mean speedup of **1.74x for causal attention** and **2.12x for non-causal attention** over the TritonAscend FlashAttention v2 baseline.

| Sequence length | Head dim | Causal | Non-causal |
| ---: | ---: | ---: | ---: |
| 1,024 | 128 | 1.99x | 2.11x |
| 1,024 | 256 | 1.37x | 2.18x |
| 2,048 | 128 | 2.02x | 2.16x |
| 2,048 | 256 | 1.27x | 2.11x |
| 4,096 | 128 | 2.04x | 2.20x |
| 8,192 | 64 | 1.96x | 1.96x |

Measurements were collected on an Ascend 910B3 with CANN 8.5.0, PyTorch 2.7.1, torch_npu 2.7.1, and TritonAscend 3.2.1. See [the benchmark report](benchmarks/benchmark_results.txt) and [raw measurements](benchmarks/final_benchmark_raw.json).

## Repository layout

```text
ascend_rsfa/   Production ASCEND-RSFA forward kernel
baseline/      TritonAscend FlashAttention v2 baseline
ablation/      Five cumulative optimization variants and runner
benchmarks/    Correctness, latency, raw data, and reports
tiling/        Tiling search and validation utilities
profiling/     torch_npu profiler and msprof analysis tools
```

## Requirements

- Linux on AArch64
- Ascend 910B3 NPU
- CANN 8.5.0
- Python 3.10+
- PyTorch 2.7.1 and torch_npu 2.7.1
- TritonAscend 3.2.1

Install CANN, PyTorch/torch_npu, and TritonAscend from their platform-specific distributions first. They are intentionally not downloaded by this package because the correct wheels depend on the CANN and AArch64 environment.

```bash
git clone https://github.com/asdfghjkl509/ascend-rsfa.git
cd ascend-rsfa
python -m pip install -e .
```

Before running a kernel, verify the device stack:

```bash
python -c "import torch, torch_npu, triton; print(torch.__version__, triton.__version__, torch.npu.is_available())"
npu-smi info
```

## Usage

```python
import torch

from ascend_rsfa.flash_attention_forward import attention, get_tiling

batch, heads, sequence, head_dim = 128, 8, 2048, 128
causal = True
q = torch.randn(batch, heads, sequence, head_dim, device="npu", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

block_m, block_n = get_tiling(batch, heads, sequence, head_dim, causal)
output = attention(q, k, v, causal, 0.5, block_m, block_n)
```

`get_tiling` contains configurations validated by this release. For a new shape, pass a tested `(BLOCK_M, BLOCK_N)` pair directly or run the tiling search before production use.

## Reproducing the paper results

Run commands from the repository root:

```bash
# Numerical correctness against torch_npu.npu_fusion_attention
python benchmarks/run_correctness_test.py

# Final baseline versus ASCEND-RSFA latency table
python benchmarks/run_final_benchmark.py

# All five cumulative ablation variants
python ablation/run_ablation.py --mode timing

# Example tiling search
python tiling/search_optimal_tiling.py \
  --kernel ascend_rsfa/flash_attention_forward.py \
  --kernel-name ascend-rsfa \
  --output tiling/tiling_search_optimized.json
```

Hardware profiling additionally requires `msprof`. Generated profiling directories are ignored by Git; the summarized paper data remains versioned under `profiling/`.

## Correctness

The released correctness report covers 12 causal and non-causal configurations against `torch_npu.npu_fusion_attention`. The reported maximum absolute error is below `9.77e-4` with `atol=1e-2` and `rtol=1e-2`. See [correctness_test_results.txt](benchmarks/correctness_test_results.txt).

## Citation

The paper venue metadata is not included in this release yet. Until the final bibliographic record is available, cite the manuscript and the software:

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

Machine-readable metadata is available in [CITATION.cff](CITATION.cff).

## License and attribution

ASCEND-RSFA is released under the [BSD 3-Clause License](LICENSE). The baseline in `baseline/06-fused-attention.py` is derived from TritonAscend and retains its original MIT license notice. See [NOTICE](NOTICE) for third-party attribution.

## Contact

- Jie Ren, corresponding author: `renjie@snnu.edu.cn`
- Jingyan Zhao and Yiqi Liu, co-first authors
