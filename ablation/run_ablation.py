#!/usr/bin/env python3
"""
FlashAttention V2 消融实验测试脚本
====================================
按照消融实验设置，分别测试非因果和因果路径下各优化版本的性能。

依次运行 01_official.py 到 05_ascend_rsfa.py，对应论文消融表中的
Official、V1、V2、V3 和 ASCEND-RSFA 五个版本。

用法:
  # 纯 Python 计时模式
  python3 run_ablation.py --mode timing

  # msprof 模式 (通过 msprof 调用)
  python3 run_ablation.py --mode msprof --output-dir ./ablation_msprof

输出:
  - ablation_report_non_causal.txt  (非因果消融报告)
  - ablation_report_causal.txt      (因果消融报告)
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import torch
import torch_npu

DEVICE = "npu"
WORKSPACE = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 测试用例定义
# ============================================================================
# 非因果测试用例 (Z, H, N_CTX, HEAD_DIM, causal, dtype_str, BM, BN, label)
NON_CAUSAL_CASES = [
    (128, 8, 1024, 128, False, "fp16", 128, 256, "nc_1024_128"),  # 最优分块
    (128, 8, 1024, 256, False, "fp16", 64,  128, "nc_1024_256"),
    (128, 8, 2048, 128, False, "fp16", 128, 256, "nc_2048_128"),  # 最优分块
    (128, 8, 2048, 256, False, "fp16", 64,  256, "nc_2048_256"),
    (128, 8, 4096, 128, False, "fp16", 128, 256, "nc_4096_128"),
    (128, 8, 8192, 64,  False, "fp16", 128, 512, "nc_8192_64"),
]

# 因果测试用例 (shape定义, BM/BN由各variant独立指定)
CAUSAL_SHAPES = [
    (128, 8, 1024, 128, True,  "fp16", "c_1024_128"),
    (128, 8, 1024, 256, True,  "fp16", "c_1024_256"),
    (128, 8, 2048, 128, True,  "fp16", "c_2048_128"),
    (128, 8, 2048, 256, True,  "fp16", "c_2048_256"),
    (128, 8, 4096, 128, True,  "fp16", "c_4096_128"),
    (128, 8, 8192, 64,  True,  "fp16", "c_8192_64"),
]

# 各因果variant的独立最优分块
FUSED_CAUSAL_TILING = {    # STAGE=3 (单kernel, 不能太大否则UB溢出)
    "c_1024_128": (128, 64),
    "c_1024_256": (64,  64),   # (64,256) UB溢出, (64,64)为最优可行
    "c_2048_128": (128, 64),
    "c_2048_256": (64,  64),   # (64,256) UB溢出, (64,64)为最优可行
    "c_4096_128": (128, 64),
    "c_8192_64":  (128, 64),
}

FUSED_SPLIT_CAUSAL_TILING = {  # STAGE=4/5 (拆分为两个kernel, 允许更大BN)
    "c_1024_128": (128, 64),
    "c_1024_256": (64,  256),  # 最优分块
    "c_2048_128": (128, 64),
    "c_2048_256": (64,  256),  # 最优分块
    "c_4096_128": (128, 64),   # (128,32)更慢
    "c_8192_64":  (128, 64),   # (64,256)略慢
}

# Baseline 的 tiling 参数（可能和优化版不同，因为 UB 限制）
BASELINE_NC_TILING = {
    "nc_1024_128": (128, 128),  # 最优分块
    "nc_1024_256": (64,  256),  # 最优分块 (原64,128为次优)
    "nc_2048_128": (128, 128),  # 最优分块
    "nc_2048_256": (64,  256),
    "nc_4096_128": (128, 128),
    "nc_8192_64":  (128, 256),  # (64,512)略慢, (128,256)最优
}

BASELINE_C_TILING = {
    "c_1024_128": (128, 64),
    "c_1024_256": (64,  64),
    "c_2048_128": (128, 64),
    "c_2048_256": (64,  64),
    "c_4096_128": (128, 64),
    "c_8192_64":  (128, 64),
}

NUM_WARMUP = 5
NUM_ITERS = 20


def load_kernel(kernel_path):
    """动态加载 kernel 模块"""
    clean_name = os.path.splitext(os.path.basename(kernel_path))[0]
    clean_name = clean_name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    spec = importlib.util.spec_from_file_location(clean_name, kernel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[clean_name] = mod
    spec.loader.exec_module(mod)
    return mod


def benchmark_kernel(kernel_path, cases, tiling_map, kernel_label, warmup=NUM_WARMUP, iters=NUM_ITERS):
    """对指定 kernel 运行所有测试用例并返回计时结果。
    cases 格式: [(Z,H,N_CTX,HEAD_DIM,causal,dtype_str,BM,BN,label), ...] 或
                [(Z,H,N_CTX,HEAD_DIM,causal,dtype_str,label), ...]
    当 cases 为7元组时，BM/BN从 tiling_map[label] 获取。
    """
    print(f"\n{'='*80}")
    print(f"  Benchmarking: {kernel_label}")
    print(f"  Kernel: {kernel_path}")
    print(f"{'='*80}")

    mod = load_kernel(kernel_path)
    attention_fn = mod.attention
    results = {}

    for case in cases:
        if len(case) == 9:
            Z, H, N_CTX, HEAD_DIM, causal, dtype_str, BM, BN, label = case
        else:
            Z, H, N_CTX, HEAD_DIM, causal, dtype_str, label = case
            BM, BN = tiling_map[label]
        dtype = torch.float16 if dtype_str == "fp16" else torch.bfloat16
        causal_str = "causal" if causal else "noncausal"

        print(f"\n  [{label}] Z={Z}, H={H}, N_CTX={N_CTX}, D={HEAD_DIM}, "
              f"{causal_str}, BM={BM}, BN={BN}")

        torch.manual_seed(42)
        q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
        k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
        v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
        sm_scale = 0.5

        try:
            # Warmup
            for _ in range(warmup):
                _ = attention_fn(q, k, v, causal, sm_scale, BM, BN)
                torch.npu.synchronize()

            # Timed runs
            torch.npu.synchronize()
            times = []
            for _ in range(iters):
                t0 = time.perf_counter()
                _ = attention_fn(q, k, v, causal, sm_scale, BM, BN)
                torch.npu.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)

            avg_ms = sum(times) / len(times)
            min_ms = min(times)
            max_ms = max(times)
            std_ms = (sum((t - avg_ms) ** 2 for t in times) / len(times)) ** 0.5
            print(f"    avg={avg_ms:.3f} ms, min={min_ms:.3f} ms, max={max_ms:.3f} ms, std={std_ms:.3f} ms")

            results[label] = {
                "avg_ms": round(avg_ms, 3),
                "min_ms": round(min_ms, 3),
                "max_ms": round(max_ms, 3),
                "std_ms": round(std_ms, 3),
                "times_ms": [round(t, 3) for t in times],
                "BM": BM, "BN": BN,
                "error": None,
            }
        except Exception as e:
            print(f"    FAILED: {e}")
            results[label] = {
                "avg_ms": None, "min_ms": None, "max_ms": None, "std_ms": None,
                "BM": BM, "BN": BN,
                "error": str(e)[:300],
            }

    return results


def compute_speedup(baseline_results, variant_results, cases):
    """计算加速比"""
    speedups = {}
    for case in cases:
        label = case[-1]  # label is always the last element
        base = baseline_results.get(label, {})
        var = variant_results.get(label, {})
        base_ms = base.get("avg_ms")
        var_ms = var.get("avg_ms")
        if base_ms is not None and var_ms is not None and var_ms > 0:
            speedups[label] = round(base_ms / var_ms, 3)
        else:
            speedups[label] = None
    return speedups


def generate_report(cases, baseline_results, variant_results_list, variant_labels, title, report_type):
    """生成消融实验报告"""
    lines = []
    sep = "=" * 130

    lines.append(sep)
    lines.append(f"  FlashAttention V2 消融实验报告 - {title}")
    lines.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  硬件环境: Ascend910B3 | 框架: torch_npu + Triton-Ascend")
    lines.append(f"  测试配置: warmup={NUM_WARMUP}, iters={NUM_ITERS}")
    lines.append(sep)
    lines.append("")

    # 实验说明
    lines.append("  实验设置:")
    lines.append("    Baseline: 01_official.py (TritonAscend FlashAttention V2 baseline)")
    for i, vlabel in enumerate(variant_labels):
        lines.append(f"    变体{i+1}: {vlabel}")
    lines.append("")

    # 加速比公式
    lines.append("  加速比定义:")
    short_names = []
    for i, vlabel in enumerate(variant_labels):
        sn = f"V{i+1}"
        short_names.append(sn)
        short_desc = vlabel.split("(")[0].strip()
        lines.append(f"    S{i+1} = T_baseline / T_{sn}   ({short_desc})")
    lines.append("")

    # 表头
    header = (f"{'Shape':<18s} {'Z':>4s} {'H':>2s} {'N_CTX':>6s} {'D':>4s} "
              f"{'Baseline(ms)':>13s}")
    for sn in short_names:
        header += f"  {sn+'(ms)':>14s}"
    for i in range(len(variant_labels)):
        header += f"  {'S'+str(i+1):>8s}"
    lines.append(header)
    lines.append("-" * 130)

    # 数据行
    total_base = 0.0
    total_var = [0.0] * len(variant_labels)
    valid_count = 0

    for case in cases:
        if len(case) == 9:
            Z, H, N_CTX, HEAD_DIM, causal, dtype_str, _, _, label = case
        else:
            Z, H, N_CTX, HEAD_DIM, causal, dtype_str, label = case

        base = baseline_results.get(label, {})
        base_ms = base.get("avg_ms")

        row = f"{label:<18s} {Z:4d} {H:2d} {N_CTX:6d} {HEAD_DIM:4d} "

        if base_ms is not None:
            row += f"{base_ms:13.2f}"
            total_base += base_ms
        else:
            row += f"{'N/A':>13s}"

        all_valid = base_ms is not None
        for vi, var_results in enumerate(variant_results_list):
            var = var_results.get(label, {})
            var_ms = var.get("avg_ms")
            if var_ms is not None:
                row += f"  {var_ms:14.2f}"
                if all_valid:
                    total_var[vi] += var_ms
            else:
                row += f"  {'N/A':>14s}"
                all_valid = False

        for vi, var_results in enumerate(variant_results_list):
            var = var_results.get(label, {})
            var_ms = var.get("avg_ms")
            if base_ms is not None and var_ms is not None and var_ms > 0:
                sp = base_ms / var_ms
                row += f"  {sp:6.2f}x"
            else:
                row += f"  {'N/A':>8s}"

        if all_valid:
            valid_count += 1

        lines.append(row)

    lines.append("-" * 130)
    lines.append("")

    # 汇总
    lines.append(sep)
    lines.append("  汇总统计")
    lines.append(sep)

    if valid_count > 0:
        for vi in range(len(variant_labels)):
            if total_base > 0 and total_var[vi] > 0:
                overall_sp = total_base / total_var[vi]
                lines.append(f"  S{vi+1} ({short_names[vi]}) 总体加速比: {overall_sp:.2f}x")
        lines.append(f"  有效测试数: {valid_count} / {len(cases)}")
        lines.append(f"  Baseline 总耗时: {total_base:.2f} ms")
        for vi in range(len(variant_labels)):
            if total_var[vi] > 0:
                lines.append(f"  {short_names[vi]} 总耗时: {total_var[vi]:.2f} ms")
    else:
        lines.append("  无有效数据。")

    # 错误报告
    errors = []
    for case in cases:
        label = case[-1]
        base = baseline_results.get(label, {})
        if base.get("error"):
            errors.append(f"  [{label}] Baseline ERROR: {base['error'][:200]}")
        for vi, var_results in enumerate(variant_results_list):
            var = var_results.get(label, {})
            if var.get("error"):
                errors.append(f"  [{label}] {variant_labels[vi]} ERROR: {var['error'][:200]}")

    if errors:
        lines.append("")
        lines.append("  错误列表:")
        for e in errors:
            lines.append(e)

    # 逐用例详情
    lines.append("")
    lines.append(sep)
    lines.append("  逐用例详情")
    lines.append(sep)

    for case in cases:
        if len(case) == 9:
            Z, H, N_CTX, HEAD_DIM, causal, dtype_str, _, _, label = case
        else:
            Z, H, N_CTX, HEAD_DIM, causal, dtype_str, label = case
        lines.append(f"\n  --- {label}: Z={Z}, H={H}, N_CTX={N_CTX}, D={HEAD_DIM} ---")

        base = baseline_results.get(label, {})
        if base.get("error") is None and base.get("avg_ms") is not None:
            lines.append(f"    Baseline:       avg={base['avg_ms']:.3f} ms, "
                         f"min={base['min_ms']:.3f} ms, max={base['max_ms']:.3f} ms, "
                         f"BM={base['BM']}, BN={base['BN']}")
        else:
            lines.append(f"    Baseline:       ERROR - {base.get('error', 'N/A')}")

        for vi, (vlabel, var_results) in enumerate(zip(variant_labels, variant_results_list)):
            var = var_results.get(label, {})
            if var.get("error") is None and var.get("avg_ms") is not None:
                lines.append(f"    {vlabel}: avg={var['avg_ms']:.3f} ms, "
                             f"min={var['min_ms']:.3f} ms, max={var['max_ms']:.3f} ms, "
                             f"BM={var['BM']}, BN={var['BN']}")
            else:
                lines.append(f"    {vlabel}: ERROR - {var.get('error', 'N/A')}")

            base_ms = base.get("avg_ms")
            var_ms = var.get("avg_ms")
            if base_ms is not None and var_ms is not None and var_ms > 0:
                sp = base_ms / var_ms
                lines.append(f"      Speedup S{vi+1} ({short_names[vi]}) = {sp:.2f}x")

    lines.append("")
    lines.append(sep)
    lines.append("  报告结束")
    lines.append(sep)

    return "\n".join(lines)


def run_msprof_for_case(kernel_path, case, output_dir):
    """使用 msprof 对单个测试用例进行 profiling"""
    label = case[8]
    Z, H, N_CTX, HEAD_DIM, causal, dtype_str, BM, BN = case[0], case[1], case[2], case[3], case[4], case[5], case[6], case[7]

    script = os.path.abspath(__file__)
    app_cmd = (
        f"{sys.executable} {script} --mode single "
        f"--kernel '{kernel_path}' "
        f"--Z {Z} --H {H} --N_CTX {N_CTX} --HEAD_DIM {HEAD_DIM} "
        f"--causal {1 if causal else 0} --dtype {dtype_str} "
        f"--BM {BM} --BN {BN} --label {label}"
    )

    wrapper_path = os.path.join(output_dir, "_msprof_wrapper.sh")
    os.makedirs(output_dir, exist_ok=True)
    with open(wrapper_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"exec {app_cmd}\n")
    os.chmod(wrapper_path, 0o755)

    cmd = ["msprof", f"--application={wrapper_path}", f"--output={output_dir}"]
    print(f"    msprof: {output_dir}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            env={**os.environ, "ASCEND_GLOBAL_LOG_LEVEL": "3"},
        )
        if result.returncode != 0:
            err_line = result.stderr.split('\n')[0] if result.stderr else "unknown"
            print(f"    WARNING: msprof returned {result.returncode}: {err_line[:200]}")
    except subprocess.TimeoutExpired:
        print(f"    WARNING: msprof timed out for {label}")
    except Exception as e:
        print(f"    WARNING: msprof failed: {e}")
    finally:
        if os.path.exists(wrapper_path):
            os.remove(wrapper_path)


def run_single_case(args):
    """运行单个测试用例（供 msprof subprocess 调用）"""
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    mod = load_kernel(args.kernel)
    attention_fn = mod.attention

    torch.manual_seed(42)
    q = torch.empty((args.Z, args.H, args.N_CTX, args.HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    k = torch.empty((args.Z, args.H, args.N_CTX, args.HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    v = torch.empty((args.Z, args.H, args.N_CTX, args.HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    sm_scale = 0.5
    causal = bool(args.causal)

    print(f"[{args.label}] kernel={os.path.basename(args.kernel)}, "
          f"Z={args.Z}, H={args.H}, N_CTX={args.N_CTX}, D={args.HEAD_DIM}, "
          f"causal={causal}, BM={args.BM}, BN={args.BN}")

    # Warmup
    for i in range(NUM_WARMUP):
        _ = attention_fn(q, k, v, causal, sm_scale, args.BM, args.BN)
        torch.npu.synchronize()

    # Timed runs
    times = []
    for i in range(NUM_ITERS):
        t0 = time.perf_counter()
        _ = attention_fn(q, k, v, causal, sm_scale, args.BM, args.BN)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        dt_ms = (t1 - t0) * 1000.0
        times.append(dt_ms)

    avg_ms = sum(times) / len(times)
    print(f"  Result: avg={avg_ms:.3f} ms, times={[f'{t:.2f}' for t in times]}")
    print("DONE")


def main():
    parser = argparse.ArgumentParser(description="FlashAttention V2 消融实验")
    parser.add_argument("--mode", choices=["timing", "msprof", "single"], default="timing",
                        help="运行模式: timing=纯Python计时, msprof=msprof profiling, single=单用例")
    parser.add_argument("--kernel", type=str, default=None, help="Kernel 文件路径")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")

    # 单用例参数
    parser.add_argument("--Z", type=int, default=None)
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--N_CTX", type=int, default=None)
    parser.add_argument("--HEAD_DIM", type=int, default=None)
    parser.add_argument("--causal", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--BM", type=int, default=None)
    parser.add_argument("--BN", type=int, default=None)
    parser.add_argument("--label", type=str, default="unknown")

    args = parser.parse_args()

    if args.mode == "single":
        run_single_case(args)
        return

    kernels = [
        ("Official", os.path.join(WORKSPACE, "01_official.py")),
        ("V1: + K/V address advance", os.path.join(WORKSPACE, "02_v1_kv_advance.py")),
        ("V2: + dynamic AI cores", os.path.join(WORKSPACE, "03_v2_numcores.py")),
        ("V3: + compiler pipeline", os.path.join(WORKSPACE, "04_v3_pipeline_optimized.py")),
        ("ASCEND-RSFA: + causal split", os.path.join(WORKSPACE, "05_ascend_rsfa.py")),
    ]
    baseline_path = kernels[0][1]
    variants = kernels[1:]

    if args.mode == "timing":
        print("=" * 80)
        print("  FlashAttention V2 消融实验")
        print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # ================================================================
        # 非因果路径消融实验
        # ================================================================
        print("\n" + "=" * 80)
        print("  6.3.1 非因果路径消融实验")
        print("=" * 80)

        # 为baseline构造不带BM/BN的7元组cases（强制使用BASELINE_NC_TILING）
        nc_shapes_only = [(Z,H,N,D,c,d,l) for Z,H,N,D,c,d,_,_,l in NON_CAUSAL_CASES]

        print(f"\n  [1/{len(kernels)}] Benchmarking Official (非因果)...")
        baseline_nc_results = benchmark_kernel(
            baseline_path, nc_shapes_only, BASELINE_NC_TILING,
            "Official (non-causal)"
        )

        nc_variant_results = []
        for index, (label, kernel_path) in enumerate(variants, start=2):
            print(f"\n  [{index}/{len(kernels)}] Benchmarking {label} (非因果)...")
            nc_variant_results.append(
                benchmark_kernel(kernel_path, NON_CAUSAL_CASES, {}, label)
            )

        nc_report = generate_report(
            NON_CAUSAL_CASES,
            baseline_nc_results,
            nc_variant_results,
            [label for label, _ in variants],
            "6.3.1 非因果路径",
            "non_causal"
        )

        nc_path = os.path.join(WORKSPACE, "ablation_report_non_causal.txt")
        with open(nc_path, "w") as f:
            f.write(nc_report)
        print(f"\n非因果消融报告已保存: {nc_path}")

        # ================================================================
        # 因果路径消融实验
        # ================================================================
        print("\n" + "=" * 80)
        print("  6.3.2 因果路径消融实验")
        print("=" * 80)

        print(f"\n  [1/{len(kernels)}] Benchmarking Official (因果)...")
        baseline_c_results = benchmark_kernel(
            baseline_path, CAUSAL_SHAPES, BASELINE_C_TILING,
            "Official (causal)"
        )

        c_variant_results = []
        for index, (label, kernel_path) in enumerate(variants, start=2):
            print(f"\n  [{index}/{len(kernels)}] Benchmarking {label} (因果)...")
            tiling = FUSED_SPLIT_CAUSAL_TILING if index == len(kernels) else FUSED_CAUSAL_TILING
            c_variant_results.append(
                benchmark_kernel(kernel_path, CAUSAL_SHAPES, tiling, label)
            )

        causal_report = generate_report(
            CAUSAL_SHAPES,
            baseline_c_results,
            c_variant_results,
            [label for label, _ in variants],
            "6.3.2 因果路径",
            "causal"
        )

        causal_path = os.path.join(WORKSPACE, "ablation_report_causal.txt")
        with open(causal_path, "w") as f:
            f.write(causal_report)
        print(f"\n因果消融报告已保存: {causal_path}")

        # 打印汇总
        print("\n" + "=" * 80)
        print("  消融实验完成!")
        print(f"  非因果报告: {nc_path}")
        print(f"  因果报告:   {causal_path}")
        print("=" * 80)

        # 保存原始 JSON 数据
        raw_data = {
            "non_causal": {
                "baseline": baseline_nc_results,
                **{label: results for (label, _), results in zip(variants, nc_variant_results)},
            },
            "causal": {
                "baseline": baseline_c_results,
                **{label: results for (label, _), results in zip(variants, c_variant_results)},
            },
            "timestamp": datetime.now().isoformat(),
        }
        raw_path = os.path.join(WORKSPACE, "ablation_raw_data.json")
        with open(raw_path, "w") as f:
            json.dump(raw_data, f, indent=2, default=str)
        print(f"  原始数据: {raw_path}")

    elif args.mode == "msprof":
        # msprof 模式: 对所有组合进行 msprof profiling
        output_base = args.output_dir or os.path.join(WORKSPACE, "ablation_msprof")
        os.makedirs(output_base, exist_ok=True)

        # 非因果
        nc_configs = [("01_official", baseline_path, BASELINE_NC_TILING)]
        nc_configs.extend(
            (os.path.splitext(os.path.basename(path))[0], path,
             {c[8]: (c[6], c[7]) for c in NON_CAUSAL_CASES})
            for _, path in variants
        )
        for config_name, kernel_path, tiling in nc_configs:
            for case in NON_CAUSAL_CASES:
                label = case[8]
                out_dir = os.path.join(output_base, "non_causal", config_name, label)
                os.makedirs(out_dir, exist_ok=True)
                Z, H, N_CTX, HEAD_DIM, causal, dtype_str, _, _, _ = case
                BM, BN = tiling[label]
                print(f"msprof [{config_name}/{label}] ...")
                run_msprof_for_case(
                    kernel_path,
                    (Z, H, N_CTX, HEAD_DIM, causal, dtype_str, BM, BN, label),
                    out_dir,
                )

        # 因果
        c_configs = [("01_official", baseline_path, BASELINE_C_TILING)]
        for index, (_, path) in enumerate(variants, start=2):
            tiling = FUSED_SPLIT_CAUSAL_TILING if index == len(kernels) else FUSED_CAUSAL_TILING
            c_configs.append((os.path.splitext(os.path.basename(path))[0], path, tiling))
        for config_name, kernel_path, tiling in c_configs:
            for case in CAUSAL_SHAPES:
                Z, H, N_CTX, HEAD_DIM, causal, dtype_str, label = case
                out_dir = os.path.join(output_base, "causal", config_name, label)
                os.makedirs(out_dir, exist_ok=True)
                BM, BN = tiling[label]
                print(f"msprof [{config_name}/{label}] ...")
                run_msprof_for_case(
                    kernel_path,
                    (Z, H, N_CTX, HEAD_DIM, causal, dtype_str, BM, BN, label),
                    out_dir,
                )

        print(f"\nmsprof profiling 完成, 结果保存在: {output_base}")


if __name__ == "__main__":
    main()
