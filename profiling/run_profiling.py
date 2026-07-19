#!/usr/bin/env python3
"""
Profiling harness for flash attention kernels.
Runs under msprof and outputs profiling data.
Usage (standalone or under msprof):
    python3 run_profiling.py --kernel <path> --case-idx <N> [--warmup 3] [--iters 10]
    msprof --application="python3 run_profiling.py --kernel <path> --case-idx <N>" --aic-metrics=... --output=<dir>
"""

import argparse
import importlib.util
import os
import sys
import time

import torch
import torch_npu

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

DEVICE = "npu"

# Test cases definition (same as benchmark)
TEST_CASES = {
    3:  (128, 8, 2048, 128, True,  torch.float16, 128, 64),
    9:  (128, 8, 2048, 128, False, torch.float16, 128, 256),
}

BASELINE_TILINGS = {
    3:  (128, 64),
    9:  (128, 128),
}


def load_kernel(kernel_path):
    clean_name = os.path.splitext(os.path.basename(kernel_path))[0]
    clean_name = clean_name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    spec = importlib.util.spec_from_file_location(clean_name, kernel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[clean_name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description="Flash attention profiling runner")
    parser.add_argument("--kernel", required=True, help="Path to kernel .py file")
    parser.add_argument("--case-idx", type=int, required=True, choices=[3, 9],
                        help="Test case index (3=causal, 9=noncausal)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    # Get kernel type
    is_baseline = "06" in args.kernel

    Z, H, N_CTX, HEAD_DIM, causal, dtype, _, _ = TEST_CASES[args.case_idx]

    if is_baseline:
        BM, BN = BASELINE_TILINGS[args.case_idx]
    else:
        from run_final_benchmark import OPTIMAL_TILINGS as OPT
        BM, BN = OPT[args.case_idx]

    kernel_type = "baseline" if is_baseline else "optimized"
    causal_str = "causal" if causal else "noncausal"
    print(f"Profiling: {kernel_type} kernel, Case {args.case_idx} ({causal_str}), "
          f"Z={Z}, H={H}, N_CTX={N_CTX}, D={HEAD_DIM}, BM={BM}, BN={BN}")

    # Load kernel
    mod = load_kernel(args.kernel)
    attention_fn = mod.attention

    # Create tensors
    torch.manual_seed(42)
    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    sm_scale = 0.5

    # Check if running under msprof (MSProf environment variable or CtrlMode)
    running_under_msprof = "MSPROF" in os.environ.get("_", "") or os.path.exists("/usr/local/Ascend/cann-8.5.0/bin/msprof")

    # Warmup
    print(f"Warmup: {args.warmup} iterations...")
    for i in range(args.warmup):
        _ = attention_fn(q, k, v, causal, sm_scale, BM, BN)
        torch.npu.synchronize()
        print(f"  Warmup {i+1}/{args.warmup} done")

    # Timed iterations
    print(f"Running: {args.iters} iterations...")
    torch.npu.synchronize()
    times = []
    for i in range(args.iters):
        t0 = time.perf_counter()
        _ = attention_fn(q, k, v, causal, sm_scale, BM, BN)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        dt_ms = (t1 - t0) * 1000.0
        times.append(dt_ms)
        print(f"  Iter {i+1}/{args.iters}: {dt_ms:.4f} ms")

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    print(f"\nResults: avg={avg_ms:.4f} ms, min={min_ms:.4f} ms, max={max_ms:.4f} ms")
    print("DONE")


if __name__ == "__main__":
    main()
