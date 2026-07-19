#!/usr/bin/env python3
"""
Single test case runner for flash attention benchmark.
Usage:
    python3 run_single_test.py --kernel <path> --Z <Z> --H <H> --N_CTX <N> --HEAD_DIM <D> --causal <0|1> --dtype <fp16|bf16> --BM <BM> --BN <BN> [--warmup <N>] [--iters <N>]
Output: JSON with timing info
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import torch
import torch_npu


DEVICE = "npu"


def load_kernel_module(kernel_path):
    """Load a kernel file as a Python module."""
    module_name = os.path.splitext(os.path.basename(kernel_path))[0].replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, kernel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_benchmark(kernel_path, Z, H, N_CTX, HEAD_DIM, causal, dtype_str, BM, BN,
                  warmup=5, iters=20):
    """Run benchmark for a single test case."""
    dtype = torch.float16 if dtype_str == "fp16" else torch.bfloat16

    # Load the kernel module
    mod = load_kernel_module(kernel_path)
    attention_fn = mod.attention

    # Create tensors
    torch.manual_seed(42)
    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    sm_scale = 0.5

    # Warmup
    for _ in range(warmup):
        _ = attention_fn(q, k, v, causal, sm_scale, BM, BN)
        torch.npu.synchronize()

    # Benchmark
    torch.npu.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = attention_fn(q, k, v, causal, sm_scale, BM, BN)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)

    return {
        "avg_ms": round(avg_ms, 4),
        "min_ms": round(min_ms, 4),
        "max_ms": round(max_ms, 4),
        "iters": iters,
        "warmup": warmup,
        "params": {
            "Z": Z, "H": H, "N_CTX": N_CTX, "HEAD_DIM": HEAD_DIM,
            "causal": causal, "dtype": dtype_str, "BM": BM, "BN": BN
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Single flash attention benchmark")
    parser.add_argument("--kernel", required=True, help="Path to kernel .py file")
    parser.add_argument("--Z", type=int, required=True)
    parser.add_argument("--H", type=int, required=True)
    parser.add_argument("--N_CTX", type=int, required=True)
    parser.add_argument("--HEAD_DIM", type=int, required=True)
    parser.add_argument("--causal", type=int, required=True)
    parser.add_argument("--dtype", required=True, choices=["fp16", "bf16"])
    parser.add_argument("--BM", type=int, required=True)
    parser.add_argument("--BN", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    try:
        result = run_benchmark(
            kernel_path=args.kernel,
            Z=args.Z, H=args.H, N_CTX=args.N_CTX, HEAD_DIM=args.HEAD_DIM,
            causal=bool(args.causal), dtype_str=args.dtype,
            BM=args.BM, BN=args.BN,
            warmup=args.warmup, iters=args.iters
        )
        print(json.dumps(result))
    except Exception as e:
        error_result = {
            "error": str(e),
            "params": {
                "Z": args.Z, "H": args.H, "N_CTX": args.N_CTX, "HEAD_DIM": args.HEAD_DIM,
                "causal": bool(args.causal), "dtype": args.dtype, "BM": args.BM, "BN": args.BN
            }
        }
        print(json.dumps(error_result))
        sys.exit(1)


if __name__ == "__main__":
    main()
