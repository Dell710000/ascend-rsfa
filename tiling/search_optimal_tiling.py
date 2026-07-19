#!/usr/bin/env python3
"""
Optimal tiling search for flash attention kernels.
Tests all candidate (BM, BN) combinations for 12 test cases within one Python process.

Usage:
    python3 search_optimal_tiling.py --kernel <path.py> [--kernel-name <name>]
Output: JSON with optimal BM, BN for each test case
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import math

import torch
import torch_npu

# Force unbuffered output for progress visibility in subprocess
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

DEVICE = "npu"


# ---- 12 Test Cases ----
TEST_CASES = [
    # idx, Z, H, N_CTX, HEAD_DIM, causal, dtype
    (1,  128, 8, 1024, 128, True,  torch.float16),
    (2,  128, 8, 1024, 256, True,  torch.float16),
    (3,  128, 8, 2048, 128, True,  torch.float16),
    (4,  128, 8, 2048, 256, True,  torch.float16),
    (5,  128, 8, 4096, 128, True,  torch.float16),
    (6,  128, 8, 8192, 64,  True,  torch.float16),
    (7,  128, 8, 1024, 128, False, torch.float16),
    (8,  128, 8, 1024, 256, False, torch.float16),
    (9,  128, 8, 2048, 128, False, torch.float16),
    (10, 128, 8, 2048, 256, False, torch.float16),
    (11, 128, 8, 4096, 128, False, torch.float16),
    (12, 128, 8, 8192, 64,  False, torch.float16),
]


def generate_candidates(N_CTX, HEAD_DIM, causal, is_06_kernel):
    """
    Generate candidate (BM, BN) pairs.
    06 kernel requires BM to divide N_CTX exactly.
    Optimized kernel uses cdiv so any BM works.
    """
    if is_06_kernel:
        BM_candidates = [b for b in [32, 64, 128, 256] if N_CTX % b == 0]
    else:
        BM_candidates = [b for b in [32, 64, 128, 256] if b <= N_CTX]

    BN_candidates = [b for b in [32, 64, 128, 256] if b <= N_CTX]

    # For large N_CTX, allow BN=512
    if N_CTX >= 4096:
        BN_candidates.append(512)

    # HEAD_DIM constraints: large D -> smaller BM to avoid register pressure
    if HEAD_DIM >= 256:
        BM_candidates = [b for b in BM_candidates if b <= 128]

    # Generate all pairs, prioritizing promising ones first
    candidates = []
    for bm in BM_candidates:
        for bn in BN_candidates:
            # Skip obviously bad combos
            if bm * bn > 65536:  # Too large
                continue
            candidates.append((bm, bn))

    # Sort: prioritize the current default tilings
    # Put smaller combos first (faster to test)
    candidates.sort(key=lambda x: (x[0] * x[1], x[0], x[1]))

    return candidates


def load_kernel(kernel_path):
    """Load a kernel file as a Python module."""
    clean_name = os.path.splitext(os.path.basename(kernel_path))[0]
    clean_name = clean_name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    spec = importlib.util.spec_from_file_location(clean_name, kernel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[clean_name] = mod
    spec.loader.exec_module(mod)
    return mod


def time_kernel(attention_fn, q, k, v, causal, sm_scale, BM, BN, warmup=2, iters=3):
    """Time a single (BM, BN) combination. Returns avg_ms or None on error."""
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
        return sum(times) / len(times)
    except Exception as e:
        return None


def search_case(attention_fn, case, is_06_kernel, warmup=2, iters=3):
    """Search optimal (BM, BN) for a single test case."""
    idx, Z, H, N_CTX, HEAD_DIM, causal, dtype = case
    candidates = generate_candidates(N_CTX, HEAD_DIM, causal, is_06_kernel)

    # Create tensors once for this case
    torch.manual_seed(42)
    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    sm_scale = 0.5

    results = []
    for bm, bn in candidates:
        avg_ms = time_kernel(attention_fn, q, k, v, causal, sm_scale, bm, bn, warmup, iters)
        if avg_ms is not None:
            results.append({"BM": bm, "BN": bn, "avg_ms": round(avg_ms, 4)})
            print(f"    BM={bm:3d}, BN={bn:3d} -> {avg_ms:10.4f} ms")
        else:
            print(f"    BM={bm:3d}, BN={bn:3d} -> FAILED (overflow or compile error)")

    if not results:
        return None

    # Sort by time
    results.sort(key=lambda r: r["avg_ms"])
    return results


def main():
    parser = argparse.ArgumentParser(description="Search optimal tiling for flash attention")
    parser.add_argument("--kernel", required=True, help="Path to kernel .py file")
    parser.add_argument("--kernel-name", default=None, help="Kernel name for output")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()

    kernel_name = args.kernel_name or os.path.basename(args.kernel)
    is_06 = "06" in kernel_name

    print(f"{'='*80}")
    print(f"  Optimal Tiling Search: {kernel_name}")
    print(f"  Kernel type: {'06 baseline (BM must divide N_CTX)' if is_06 else 'Optimized (cdiv, any BM)'}")
    print(f"  Test cases: {len(TEST_CASES)}")
    print(f"  Warmup: {args.warmup}, Timed iters: {args.iters}")
    print(f"{'='*80}")
    print()

    # Load kernel once
    mod = load_kernel(args.kernel)
    attention_fn = mod.attention

    all_results = {}

    for case in TEST_CASES:
        idx = case[0]
        _, Z, H, N_CTX, HEAD_DIM, causal, dtype = case
        causal_str = "causal" if causal else "noncausal"
        print(f"\n[Case {idx:2d}] Z={Z}, H={H}, N_CTX={N_CTX}, D={HEAD_DIM}, {causal_str}")
        print(f"{'~'*60}")

        results = search_case(attention_fn, case, is_06, args.warmup, args.iters)

        if results and len(results) > 0:
            best = results[0]
            print(f"  => BEST: BM={best['BM']}, BN={best['BN']}, time={best['avg_ms']} ms")
            # Show top 3
            if len(results) >= 2:
                print(f"     Top 3: " + ", ".join(
                    f"({r['BM']},{r['BN']})={r['avg_ms']}ms" for r in results[:3]
                ))
            all_results[f"case_{idx:02d}"] = {
                "params": {"Z": Z, "H": H, "N_CTX": N_CTX, "HEAD_DIM": HEAD_DIM,
                           "causal": causal, "dtype": str(dtype)},
                "best": best,
                "top3": results[:3],
                "all_tested": results,
                "failed_count": len(generate_candidates(N_CTX, HEAD_DIM, causal, is_06)) - len(results)
            }
        else:
            print(f"  => ALL FAILED for this case!")
            all_results[f"case_{idx:02d}"] = {
                "params": {"Z": Z, "H": H, "N_CTX": N_CTX, "HEAD_DIM": HEAD_DIM,
                           "causal": causal, "dtype": str(dtype)},
                "best": None,
                "error": "All candidate tilings failed"
            }

    # Output summary
    print(f"\n{'='*80}")
    print(f"  Search Complete - Summary")
    print(f"{'='*80}")
    print(f"  {'Case':<6s} {'Best BM':>8s} {'Best BN':>8s} {'Time (ms)':>12s} {'Tested':>8s} {'Failed':>8s}")
    print(f"  {'-'*50}")
    for case in TEST_CASES:
        idx = case[0]
        key = f"case_{idx:02d}"
        r = all_results.get(key, {})
        if r.get("best"):
            b = r["best"]
            n_tested = len(r.get("all_tested", []))
            n_failed = r.get("failed_count", 0)
            print(f"  Case{idx:02d}  {b['BM']:8d} {b['BN']:8d} {b['avg_ms']:12.4f} {n_tested:8d} {n_failed:8d}")
        else:
            print(f"  Case{idx:02d}  {'N/A':>8s} {'N/A':>8s} {'N/A':>12s}")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    else:
        # Print JSON to stdout for capture
        print("\n__JSON_START__")
        print(json.dumps(all_results, indent=2))
        print("__JSON_END__")


if __name__ == "__main__":
    main()
