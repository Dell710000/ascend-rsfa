#!/usr/bin/env python
"""
Benchmark script for Triton-Ascend FlashAttention kernels with msprof profiling.

Measures Triton kernel execution time vs baseline (npu_fusion_attention)
across 12 test cases, verifies correctness, and collects msprof profiling data.

Usage:
    # Benchmark the baseline
    python benchmark_06_msprof.py --kernel ../baseline/06-fused-attention.py

    # Benchmark ASCEND-RSFA
    python benchmark_06_msprof.py --kernel ../ascend_rsfa/flash_attention_forward.py --out result_opt

    # Profile a single case with msprof (used internally)
    msprof --application="python benchmark_06_msprof.py --kernel kernel.py --single 1" --output=./result_opt/case_01/
"""

import os
import sys
import json
import csv
import time
import argparse
import subprocess
import importlib.util
import importlib.machinery

import torch
import torch_npu

DEVICE = "npu"
NUM_TIMING_RUNS = 5

DEFAULT_KERNEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "baseline/06-fused-attention.py")
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")

# ============================================================================
# 12 test cases used by the ASCEND-RSFA paper benchmarks.
# Format: (case_id, Z, H, N_CTX, HEAD_DIM, causal, dtype, label)
# NOTE: BM, BN are resolved at runtime via kernel's get_tiling() if available,
#       otherwise fallback to FALLBACK_TILING below.
# ============================================================================
TEST_CASES = [
    (1,  128, 8, 1024, 128, True,  torch.float16, "causal_1024_128"),
    (2,  128, 8, 1024, 256, True,  torch.float16, "causal_1024_256"),
    (3,  128, 8, 2048, 128, True,  torch.float16, "causal_2048_128"),
    (4,  128, 8, 2048, 256, True,  torch.float16, "causal_2048_256"),
    (5,  128, 8, 4096, 128, True,  torch.float16, "causal_4096_128"),
    (6,  128, 8, 8192, 64,  True,  torch.float16, "causal_8192_64"),
    (7,  128, 8, 1024, 128, False, torch.float16, "noncausal_1024_128"),
    (8,  128, 8, 1024, 256, False, torch.float16, "noncausal_1024_256"),
    (9,  128, 8, 2048, 128, False, torch.float16, "noncausal_2048_128"),
    (10, 128, 8, 2048, 256, False, torch.float16, "noncausal_2048_256"),
    (11, 128, 8, 4096, 128, False, torch.float16, "noncausal_4096_128"),
    (12, 128, 8, 8192, 64,  False, torch.float16, "noncausal_8192_64"),
]

# Fallback BM/BN for kernels that don't have get_tiling().
# These are tuned for 06-fused-attention.py (cases 11/12 adjusted for UB limit).
FALLBACK_TILING = {
    (128, 8, 1024, 128, True):  (128, 64),
    (128, 8, 1024, 256, True):  (64,  64),
    (128, 8, 2048, 128, True):  (128, 64),
    (128, 8, 2048, 256, True):  (64,  64),
    (128, 8, 4096, 128, True):  (128, 64),
    (128, 8, 8192, 64,  True):  (128, 64),
    (128, 8, 1024, 128, False): (64,  128),
    (128, 8, 1024, 256, False): (64,  128),
    (128, 8, 2048, 128, False): (64,  128),
    (128, 8, 2048, 256, False): (64,  256),
    (128, 8, 4096, 128, False): (128, 128),   # UB-safe for 06 baseline (original (128,256) overflows)
    (128, 8, 8192, 64,  False): (128, 256),    # UB-safe for 06 baseline (original (128,512) overflows)
}


def load_kernel(kernel_path):
    """Dynamically import the attention kernel from the given file."""
    if not os.path.isfile(kernel_path):
        raise FileNotFoundError(f"Kernel file not found: {kernel_path}")

    spec = importlib.util.spec_from_file_location("fused_attention_kernel", kernel_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {kernel_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_tiling(mod, case_tuple, force_fallback=False):
    """
    Resolve (BM, BN) for a test case.
    Prefers the kernel module's get_tiling() if available, else fallback.
    case_tuple: (Z, H, N_CTX, HEAD_DIM, causal)
    force_fallback: if True, skip get_tiling() and use fallback table.
    Returns (BM, BN, note)
    """
    if not force_fallback:
        get_tiling = getattr(mod, "get_tiling", None)
        if get_tiling is not None:
            try:
                BM, BN = get_tiling(*case_tuple)
                fallback = FALLBACK_TILING.get(case_tuple)
                if fallback and (BM, BN) != fallback:
                    note = f"kernel get_tiling: ({BM},{BN})"
                else:
                    note = ""
                return BM, BN, note
            except (KeyError, Exception):
                pass

    # Fallback
    if case_tuple in FALLBACK_TILING:
        BM, BN = FALLBACK_TILING[case_tuple]
        note = "fallback tiling (overridden)"
        return BM, BN, note

    raise ValueError(f"No tiling found for {case_tuple}")


def generate_inputs(Z, H, N_CTX, HEAD_DIM, dtype):
    """Create random Q/K/V tensors on NPU with a fixed seed."""
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(0.0, 0.5).requires_grad_()
    k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(0.0, 0.5).requires_grad_()
    v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(0.0, 0.5).requires_grad_()
    return q, k, v


def run_baseline(q, k, v, H, sm_scale, causal, N_CTX):
    """Run torch_npu.npu_fusion_attention and return the output.

    For causal, uses compressed_len=2048 atten_mask (matching the
    ASCEND-RSFA-compatible calling convention).
    """
    compressed_len = 2048
    if causal:
        atten_golden_mask = torch.triu(torch.ones(compressed_len, compressed_len, device=DEVICE), diagonal=1).bool()
        sparse_mode = 2
    else:
        atten_golden_mask = None
        sparse_mode = 0

    ref_out = torch_npu.npu_fusion_attention(
        q, k, v, H,
        padding_mask=None,
        atten_mask=atten_golden_mask,
        scale=sm_scale,
        keep_prob=1.0,
        input_layout="BNSD",
        pre_tockens=65535,
        next_tockens=65535,
        sparse_mode=sparse_mode,
    )[0]
    return ref_out


def verify_correctness(tri_out, ref_out, atol=1e-2, rtol=1e-2):
    """Check that the Triton output matches the baseline."""
    return torch.allclose(ref_out, tri_out, atol=atol, rtol=rtol, equal_nan=True)


# ============================================================================
# Workload for msprof profiling  (--single mode)
# ============================================================================
def run_single_case(case_id, attention_fn, kernel_path, force_fallback=False):
    """
    Execute one test case as a profiling workload.
    Called by msprof as:  msprof --application="python benchmark_06_msprof.py --single N"
    """
    case = TEST_CASES[case_id - 1]
    cid, Z, H, N_CTX, HEAD_DIM, causal, dtype, label = case
    sm_scale = 0.5

    # Load kernel to get tiling (same as main benchmark)
    mod = load_kernel(kernel_path)
    BM, BN, note = resolve_tiling(mod, (Z, H, N_CTX, HEAD_DIM, causal), force_fallback=force_fallback)

    print(f"[Case {case_id}/{len(TEST_CASES)}] {label}  "
          f"Z={Z} H={H} N_CTX={N_CTX} HEAD_DIM={HEAD_DIM} "
          f"causal={causal} BM={BM} BN={BN}  [{note}]")

    q, k, v = generate_inputs(Z, H, N_CTX, HEAD_DIM, dtype)

    # Warmup (compiles JIT kernel)
    _ = attention_fn(q, k, v, causal, sm_scale, BM, BN)
    torch.npu.synchronize()

    # Timed runs
    times = []
    for i in range(NUM_TIMING_RUNS):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        out = attention_fn(q, k, v, causal, sm_scale, BM, BN)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    avg_ms = sum(times) / len(times)
    print(f"  Triton: {avg_ms:.3f} ms (avg of {NUM_TIMING_RUNS} runs)")

    # Reference run + correctness check
    ref = run_baseline(q, k, v, H, sm_scale, causal, N_CTX)
    passed = verify_correctness(out.to(dtype), ref)
    print(f"  Correctness: {'PASS' if passed else 'FAIL'}")


# ============================================================================
# Full benchmark   (--all mode)
# ============================================================================
def run_all_cases(attention_fn, kernel_name, force_fallback=False):
    """
    Run all 12 test cases:
      1. Triton timing (pure Python, no msprof overhead)
      2. Baseline timing (npu_fusion_attention)
      3. Correctness verification
      4. Save timing.json per case
      5. Launch msprof subprocess for profiling data
      6. Aggregate result_summary.csv
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_rows = []

    for entry in TEST_CASES:
        case_id, Z, H, N_CTX, HEAD_DIM, causal, dtype, label = entry
        sm_scale = 0.5

        # Resolve BM, BN for this case
        BM, BN, note = resolve_tiling(kernel_mod, (Z, H, N_CTX, HEAD_DIM, causal), force_fallback=force_fallback)

        header = f"  Case {case_id:2d}/{len(TEST_CASES)}  {label}  "
        print(f"\n{'=' * 80}")
        print(f"{header:^80}")
        print(f"  Z={Z} H={H} N_CTX={N_CTX} HEAD_DIM={HEAD_DIM}  "
              f"causal={causal}  dtype={dtype}  BM={BM}  BN={BN}   [{note}]")
        print(f"{'=' * 80}")

        case_dir = os.path.join(RESULTS_DIR, f"case_{case_id:02d}_{label}")
        os.makedirs(case_dir, exist_ok=True)

        # ---- Step 1: Generate inputs ----
        q, k, v = generate_inputs(Z, H, N_CTX, HEAD_DIM, dtype)

        # ---- Step 2: Warmup ----
        print("  [1/5] Warming up Triton kernel ...", end=" ")
        sys.stdout.flush()
        _ = attention_fn(q, k, v, causal, sm_scale, BM, BN)
        torch.npu.synchronize()
        print("done")

        # ---- Step 3: Triton timing ----
        print("  [2/5] Timing Triton kernel ...", end=" ")
        sys.stdout.flush()
        triton_times = []
        for _ in range(NUM_TIMING_RUNS):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            tri_out = attention_fn(q, k, v, causal, sm_scale, BM, BN)
            torch.npu.synchronize()
            t1 = time.perf_counter()
            triton_times.append((t1 - t0) * 1000)
        triton_ms = sum(triton_times) / NUM_TIMING_RUNS
        triton_std = (sum((t - triton_ms) ** 2 for t in triton_times) / NUM_TIMING_RUNS) ** 0.5
        print(f" {triton_ms:.3f} +/- {triton_std:.3f} ms")

        # ---- Step 4: Baseline timing ----
        print("  [3/5] Timing baseline (npu_fusion_attention) ...", end=" ")
        sys.stdout.flush()
        baseline_times = []
        for _ in range(NUM_TIMING_RUNS):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            ref_out = run_baseline(q, k, v, H, sm_scale, causal, N_CTX)
            torch.npu.synchronize()
            t1 = time.perf_counter()
            baseline_times.append((t1 - t0) * 1000)
        baseline_ms = sum(baseline_times) / NUM_TIMING_RUNS
        baseline_std = (sum((t - baseline_ms) ** 2 for t in baseline_times) / NUM_TIMING_RUNS) ** 0.5
        speedup = baseline_ms / triton_ms if triton_ms > 0 else 0.0
        print(f" {baseline_ms:.3f} +/- {baseline_std:.3f} ms  (speedup: {speedup:.2f}x)")

        # ---- Step 5: Correctness ----
        passed = verify_correctness(tri_out.to(dtype), ref_out)
        print(f"  [4/5] Correctness: {'PASS' if passed else 'FAIL'}")

        # ---- Save timing.json ----
        timing = {
            "case_id": case_id,
            "label": label,
            "Z": Z, "H": H, "N_CTX": N_CTX, "HEAD_DIM": HEAD_DIM,
            "causal": causal, "dtype": str(dtype),
            "BM": BM, "BN": BN,
            "tiling_note": note,
            "kernel": kernel_name,
            "triton_ms": round(triton_ms, 3),
            "triton_std": round(triton_std, 3),
            "triton_times_ms": [round(t, 4) for t in triton_times],
            "baseline_ms": round(baseline_ms, 3),
            "baseline_std": round(baseline_std, 3),
            "baseline_times_ms": [round(t, 4) for t in baseline_times],
            "speedup": round(speedup, 3),
            "correctness_pass": bool(passed),
        }
        timing_path = os.path.join(case_dir, "timing.json")
        with open(timing_path, "w") as f:
            json.dump(timing, f, indent=2)
        print(f"  [5/5] Timing saved to {timing_path}")

        # ---- msprof profiling ----
        print("  [msprof] Launching msprof profiling subprocess ...")
        sys.stdout.flush()
        msprof_output_dir = os.path.join(case_dir, "msprof_output")
        _launch_msprof(case_id, msprof_output_dir)

        # Collect row for summary CSV
        summary_rows.append({
            "case_id": case_id,
            "label": label,
            "Z": Z, "H": H, "N_CTX": N_CTX, "HEAD_DIM": HEAD_DIM,
            "causal": causal, "dtype": str(dtype),
            "BM": BM, "BN": BN,
            "tiling_note": note,
            "triton_ms": f"{triton_ms:.3f}",
            "triton_std": f"{triton_std:.3f}",
            "baseline_ms": f"{baseline_ms:.3f}",
            "baseline_std": f"{baseline_std:.3f}",
            "speedup": f"{speedup:.2f}",
            "correctness": "PASS" if passed else "FAIL",
        })

    # ---- Write summary CSV ----
    csv_path = os.path.join(RESULTS_DIR, "result_summary.csv")
    fieldnames = [
        "case_id", "label", "Z", "H", "N_CTX", "HEAD_DIM",
        "causal", "dtype", "BM", "BN", "tiling_note",
        "triton_ms", "triton_std", "baseline_ms", "baseline_std",
        "speedup", "correctness",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # ---- Print summary table ----
    col_w = [5, 24, 11, 11, 8, 7]
    sep_line = "  ".join("=" * w for w in col_w)
    hdr = (
        f"{'Case':>{col_w[0]}}  {'Label':<{col_w[1]}}  "
        f"{'Triton(ms)':>{col_w[2]}}  {'Base(ms)':>{col_w[3]}}  "
        f"{'Speedup':>{col_w[4]}}  {'OK':>{col_w[5]}}"
    )
    print(f"\n{sep_line}")
    print(hdr)
    print(sep_line)
    for r in summary_rows:
        print(
            f"{r['case_id']:>{col_w[0]}}  {r['label']:<{col_w[1]}}  "
            f"{r['triton_ms']:>{col_w[2]}}  {r['baseline_ms']:>{col_w[3]}}  "
            f"{r['speedup']:>{col_w[4]}}  {r['correctness']:>{col_w[5]}}"
        )
    print(sep_line)
    print(f"\nSummary CSV:  {csv_path}")
    print(f"Results root: {RESULTS_DIR}")


def _check_msprof_available():
    """Check if msprof command is available and functional."""
    try:
        result = subprocess.run(["msprof", "--help"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return False
        profiler_bin = subprocess.run(
            ["which", "msprof"], capture_output=True, text=True, timeout=5
        )
        if profiler_bin.returncode != 0:
            return False
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _launch_msprof(case_id, output_dir):
    """Spawn msprof subprocess for one test case."""
    if not _check_msprof_available():
        print("    SKIP: msprof profiling not available in this environment.\n"
              "    (profiling service may not be running)")
        return

    script = os.path.abspath(__file__)
    app_cmd = f"{sys.executable} {script} --kernel '{KERNEL_FILE}' --single {case_id}"

    # Write a wrapper script because msprof parses --application by splitting on spaces
    wrapper_path = os.path.join(output_dir, "_msprof_wrapper.sh")
    os.makedirs(output_dir, exist_ok=True)
    with open(wrapper_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"exec {app_cmd}\n")
    os.chmod(wrapper_path, 0o755)

    cmd = ["msprof", f"--application={wrapper_path}", f"--output={output_dir}"]

    print(f"    Running: msprof --application={wrapper_path} --output={output_dir}")
    sys.stdout.flush()

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode == 0:
            print(f"    msprof completed. Output in: {output_dir}")
        else:
            err_line = result.stderr.split('\n')[0] if result.stderr else "unknown error"
            print(f"    WARNING: msprof failed ({err_line}).")
            print(f"    (profiling service may not be running in this environment)")
    except subprocess.TimeoutExpired:
        print(f"    WARNING: msprof timed out after 600s for case {case_id}.")
    except Exception as e:
        print(f"    WARNING: msprof failed: {e}")
    finally:
        if os.path.exists(wrapper_path):
            os.remove(wrapper_path)


def main():
    global KERNEL_FILE, RESULTS_DIR

    parser = argparse.ArgumentParser(
        description="Benchmark Triton-Ascend FlashAttention with msprof profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Baseline kernel\n"
        "  python benchmark_06_msprof.py --kernel ../baseline/06-fused-attention.py --all\n"
            "  # Optimized kernel, separate output\n"
        "  python benchmark_06_msprof.py --kernel ../ascend_rsfa/flash_attention_forward.py --out result_opt\n"
        ),
    )
    parser.add_argument(
        "--single", type=int, default=None,
        help="Run one case (1-12) as a profiling workload (used by msprof subprocess)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all 12 cases with timing, baseline, and msprof"
    )
    parser.add_argument(
        "--kernel", type=str, default=None,
        help="Path to the kernel Python file (default: 06-fused-attention.py)"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Results output directory (default: ./result)"
    )
    parser.add_argument(
        "--force-fallback-tiling", action="store_true",
        help="Force fallback FALLBACK_TILING table instead of kernel get_tiling()"
    )
    args = parser.parse_args()

    if args.single is None and not args.all:
        parser.print_help()
        sys.exit(1)

    # Resolve kernel file
    tutorial_dir = os.path.dirname(os.path.abspath(__file__))
    if args.kernel:
        KERNEL_FILE = os.path.join(tutorial_dir, args.kernel) if not os.path.isabs(args.kernel) else args.kernel
    else:
        KERNEL_FILE = DEFAULT_KERNEL

    # Resolve output directory
    if args.out:
        RESULTS_DIR = os.path.join(tutorial_dir, args.out) if not os.path.isabs(args.out) else args.out
    else:
        RESULTS_DIR = DEFAULT_OUT

    kernel_name = os.path.basename(KERNEL_FILE)

    # Dynamic import of the kernel
    print(f"Kernel file    : {KERNEL_FILE}")
    print(f"Output dir     : {RESULTS_DIR}")
    try:
        global kernel_mod
        kernel_mod = load_kernel(KERNEL_FILE)
        attention_fn = kernel_mod.attention
    except Exception as e:
        print(f"ERROR: Failed to load kernel: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Kernel loaded. Device: {DEVICE}  |  Timing runs: {NUM_TIMING_RUNS}")
    if args.force_fallback_tiling:
        print(f"Tiling mode: forced fallback (kernel get_tiling() will be skipped)")

    if args.single is not None:
        if args.single < 1 or args.single > len(TEST_CASES):
            print(f"ERROR: --single must be between 1 and {len(TEST_CASES)}", file=sys.stderr)
            sys.exit(1)
        run_single_case(args.single, attention_fn, KERNEL_FILE, args.force_fallback_tiling)
    else:
        run_all_cases(attention_fn, kernel_name, args.force_fallback_tiling)


kernel_mod = None
RESULTS_DIR = DEFAULT_OUT
KERNEL_FILE = DEFAULT_KERNEL

if __name__ == "__main__":
    main()
