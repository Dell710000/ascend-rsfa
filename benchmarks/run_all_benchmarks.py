#!/usr/bin/env python3
"""
Master benchmark orchestrator for flash attention performance comparison.
Runs 12 test cases for both the optimized (flash_attention_forward.py) and
baseline (06-fused-attention.py) kernels, collects results, and computes speedups.

Output: user_data/benchmark_results.txt
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime


# ---- Configuration ----
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPTIMIZED_KERNEL = os.path.join(WORKSPACE, "ascend_rsfa/flash_attention_forward.py")
BASELINE_KERNEL = os.path.join(WORKSPACE, "baseline/06-fused-attention.py")
RUNNER_SCRIPT = os.path.join(WORKSPACE, "benchmarks", "run_single_test.py")
OUTPUT_FILE = os.path.join(WORKSPACE, "benchmarks", "benchmark_results.txt")

# ---- 12 Test Cases (from flash_attention_forward.py __main__) ----
TEST_CASES = [
    # Causal=True (6 cases)
    {"Z": 128, "H": 8, "N_CTX": 1024, "HEAD_DIM": 128, "causal": True,  "dtype": "fp16", "BM": 128, "BN": 64},
    {"Z": 128, "H": 8, "N_CTX": 1024, "HEAD_DIM": 256, "causal": True,  "dtype": "fp16", "BM": 64,  "BN": 64},
    {"Z": 128, "H": 8, "N_CTX": 2048, "HEAD_DIM": 128, "causal": True,  "dtype": "fp16", "BM": 128, "BN": 64},
    {"Z": 128, "H": 8, "N_CTX": 2048, "HEAD_DIM": 256, "causal": True,  "dtype": "fp16", "BM": 64,  "BN": 64},
    {"Z": 128, "H": 8, "N_CTX": 4096, "HEAD_DIM": 128, "causal": True,  "dtype": "fp16", "BM": 128, "BN": 64},
    {"Z": 128, "H": 8, "N_CTX": 8192, "HEAD_DIM": 64,  "causal": True,  "dtype": "fp16", "BM": 128, "BN": 64},
    # Causal=False (6 cases)
    {"Z": 128, "H": 8, "N_CTX": 1024, "HEAD_DIM": 128, "causal": False, "dtype": "fp16", "BM": 64,  "BN": 128},
    {"Z": 128, "H": 8, "N_CTX": 1024, "HEAD_DIM": 256, "causal": False, "dtype": "fp16", "BM": 64,  "BN": 128},
    {"Z": 128, "H": 8, "N_CTX": 2048, "HEAD_DIM": 128, "causal": False, "dtype": "fp16", "BM": 64,  "BN": 128},
    {"Z": 128, "H": 8, "N_CTX": 2048, "HEAD_DIM": 256, "causal": False, "dtype": "fp16", "BM": 64,  "BN": 256},
    {"Z": 128, "H": 8, "N_CTX": 4096, "HEAD_DIM": 128, "causal": False, "dtype": "fp16", "BM": 128, "BN": 256},
    {"Z": 128, "H": 8, "N_CTX": 8192, "HEAD_DIM": 64,  "causal": False, "dtype": "fp16", "BM": 128, "BN": 512},
]


def case_label(case, idx):
    """Return a short label for a test case."""
    causal_str = "causal" if case["causal"] else "noncausal"
    return (f"Case {idx:2d}: Z={case['Z']:3d}, H={case['H']}, N_CTX={case['N_CTX']:4d}, "
            f"D={case['HEAD_DIM']:3d}, {causal_str:9s}, {case['dtype']}, "
            f"BM={case['BM']:3d}, BN={case['BN']:3d}")


def run_one(kernel_path, case, warmup=5, iters=20):
    """Run a single benchmark case via subprocess."""
    cmd = [
        sys.executable, RUNNER_SCRIPT,
        "--kernel", kernel_path,
        "--Z", str(case["Z"]),
        "--H", str(case["H"]),
        "--N_CTX", str(case["N_CTX"]),
        "--HEAD_DIM", str(case["HEAD_DIM"]),
        "--causal", "1" if case["causal"] else "0",
        "--dtype", case["dtype"],
        "--BM", str(case["BM"]),
        "--BN", str(case["BN"]),
        "--warmup", str(warmup),
        "--iters", str(iters),
    ]

    # Suppress stderr from torch_npu/Triton compilation noise
    env = os.environ.copy()
    env.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "3")  # Reduce CANN log noise

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout per case
            env=env,
            cwd=WORKSPACE,
        )
        stdout = result.stdout.strip() if result.stdout else ""
        stderr = result.stderr.strip() if result.stderr else ""
        # Find the JSON line (last non-empty line that starts with '{')
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        # For debugging: include raw output in error
        err_detail = f"RC={result.returncode}"
        if stdout:
            err_detail += f" | stdout[0:500]={stdout[:500]}"
        if stderr:
            err_detail += f" | stderr[0:500]={stderr[:500]}"
        return {"error": err_detail}
    except subprocess.TimeoutExpired:
        return {"error": "Timeout (>600s)"}
    except Exception as e:
        return {"error": str(e)}


def format_time(ms):
    """Format milliseconds nicely."""
    if ms is None:
        return "    N/A"
    if ms < 1:
        return f"{ms*1000:7.2f} us"
    elif ms < 1000:
        return f"{ms:7.2f} ms"
    else:
        return f"{ms/1000:7.3f} s"


def generate_report(opt_results, base_results):
    """Generate the benchmark report txt file."""
    lines = []
    sep = "=" * 120

    lines.append(sep)
    lines.append("  Flash Attention Performance Benchmark Report")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Hardware: Ascend910B3")
    lines.append(f"  Framework: torch_npu + Triton-Ascend")
    lines.append(sep)
    lines.append("")

    # Table header
    header = (f"{'Case':<6s} {'Z':>4s} {'H':>2s} {'N_CTX':>6s} {'D':>4s} {'Causal':>9s} {'dtype':>6s} {'BM':>4s} {'BN':>4s}  "
              f"{'06-Baseline':>12s}  {'Optimized':>12s}  {'Speedup':>8s}  {'Faster':>10s}")
    lines.append(header)
    lines.append("-" * 120)

    total_opt = 0.0
    total_base = 0.0
    valid_count = 0

    for i, (opt_r, base_r) in enumerate(zip(opt_results, base_results)):
        case = TEST_CASES[i]
        causal_str = "causal" if case["causal"] else "noncausal"

        opt_ms = opt_r.get("avg_ms") if isinstance(opt_r, dict) and "error" not in opt_r else None
        base_ms = base_r.get("avg_ms") if isinstance(base_r, dict) and "error" not in base_r else None

        if opt_ms is not None and base_ms is not None:
            speedup = base_ms / opt_ms
            faster = "optimized" if speedup > 1.0 else "baseline"
            speedup_str = f"{speedup:6.2f}x"
            faster_str = f"{faster}"
            total_opt += opt_ms
            total_base += base_ms
            valid_count += 1
        elif opt_ms is not None:
            speedup_str = "   N/A"
            faster_str = "(baseline failed)"
        elif base_ms is not None:
            speedup_str = "   N/A"
            faster_str = "(optimized failed)"
        else:
            speedup_str = "   N/A"
            faster_str = "(both failed)"

        row = (f"Case{i+1:02d}  {case['Z']:4d} {case['H']:2d} {case['N_CTX']:6d} {case['HEAD_DIM']:4d} "
               f"{causal_str:>9s} {case['dtype']:>6s} {case['BM']:4d} {case['BN']:4d}  "
               f"{format_time(base_ms):>12s}  {format_time(opt_ms):>12s}  {speedup_str:>8s}  {faster_str:>10s}")
        lines.append(row)

    lines.append("-" * 120)

    # Error reporting
    errors = []
    for i, (opt_r, base_r) in enumerate(zip(opt_results, base_results)):
        if isinstance(opt_r, dict) and "error" in opt_r:
            errors.append(f"  Case {i+1:2d} Optimized ERROR: {opt_r['error']}")
        if isinstance(base_r, dict) and "error" in base_r:
            errors.append(f"  Case {i+1:2d} Baseline  ERROR: {base_r['error']}")

    if errors:
        lines.append("")
        lines.append("Errors encountered:")
        for e in errors:
            lines.append(e)
        lines.append("")

    # Summary
    lines.append("")
    lines.append(sep)
    lines.append("  Summary")
    lines.append(sep)

    if valid_count > 0:
        avg_speedup = total_base / total_opt
        lines.append(f"  Valid test cases:     {valid_count} / {len(TEST_CASES)}")
        lines.append(f"  Total baseline time:  {format_time(total_base)}")
        lines.append(f"  Total optimized time: {format_time(total_opt)}")
        lines.append(f"  Overall speedup:      {avg_speedup:.2f}x")
        lines.append(f"  Time saved:           {format_time(total_base - total_opt)}")
    else:
        lines.append("  No valid comparisons available.")

    # Per-test-case details
    lines.append("")
    lines.append(sep)
    lines.append("  Detailed Results (per test case)")
    lines.append(sep)

    for i, (opt_r, base_r) in enumerate(zip(opt_results, base_results)):
        case = TEST_CASES[i]
        lines.append(f"\n  --- {case_label(case, i+1)} ---")

        if isinstance(opt_r, dict) and "error" not in opt_r:
            lines.append(f"    Optimized:  avg={format_time(opt_r['avg_ms'])}, "
                         f"min={format_time(opt_r['min_ms'])}, max={format_time(opt_r['max_ms'])}, "
                         f"iters={opt_r.get('iters', '?')}")
        else:
            lines.append(f"    Optimized:  ERROR - {opt_r.get('error', 'unknown') if isinstance(opt_r, dict) else str(opt_r)}")

        if isinstance(base_r, dict) and "error" not in base_r:
            lines.append(f"    Baseline:   avg={format_time(base_r['avg_ms'])}, "
                         f"min={format_time(base_r['min_ms'])}, max={format_time(base_r['max_ms'])}, "
                         f"iters={base_r.get('iters', '?')}")
        else:
            lines.append(f"    Baseline:   ERROR - {base_r.get('error', 'unknown') if isinstance(base_r, dict) else str(base_r)}")

        if (isinstance(opt_r, dict) and "error" not in opt_r and
            isinstance(base_r, dict) and "error" not in base_r):
            sp = base_r["avg_ms"] / opt_r["avg_ms"]
            lines.append(f"    Speedup:    {sp:.2f}x ({'optimized' if sp > 1 else 'baseline'} is faster)")

    lines.append("")
    lines.append(sep)
    lines.append("  End of Report")
    lines.append(sep)

    return "\n".join(lines)


def main():
    print("=" * 80)
    print("  Flash Attention Performance Benchmark")
    print("  Comparing: flash_attention_forward.py (optimized) vs 06-fused-attention.py (baseline)")
    print(f"  Test cases: {len(TEST_CASES)}")
    print("=" * 80)
    print()

    opt_results = []
    base_results = []

    for idx, case in enumerate(TEST_CASES):
        label = case_label(case, idx + 1)
        print(f"\n[{idx+1}/{len(TEST_CASES)}] {label}")
        print("-" * 80)

        # Run optimized kernel
        print(f"  Running optimized kernel... ", end="", flush=True)
        opt_r = run_one(OPTIMIZED_KERNEL, case, warmup=3, iters=10)
        opt_results.append(opt_r)
        if isinstance(opt_r, dict) and "error" not in opt_r:
            print(f"avg={format_time(opt_r['avg_ms'])}")
        else:
            print(f"ERROR: {opt_r.get('error', 'unknown')}")

        # Run baseline kernel
        print(f"  Running baseline kernel ... ", end="", flush=True)
        base_r = run_one(BASELINE_KERNEL, case, warmup=3, iters=10)
        base_results.append(base_r)
        if isinstance(base_r, dict) and "error" not in base_r:
            print(f"avg={format_time(base_r['avg_ms'])}")
        else:
            print(f"ERROR: {base_r.get('error', 'unknown')}")

    # Generate report
    print("\n" + "=" * 80)
    print("  Generating report...")

    report = generate_report(opt_results, base_results)

    with open(OUTPUT_FILE, "w") as f:
        f.write(report)

    print(f"  Report saved to: {OUTPUT_FILE}")
    print("=" * 80)

    # Print summary to console
    print("\n" + report)


if __name__ == "__main__":
    main()
