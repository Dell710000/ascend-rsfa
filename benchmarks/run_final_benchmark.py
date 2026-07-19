#!/usr/bin/env python3
"""
Final benchmark: compare optimized vs baseline using OPTIMAL tilings for each case.
Runs with more iterations for stable timing.
"""

import importlib.util
import json
import os
import sys
import time

import torch
import torch_npu

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

DEVICE = "npu"
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Optimal tilings from search results ----
OPTIMAL_TILINGS = {
    "optimized": {
        # case_idx: (BM, BN) — 全部通过 torch.allclose(rtol=1e-2) 正确性验证
        1:  (128, 64),
        2:  (64, 64),     # (64,256)→NaN, 仅 (64,64) 通过验证
        3:  (128, 64),
        4:  (64, 64),     # (64,256)→NaN, 仅 (64,64) 通过验证
        5:  (128, 64),    # (128,64) 比 (128,32) 快 10%
        6:  (128, 64),    # (64,256)→NaN, (128,64) 通过验证
        7:  (128, 256),
        8:  (64, 256),    # (64,256) 比 (64,128) 快 2x
        9:  (128, 256),
        10: (64, 256),
        11: (128, 256),
        12: (128, 512),
    },
    "baseline": {
        # case_idx: (BM, BN) — 与 optimized 相同分块 (全部通过正确性验证)
        1:  (128, 64),    # 同 opt ✓
        2:  (64, 64),     # 同 opt ✓
        3:  (128, 64),    # 同 opt ✓
        4:  (64, 64),     # 同 opt ✓
        5:  (128, 64),    # 同 opt ✓ (baseline 用 (128,32) 会退化到 385ms)
        6:  (128, 64),    # 同 opt ✓
        7:  (128, 128),   # opt(128,256) UB溢出, (128,128) 最快可行
        8:  (64, 256),    # 同 opt ✓
        9:  (128, 128),   # opt(128,256) UB溢出, (128,128) 最快可行
        10: (64, 256),    # 同 opt ✓
        11: (128, 128),   # opt(128,256) UB溢出, (128,128) 最快可行
        12: (128, 256),   # opt(128,512) UB溢出, (128,256) 最快且最接近 opt
    },
}

# ---- 12 Test Cases ----
TEST_CASES = [
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


def load_kernel(kernel_path):
    clean_name = os.path.splitext(os.path.basename(kernel_path))[0]
    clean_name = clean_name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    spec = importlib.util.spec_from_file_location(clean_name, kernel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[clean_name] = mod
    spec.loader.exec_module(mod)
    return mod


def benchmark_kernel(kernel_path, tilings_map, kernel_label, warmup=5, iters=20):
    """Run benchmark for all 12 cases with optimal tilings."""
    print(f"\n{'='*80}")
    print(f"  Benchmarking: {kernel_label}")
    print(f"  Kernel: {kernel_path}")
    print(f"  Warmup: {warmup}, Iters: {iters}")
    print(f"{'='*80}")

    mod = load_kernel(kernel_path)
    attention_fn = mod.attention
    results = {}

    for case in TEST_CASES:
        idx, Z, H, N_CTX, HEAD_DIM, causal, dtype = case
        BM, BN = tilings_map[idx]
        causal_str = "causal" if causal else "noncausal"

        print(f"\n  [Case {idx:2d}] Z={Z}, H={H}, N_CTX={N_CTX}, D={HEAD_DIM}, "
              f"{causal_str}, BM={BM}, BN={BN}")

        # Create tensors
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
            print(f"    avg={avg_ms:.4f} ms, min={min_ms:.4f} ms, max={max_ms:.4f} ms")

            results[idx] = {
                "avg_ms": round(avg_ms, 4),
                "min_ms": round(min_ms, 4),
                "max_ms": round(max_ms, 4),
                "BM": BM, "BN": BN,
                "error": None
            }
        except Exception as e:
            print(f"    FAILED: {e}")
            results[idx] = {
                "avg_ms": None, "min_ms": None, "max_ms": None,
                "BM": BM, "BN": BN,
                "error": str(e)[:200]
            }

    return results


def generate_report(opt_results, base_results):
    """Generate the final comparison report."""
    lines = []
    sep = "=" * 120

    lines.append(sep)
    lines.append("  Flash Attention - Final Performance Benchmark (Optimal Tiling)")
    lines.append(f"  Hardware: Ascend910B3 | Framework: torch_npu + Triton-Ascend")
    lines.append(sep)
    lines.append("")
    lines.append("  Optimized kernel:  flash_attention_forward.py (FlashAttention v2 optimized)")
    lines.append("  Baseline kernel:   06-fused-attention.py       (FlashAttention v2 baseline)")
    lines.append("")
    lines.append("  Note: Optimal BM/BN tiling determined via exhaustive search for each case+kernel.")
    lines.append("")

    # Table header
    header = (f"{'Case':<6s} {'Z':>4s} {'H':>2s} {'N_CTX':>6s} {'D':>4s} {'Causal':>9s}  "
              f"{'06-Baseline':>12s}  {'Optimized':>12s}  {'Speedup':>8s}  {'Faster':>10s}  "
              f"{'Opt BM,BN':>14s}  {'Base BM,BN':>14s}")
    lines.append(header)
    lines.append("-" * 120)

    total_opt = 0.0
    total_base = 0.0
    valid_count = 0
    max_speedup = 0
    min_speedup = float('inf')

    for case in TEST_CASES:
        idx = case[0]
        _, Z, H, N_CTX, HEAD_DIM, causal, dtype = case
        causal_str = "causal" if causal else "noncausal"

        opt_r = opt_results.get(idx, {})
        base_r = base_results.get(idx, {})

        opt_ms = opt_r.get("avg_ms") if opt_r and not opt_r.get("error") else None
        base_ms = base_r.get("avg_ms") if base_r and not base_r.get("error") else None

        opt_bm, opt_bn = opt_r.get("BM", "?"), opt_r.get("BN", "?")
        base_bm, base_bn = base_r.get("BM", "?"), base_r.get("BN", "?")

        opt_tile_str = f"({opt_bm},{opt_bn})" if opt_bm != "?" else "?"
        base_tile_str = f"({base_bm},{base_bn})" if base_bm != "?" else "?"

        if opt_ms is not None and base_ms is not None:
            speedup = base_ms / opt_ms
            faster = "optimized" if speedup > 1.005 else ("baseline" if speedup < 0.995 else "tie")
            speedup_str = f"{speedup:6.2f}x"
            total_opt += opt_ms
            total_base += base_ms
            valid_count += 1
            if speedup > max_speedup:
                max_speedup = speedup
            if speedup < min_speedup:
                min_speedup = speedup
        elif opt_ms is not None:
            speedup_str = "   N/A"
            faster = "(base failed)"
        elif base_ms is not None:
            speedup_str = "   N/A"
            faster = "(opt failed)"
        else:
            speedup_str = "   N/A"
            faster = "(both failed)"

        def fmt_t(t):
            if t is None: return "        N/A"
            if t < 1: return f"{t*1000:7.2f} us"
            if t < 1000: return f"{t:7.2f} ms"
            return f"{t/1000:7.3f} s"

        row = (f"Case{idx:02d}  {Z:4d} {H:2d} {N_CTX:6d} {HEAD_DIM:4d} "
               f"{causal_str:>9s}  {fmt_t(base_ms):>12s}  {fmt_t(opt_ms):>12s}  "
               f"{speedup_str:>8s}  {faster:>10s}  {opt_tile_str:>14s}  {base_tile_str:>14s}")
        lines.append(row)

    lines.append("-" * 120)

    # Errors
    for case in TEST_CASES:
        idx = case[0]
        opt_r = opt_results.get(idx, {})
        base_r = base_results.get(idx, {})
        if opt_r and opt_r.get("error"):
            lines.append(f"  Case {idx:2d} Optimized ERROR: {opt_r['error'][:150]}")
        if base_r and base_r.get("error"):
            lines.append(f"  Case {idx:2d} Baseline  ERROR: {base_r['error'][:150]}")

    lines.append("")
    lines.append(sep)
    lines.append("  Summary")
    lines.append(sep)

    if valid_count > 0:
        avg_speedup = total_base / total_opt
        lines.append(f"  Valid test cases:     {valid_count} / {len(TEST_CASES)}")
        lines.append(f"  Total baseline time:  {total_base:.3f} ms  ({total_base/1000:.3f} s)")
        lines.append(f"  Total optimized time: {total_opt:.3f} ms  ({total_opt/1000:.3f} s)")
        lines.append(f"  Overall speedup:      {avg_speedup:.2f}x")
        lines.append(f"  Speedup range:        {min_speedup:.2f}x ~ {max_speedup:.2f}x")
        lines.append(f"  Time saved:           {(total_base - total_opt):.3f} ms  ({(total_base-total_opt)/1000:.3f} s)")

        # Per-category stats
        causal_opt = sum(opt_results[i]["avg_ms"] for i in range(1,7)
                        if opt_results[i] and not opt_results[i].get("error")
                        and base_results[i] and not base_results[i].get("error"))
        causal_base = sum(base_results[i]["avg_ms"] for i in range(1,7)
                         if opt_results[i] and not opt_results[i].get("error")
                         and base_results[i] and not base_results[i].get("error"))
        noncausal_opt = sum(opt_results[i]["avg_ms"] for i in range(7,13)
                           if opt_results[i] and not opt_results[i].get("error")
                           and base_results[i] and not base_results[i].get("error"))
        noncausal_base = sum(base_results[i]["avg_ms"] for i in range(7,13)
                            if opt_results[i] and not opt_results[i].get("error")
                            and base_results[i] and not base_results[i].get("error"))

        if causal_base > 0:
            lines.append(f"  Causal speedup:       {causal_base/causal_opt:.2f}x")
        if noncausal_base > 0:
            lines.append(f"  Non-causal speedup:   {noncausal_base/noncausal_opt:.2f}x")
    else:
        lines.append("  No valid comparisons.")

    # Per-case details
    lines.append("")
    lines.append(sep)
    lines.append("  Per-Case Details")
    lines.append(sep)

    for case in TEST_CASES:
        idx = case[0]
        _, Z, H, N_CTX, HEAD_DIM, causal, dtype = case
        causal_str = "causal" if causal else "noncausal"
        opt_r = opt_results.get(idx, {})
        base_r = base_results.get(idx, {})

        lines.append(f"\n  --- Case {idx:2d}: Z={Z}, H={H}, N_CTX={N_CTX}, D={HEAD_DIM}, {causal_str} ---")

        if opt_r and not opt_r.get("error"):
            lines.append(f"    Optimized:  avg={opt_r['avg_ms']:.4f} ms, "
                         f"min={opt_r['min_ms']:.4f} ms, max={opt_r['max_ms']:.4f} ms, "
                         f"BM={opt_r['BM']}, BN={opt_r['BN']}")
        else:
            lines.append(f"    Optimized:  ERROR - {opt_r.get('error', 'N/A') if opt_r else 'N/A'}")

        if base_r and not base_r.get("error"):
            lines.append(f"    Baseline:   avg={base_r['avg_ms']:.4f} ms, "
                         f"min={base_r['min_ms']:.4f} ms, max={base_r['max_ms']:.4f} ms, "
                         f"BM={base_r['BM']}, BN={base_r['BN']}")
        else:
            lines.append(f"    Baseline:   ERROR - {base_r.get('error', 'N/A') if base_r else 'N/A'}")

        if (opt_r and not opt_r.get("error") and base_r and not base_r.get("error")):
            sp = base_r["avg_ms"] / opt_r["avg_ms"]
            faster = "optimized" if sp > 1.005 else ("baseline" if sp < 0.995 else "tie")
            lines.append(f"    Speedup:    {sp:.2f}x ({faster} is faster)")

    lines.append("")
    lines.append(sep)
    lines.append("  End of Report")
    lines.append(sep)

    return "\n".join(lines)


def main():
    print("=" * 80)
    print("  Flash Attention Final Benchmark (Optimal Tiling)")
    print("=" * 80)

    # Benchmark optimized kernel
    opt_results = benchmark_kernel(
        os.path.join(WORKSPACE, "ascend_rsfa/flash_attention_forward.py"),
        OPTIMAL_TILINGS["optimized"],
        "Optimized (flash_attention_forward.py)",
        warmup=5, iters=20
    )

    # Benchmark baseline kernel
    base_results = benchmark_kernel(
        os.path.join(WORKSPACE, "baseline/06-fused-attention.py"),
        OPTIMAL_TILINGS["baseline"],
        "Baseline (06-fused-attention.py)",
        warmup=5, iters=20
    )

    # Save raw results
    with open(os.path.join(OUTPUT_DIR, "final_benchmark_raw.json"), "w") as f:
        json.dump({"optimized": opt_results, "baseline": base_results}, f, indent=2)

    # Generate and save report
    report = generate_report(opt_results, base_results)
    report_path = os.path.join(OUTPUT_DIR, "benchmark_results.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")
    print(report)


if __name__ == "__main__":
    main()
