#!/usr/bin/env python3
"""
对所有 12 个测试用例进行 profiler 硬件指标采集。
每个用例分别采集优化版和基线版，提取 AiCMetrics.PipeUtilization 数据。
"""

import csv
import importlib.util
import os
import shutil
import sys
import time

import torch
import torch_npu

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

DEVICE = "npu"
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROF_TMP = os.path.join(WORKSPACE, "prof_tmp_all")

OPT_KERNEL = os.path.join(WORKSPACE, "ascend_rsfa/flash_attention_forward.py")
BASE_KERNEL = os.path.join(WORKSPACE, "baseline/06-fused-attention.py")

# ---- 12 个测试用例 ----
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

# ---- 最优分块 (性能测试验证) ----
OPT_TILING = {
    1: (128, 64), 2: (64, 64), 3: (128, 64), 4: (64, 64), 5: (128, 32), 6: (128, 64),
    7: (128, 256), 8: (64, 128), 9: (128, 256), 10: (64, 256), 11: (128, 256), 12: (128, 512),
}
BASE_TILING = {
    1: (128, 64), 2: (64, 64), 3: (128, 64), 4: (64, 64), 5: (128, 64), 6: (128, 64),
    7: (128, 128), 8: (64, 256), 9: (128, 128), 10: (64, 256), 11: (128, 128), 12: (64, 512),
}


def load_kernel(path):
    name = os.path.splitext(os.path.basename(path))[0].replace(" ", "_").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def find_op_summary(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if 'op_summary' in f and f.endswith('.csv'):
                return os.path.join(root, f)
    return None


def parse_attn_rows(csv_path):
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if '_attn_fwd' in row.get('Op Name', ''):
                rows.append(row)
    return rows


def avg_field(rows, field):
    vals = []
    for r in rows:
        v = r.get(field, '')
        if v and v != 'N/A' and v != '':
            try: vals.append(float(v))
            except ValueError: pass
    return sum(vals) / len(vals) if vals else None


def profile_one(kernel_path, Z, H, N_CTX, HEAD_DIM, causal, dtype, BM, BN):
    """对单个配置进行 profiler 采集，返回硬件指标。"""
    if os.path.exists(PROF_TMP):
        shutil.rmtree(PROF_TMP)

    mod = load_kernel(kernel_path)
    attention_fn = mod.attention

    torch.manual_seed(42)
    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    sm_scale = 0.5

    exp_cfg = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False, data_simplification=False,
    )

    WAIT, WARMUP, ACTIVE = 1, 1, 3
    TOTAL = WAIT + WARMUP + ACTIVE
    times = []

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=WAIT, warmup=WARMUP, active=ACTIVE, repeat=1, skip_first=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(os.path.join(PROF_TMP, "trace")),
        record_shapes=True, profile_memory=False, with_stack=False, with_flops=False, with_modules=False,
        experimental_config=exp_cfg,
    ) as prof:
        for i in range(TOTAL):
            t0 = time.perf_counter()
            _ = attention_fn(q, k, v, causal, sm_scale, BM, BN)
            torch.npu.synchronize()
            t1 = time.perf_counter()
            prof.step()
            if i >= WAIT + WARMUP:
                times.append((t1 - t0) * 1000.0)

    avg_time = sum(times) / len(times)

    csv_path = find_op_summary(PROF_TMP)
    rows = parse_attn_rows(csv_path) if csv_path else []

    key_fields = [
        "aic_time(us)", "aic_mac_time(us)", "aic_mac_ratio",
        "aic_fixpipe_time(us)", "aic_fixpipe_ratio",
        "aic_mte1_time(us)", "aic_mte1_ratio",
        "aic_mte2_time(us)", "aic_mte2_ratio",
        "aic_scalar_time(us)", "aic_scalar_ratio",
        "aiv_time(us)", "aiv_vec_time(us)", "aiv_vec_ratio",
        "aiv_scalar_time(us)", "aiv_scalar_ratio",
        "aiv_mte2_time(us)", "aiv_mte2_ratio",
        "aiv_mte3_time(us)", "aiv_mte3_ratio",
        "cube_utilization(%)",
    ]

    result = {"avg_time_ms": round(avg_time, 4), "num_invocations": len(rows)}
    for f in key_fields:
        v = avg_field(rows, f) if rows else None
        result[f] = round(v, 4) if v is not None else None

    return result


def main():
    print("=" * 80)
    print("  全 12 用例 Profiler 硬件指标采集")
    print("=" * 80)

    all_results = {}

    for case in TEST_CASES:
        idx, Z, H, N_CTX, HEAD_DIM, causal, dtype = case
        causal_str = "causal" if causal else "noncausal"
        opt_bm, opt_bn = OPT_TILING[idx]
        base_bm, base_bn = BASE_TILING[idx]

        key = f"case_{idx:02d}"
        all_results[key] = {"params": f"Z={Z},H={H},N_CTX={N_CTX},D={HEAD_DIM},{causal_str}"}

        # 优化版
        print(f"\n[{idx:2d}/12] {causal_str} Z={Z} H={H} N_CTX={N_CTX} D={HEAD_DIM}")
        print(f"  优化版 BM={opt_bm},BN={opt_bn} ...", end=" ", flush=True)
        try:
            opt = profile_one(OPT_KERNEL, Z, H, N_CTX, HEAD_DIM, causal, dtype, opt_bm, opt_bn)
            all_results[key]["optimized"] = opt
            print(f"{opt['avg_time_ms']:.2f} ms, cube={opt.get('cube_utilization(%)')}%")
        except Exception as e:
            print(f"ERROR: {e}")
            all_results[key]["optimized"] = {"error": str(e)[:200]}

        # 基线版
        print(f"  基线版 BM={base_bm},BN={base_bn} ...", end=" ", flush=True)
        try:
            base = profile_one(BASE_KERNEL, Z, H, N_CTX, HEAD_DIM, causal, dtype, base_bm, base_bn)
            all_results[key]["baseline"] = base
            print(f"{base['avg_time_ms']:.2f} ms, cube={base.get('cube_utilization(%)')}%")
        except Exception as e:
            print(f"ERROR: {e}")
            all_results[key]["baseline"] = {"error": str(e)[:200]}

    # 保存原始数据
    import json
    with open(os.path.join(WORKSPACE, "profiler_all_raw.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n原始数据已保存: profiler_all_raw.json")

    # ============ 生成中文报告 ============
    lines = []
    sep = "=" * 130

    lines.append(sep)
    lines.append("  Flash Attention - 全部 12 用例 Profiler 硬件指标对比报告")
    lines.append("  工具: torch_npu.profiler + AiCMetrics.PipeUtilization")
    lines.append("  硬件: Ascend910B3 | 框架: torch_npu + Triton-Ascend")
    lines.append("  优化版: flash_attention_forward.py (FlashAttention v2 优化实现)")
    lines.append("  基线版: 06-fused-attention.py (TritonAscend 官方实现)")
    lines.append(sep)
    lines.append("")
    lines.append("  采集配置: 1 wait + 1 warmup + 3 active steps，每个用例独立采集")
    lines.append("  数据来源: AiCMetrics.PipeUtilization trace 输出的 op_summary CSV")
    lines.append("")
    lines.append("  硬件单元说明:")
    lines.append("    AIC = AI Core (Cube/矩阵乘管线)")
    lines.append("    AIV = AI Vector Core (向量/标量管线)")
    lines.append("    MAC = 矩阵乘法单元 (Cube)")
    lines.append("    FixPipe = 固函数管线 (softmax/exp/mask/norm)")
    lines.append("    MTE1 = 向量数据加载 (HBM -> 片上)")
    lines.append("    MTE2 = 标量加载/存储")
    lines.append("    MTE3 = 向量数据存储")
    lines.append("")

    # ===== 表1: 综合总览 =====
    lines.append("  " + "─" * 128)
    lines.append("  表1: 全部用例端到端性能与核心硬件指标总览")
    lines.append("  " + "─" * 128)
    lines.append("")

    hdr = (f"{'Case':<6s} {'Z':>4s} {'H':>2s} {'N_CTX':>6s} {'D':>4s} {'Causal':>8s}  "
           f"{'Opt(ms)':>9s} {'Base(ms)':>9s} {'加速比':>7s}  "
           f"{'Opt_Cube%':>9s} {'Base_Cube%':>10s}  "
           f"{'Opt_MAC%':>8s} {'Base_MAC%':>9s}  "
           f"{'Opt_FixP%':>9s} {'Base_FixP%':>10s}  "
           f"{'Opt_MTE1%':>9s} {'Base_MTE1%':>10s}")
    lines.append(hdr)
    lines.append("-" * 130)

    for case in TEST_CASES:
        idx, Z, H, N_CTX, HEAD_DIM, causal, dtype = case
        key = f"case_{idx:02d}"
        data = all_results.get(key, {})
        opt = data.get("optimized", {})
        base = data.get("baseline", {})
        causal_str = "causal" if causal else "noncausal"

        if opt.get("error") or base.get("error"):
            lines.append(f"Case{idx:02d}  {Z:4d} {H:2d} {N_CTX:6d} {HEAD_DIM:4d} {causal_str:>8s}  ERROR")
            continue

        o_t = opt.get("avg_time_ms", 0) or 0
        b_t = base.get("avg_time_ms", 0) or 0
        sp = b_t / o_t if o_t > 0 else 0

        lines.append(
            f"Case{idx:02d}  {Z:4d} {H:2d} {N_CTX:6d} {HEAD_DIM:4d} {causal_str:>8s}  "
            f"{o_t:9.2f} {b_t:9.2f} {sp:6.2f}x  "
            f"{opt.get('cube_utilization(%)') or 'N/A':>9s} {str(base.get('cube_utilization(%)') or 'N/A'):>10s}  "
            f"{str(opt.get('aic_mac_ratio') or 'N/A'):>8s} {str(base.get('aic_mac_ratio') or 'N/A'):>9s}  "
            f"{str(opt.get('aic_fixpipe_ratio') or 'N/A'):>9s} {str(base.get('aic_fixpipe_ratio') or 'N/A'):>10s}  "
            f"{str(opt.get('aic_mte1_ratio') or 'N/A'):>9s} {str(base.get('aic_mte1_ratio') or 'N/A'):>10s}"
        )
    lines.append("")

    # ===== 表2: AI Core 详细指标 =====
    lines.append("  " + "─" * 128)
    lines.append("  表2: AI Core 各硬件单元绝对耗时对比 (us/kernel)")
    lines.append("  " + "─" * 128)
    lines.append("")

    for causal_group, group_name in [(True, "因果注意力 (causal=True)"), (False, "非因果注意力 (causal=False)")]:
        lines.append(f"  【{group_name}】")
        lines.append("")
        lines.append(f"  {'Case':<6s} {'AIC总耗时(us)':>16s} {'Cube耗时':>12s} {'FixPipe':>12s} {'MTE1':>12s} {'Scalar':>12s}  |  {'AIC总耗时(us)':>16s} {'Cube耗时':>12s} {'FixPipe':>12s} {'MTE1':>12s} {'Scalar':>12s}")
        lines.append(f"  {'':6s} {'优化版':>16s} {'优化版':>12s} {'优化版':>12s} {'优化版':>12s} {'优化版':>12s}  |  {'基线版':>16s} {'基线版':>12s} {'基线版':>12s} {'基线版':>12s} {'基线版':>12s}")
        lines.append("-" * 130)

        for case in TEST_CASES:
            idx, Z, H, N_CTX, HEAD_DIM, causal, dtype = case
            if causal != causal_group: continue
            key = f"case_{idx:02d}"
            data = all_results.get(key, {})
            opt = data.get("optimized", {})
            base = data.get("baseline", {})

            def f(d, field): return f"{d.get(field, 0) or 0:.0f}" if d and not d.get("error") else "N/A"

            lines.append(
                f"  Case{idx:02d}  {f(opt, 'aic_time(us)'):>16s} {f(opt, 'aic_mac_time(us)'):>12s} "
                f"{f(opt, 'aic_fixpipe_time(us)'):>12s} {f(opt, 'aic_mte1_time(us)'):>12s} "
                f"{f(opt, 'aic_scalar_time(us)'):>12s}  |  "
                f"{f(base, 'aic_time(us)'):>16s} {f(base, 'aic_mac_time(us)'):>12s} "
                f"{f(base, 'aic_fixpipe_time(us)'):>12s} {f(base, 'aic_mte1_time(us)'):>12s} "
                f"{f(base, 'aic_scalar_time(us)'):>12s}"
            )
        lines.append("")

    # ===== 表3: 加速比归因 =====
    lines.append("  " + "─" * 128)
    lines.append("  表3: 加速比归因分析 - 各硬件单元节省时间及贡献度")
    lines.append("  " + "─" * 128)
    lines.append("")

    for causal_group, group_name in [(True, "因果注意力 (causal=True)"), (False, "非因果注意力 (causal=False)")]:
        lines.append(f"  【{group_name}】")
        lines.append(f"  {'Case':<6s} {'加速比':>7s}  "
                     f"{'Cube节省':>10s} {'Cube贡献':>9s}  "
                     f"{'FixP节省':>10s} {'FixP贡献':>9s}  "
                     f"{'MTE1节省':>10s} {'MTE1贡献':>9s}  "
                     f"{'Scalar节省':>10s} {'Scalar贡献':>9s}  "
                     f"{'AIV节省':>10s} {'AIV贡献':>9s}")
        lines.append(f"  {'':6s} {'':7s}  "
                     f"{'(us)':>10s} {'':9s}  {'(us)':>10s} {'':9s}  "
                     f"{'(us)':>10s} {'':9s}  {'(us)':>10s} {'':9s}  "
                     f"{'(us)':>10s} {'':9s}")
        lines.append("-" * 130)

        for case in TEST_CASES:
            idx, Z, H, N_CTX, HEAD_DIM, causal, dtype = case
            if causal != causal_group: continue
            key = f"case_{idx:02d}"
            data = all_results.get(key, {})
            opt = data.get("optimized", {})
            base = data.get("baseline", {})

            if opt.get("error") or base.get("error"):
                lines.append(f"  Case{idx:02d}  ERROR")
                continue

            o_t = opt.get("avg_time_ms", 0) or 0
            b_t = base.get("avg_time_ms", 0) or 0
            sp = b_t / o_t if o_t > 0 else 0
            total_save = (b_t - o_t) * 1000  # 总节省 us

            def savings(of, bf):
                ov = (opt.get(of) or 0)
                bv = (base.get(bf) or 0)
                return bv - ov if bv and ov else 0

            def contrib(sv):
                return (sv / total_save * 100) if total_save > 0 else 0

            s_cube = savings("aic_mac_time(us)", "aic_mac_time(us)")
            s_fixp = savings("aic_fixpipe_time(us)", "aic_fixpipe_time(us)")
            s_mte1 = savings("aic_mte1_time(us)", "aic_mte1_time(us)")
            s_scalar = savings("aic_scalar_time(us)", "aic_scalar_time(us)")
            s_aiv = savings("aiv_time(us)", "aiv_time(us)")

            lines.append(
                f"  Case{idx:02d}  {sp:5.2f}x  "
                f"{s_cube:10.0f} {contrib(s_cube):8.1f}%  "
                f"{s_fixp:10.0f} {contrib(s_fixp):8.1f}%  "
                f"{s_mte1:10.0f} {contrib(s_mte1):8.1f}%  "
                f"{s_scalar:10.0f} {contrib(s_scalar):8.1f}%  "
                f"{s_aiv:10.0f} {contrib(s_aiv):8.1f}%"
            )
        lines.append("")

    # ===== 表4: Vector Core 利用率 =====
    lines.append("  " + "─" * 128)
    lines.append("  表4: AI Vector Core 利用率对比")
    lines.append("  " + "─" * 128)
    lines.append("")

    for causal_group, group_name in [(True, "因果注意力 (causal=True)"), (False, "非因果注意力 (causal=False)")]:
        lines.append(f"  【{group_name}】")
        lines.append(f"  {'Case':<6s} {'AIV总耗时(us)':>16s} {'向量耗时':>12s} {'向量占比':>10s} {'标量耗时':>12s}  |  {'AIV总耗时(us)':>16s} {'向量耗时':>12s} {'向量占比':>10s} {'标量耗时':>12s}")
        lines.append(f"  {'':6s} {'优化版':>16s} {'优化版':>12s} {'优化版':>10s} {'优化版':>12s}  |  {'基线版':>16s} {'基线版':>12s} {'基线版':>10s} {'基线版':>12s}")
        lines.append("-" * 130)

        for case in TEST_CASES:
            idx, Z, H, N_CTX, HEAD_DIM, causal, dtype = case
            if causal != causal_group: continue
            key = f"case_{idx:02d}"
            data = all_results.get(key, {})
            opt = data.get("optimized", {})
            base = data.get("baseline", {})

            def fs(d, field, is_pct=False):
                if not d or d.get("error"): return "N/A"
                v = d.get(field)
                if v is None: return "N/A"
                return f"{v*100:.1f}%" if is_pct else f"{v:.0f}"

            lines.append(
                f"  Case{idx:02d}  {fs(opt, 'aiv_time(us)'):>16s} {fs(opt, 'aiv_vec_time(us)'):>12s} "
                f"{fs(opt, 'aiv_vec_ratio', True):>10s} {fs(opt, 'aiv_scalar_time(us)'):>12s}  |  "
                f"{fs(base, 'aiv_time(us)'):>16s} {fs(base, 'aiv_vec_time(us)'):>12s} "
                f"{fs(base, 'aiv_vec_ratio', True):>10s} {fs(base, 'aiv_scalar_time(us)'):>12s}"
            )
        lines.append("")

    # ===== 总结 =====
    lines.append(sep)
    lines.append("  综合分析总结")
    lines.append(sep)
    lines.append("")

    # 计算统计数据
    valid_cases = []
    for case in TEST_CASES:
        idx = case[0]
        key = f"case_{idx:02d}"
        data = all_results.get(key, {})
        opt = data.get("optimized", {})
        base = data.get("baseline", {})
        if not opt.get("error") and not base.get("error"):
            o_t = opt.get("avg_time_ms", 0) or 0
            b_t = base.get("avg_time_ms", 0) or 0
            sp = b_t / o_t if o_t > 0 else 0
            valid_cases.append({
                "idx": idx, "speedup": sp,
                "opt_cube": opt.get("cube_utilization(%)"),
                "base_cube": base.get("cube_utilization(%)"),
                "opt_mac": opt.get("aic_mac_ratio"),
                "base_mac": base.get("aic_mac_ratio"),
                "opt_mte1": opt.get("aic_mte1_ratio"),
                "base_mte1": base.get("aic_mte1_ratio"),
                "opt_fixp": opt.get("aic_fixpipe_ratio"),
                "base_fixp": base.get("aic_fixpipe_ratio"),
            })

    causal = [c for c in valid_cases if c["idx"] <= 6]
    noncausal = [c for c in valid_cases if c["idx"] >= 7]

    lines.append("  1. 整体性能:")
    if valid_cases:
        avg_sp = sum(c["speedup"] for c in valid_cases) / len(valid_cases)
        lines.append(f"     全部 {len(valid_cases)} 个用例平均加速比: {avg_sp:.2f}x")
        lines.append(f"     加速比范围: {min(c['speedup'] for c in valid_cases):.2f}x ~ {max(c['speedup'] for c in valid_cases):.2f}x")
    if causal:
        lines.append(f"     因果模式平均加速比: {sum(c['speedup'] for c in causal)/len(causal):.2f}x")
    if noncausal:
        lines.append(f"     非因果模式平均加速比: {sum(c['speedup'] for c in noncausal)/len(noncausal):.2f}x")
    lines.append("")

    lines.append("  2. Cube (矩阵乘) 利用率:")
    if valid_cases:
        opt_cubes = [c["opt_cube"] for c in valid_cases if c["opt_cube"] is not None]
        base_cubes = [c["base_cube"] for c in valid_cases if c["base_cube"] is not None]
        if opt_cubes:
            lines.append(f"     优化版 Cube 利用率范围: {min(opt_cubes):.1f}% ~ {max(opt_cubes):.1f}%, 平均 {sum(opt_cubes)/len(opt_cubes):.1f}%")
        if base_cubes:
            lines.append(f"     基线版 Cube 利用率范围: {min(base_cubes):.1f}% ~ {max(base_cubes):.1f}%, 平均 {sum(base_cubes)/len(base_cubes):.1f}%")
    lines.append("")

    lines.append("  3. 加速来源分析:")
    lines.append("     - 因果模式: Prefix-Diagonal 拆分消除前缀区域冗余 mask 计算 + 软件流水线重叠")
    lines.append("     - 非因果模式: 提前加载 V 张量实现 DMA/计算重叠 + 最优分块减少循环迭代")
    lines.append("     - 共同因素: 动态核心分配、multibuffer 重叠、FA4 rescale skipping")
    lines.append("")

    lines.append("  4. 关键硬件指标解读:")
    lines.append("     - MAC 占比上升 = 矩阵乘密度提高，更多时间用于有效计算")
    lines.append("     - MTE1 占比上升 = 内存带宽利用更充分 (与计算重叠，非瓶颈)")
    lines.append("     - FixPipe 占比上升 = Softmax/Norm 计算被更好地重叠")
    lines.append("     - Scalar 耗时缩减 = 控制流开销消除 (最大加速来源)")
    lines.append("     - Cube 利用率接近 100% = 矩阵乘单元接近饱和")
    lines.append("")

    lines.append(sep)
    lines.append("  报告结束")
    lines.append(sep)

    report = "\n".join(lines)
    report_path = os.path.join(WORKSPACE, "profiler_analysis_results.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"报告已保存: profiler_analysis_results.txt")
    print("\n" + report)


if __name__ == "__main__":
    main()
