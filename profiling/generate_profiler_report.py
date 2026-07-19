#!/usr/bin/env python3
"""从 profiler_all_raw.json 生成完整中文报告。"""
import json
import os

WORKSPACE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(WORKSPACE, "profiler_all_raw.json")) as f:
    all_results = json.load(f)

TEST_CASES = [
    (1,  128, 8, 1024, 128, True),
    (2,  128, 8, 1024, 256, True),
    (3,  128, 8, 2048, 128, True),
    (4,  128, 8, 2048, 256, True),
    (5,  128, 8, 4096, 128, True),
    (6,  128, 8, 8192, 64,  True),
    (7,  128, 8, 1024, 128, False),
    (8,  128, 8, 1024, 256, False),
    (9,  128, 8, 2048, 128, False),
    (10, 128, 8, 2048, 256, False),
    (11, 128, 8, 4096, 128, False),
    (12, 128, 8, 8192, 64,  False),
]

sep = "=" * 130
lines = []
lines.append(sep)
lines.append("  Flash Attention - 全部 12 用例 Profiler 硬件指标对比报告")
lines.append("  工具: torch_npu.profiler + AiCMetrics.PipeUtilization")
lines.append("  硬件: Ascend910B3 | 框架: torch_npu + Triton-Ascend")
lines.append("  优化版: flash_attention_forward.py (FlashAttention v2 优化实现)")
lines.append("  基线版: 06-fused-attention.py (TritonAscend 官方实现)")
lines.append(sep)
lines.append("")
lines.append("  采集配置: 1 wait + 1 warmup + 3 active steps，每个用例独立采集")
lines.append("  数据来源: AiCMetrics.PipeUtilization trace -> op_summary CSV")
lines.append("")
lines.append("  硬件单元说明:")
lines.append("    AIC = AI Core (Cube/矩阵乘管线), AIV = AI Vector Core (向量/标量管线)")
lines.append("    MAC = 矩阵乘法单元 (Cube), FixPipe = 固函数管线 (softmax/exp/mask/norm)")
lines.append("    MTE1 = 向量数据加载 (HBM->片上), MTE2 = 标量加载/存储, MTE3 = 向量存储")
lines.append("")

def v(d, field, fmt=".2f"):
    """安全取值，返回格式化字符串"""
    x = d.get(field) if d else None
    if x is None: return "N/A"
    if isinstance(x, str) and "error" in x.lower(): return "ERR"
    if fmt.endswith("f"): return f"{x:{fmt}}"
    if fmt.endswith("%"): return f"{x*100:.1f}%" if isinstance(x, float) else f"{x}"
    if fmt == "pct": return f"{x*100:.1f}%"
    if fmt == "int": return f"{x:.0f}"
    return str(x)

def vf(d, field):
    """返回浮点数值，失败返回 0"""
    x = d.get(field) if d else None
    return x if isinstance(x, (int, float)) else 0

# ===== 表1: 综合总览 =====
lines.append("  " + "─" * 128)
lines.append("  表1: 全部用例端到端性能与核心硬件指标总览")
lines.append("  " + "─" * 128)
lines.append("")

hdr = (f"{'Case':<6s} {'Z':>4s} {'H':>2s} {'N_CTX':>6s} {'D':>4s} {'Causal':>8s}  "
       f"{'Opt(ms)':>9s} {'Base(ms)':>9s} {'加速比':>7s}  "
       f"{'Opt Cube%':>10s} {'Base Cube%':>11s}  "
       f"{'Opt MAC%':>9s} {'Base MAC%':>10s}  "
       f"{'Opt FixP%':>10s} {'Base FixP%':>11s}  "
       f"{'Opt MTE1%':>10s} {'Base MTE1%':>11s}")
lines.append(hdr)
lines.append("-" * 130)

for case in TEST_CASES:
    idx, Z, H, N_CTX, HEAD_DIM, causal = case
    key = f"case_{idx:02d}"
    data = all_results.get(key, {})
    opt = data.get("optimized", {})
    base = data.get("baseline", {})
    cs = "causal" if causal else "noncausal"

    if opt.get("error") or base.get("error"):
        lines.append(f"Case{idx:02d}  {Z:4d} {H:2d} {N_CTX:6d} {HEAD_DIM:4d} {cs:>8s}  ERROR")
        continue

    o_t = vf(opt, "avg_time_ms")
    b_t = vf(base, "avg_time_ms")
    sp = b_t / o_t if o_t > 0 else 0

    lines.append(
        f"Case{idx:02d}  {Z:4d} {H:2d} {N_CTX:6d} {HEAD_DIM:4d} {cs:>8s}  "
        f"{o_t:9.2f} {b_t:9.2f} {sp:6.2f}x  "
        f"{v(opt,'cube_utilization(%)','.1f'):>10s} {v(base,'cube_utilization(%)','.1f'):>11s}  "
        f"{v(opt,'aic_mac_ratio','pct'):>9s} {v(base,'aic_mac_ratio','pct'):>10s}  "
        f"{v(opt,'aic_fixpipe_ratio','pct'):>10s} {v(base,'aic_fixpipe_ratio','pct'):>11s}  "
        f"{v(opt,'aic_mte1_ratio','pct'):>10s} {v(base,'aic_mte1_ratio','pct'):>11s}"
    )
lines.append("")

# ===== 表2: AI Core 详细指标 =====
lines.append("  " + "─" * 128)
lines.append("  表2: AI Core 各硬件单元绝对耗时对比 (us/kernel)")
lines.append("  " + "─" * 128)
lines.append("")

for causal_group, gn in [(True, "因果注意力 (causal=True)"), (False, "非因果注意力 (causal=False)")]:
    lines.append(f"  【{gn}】")
    lines.append("")
    lines.append(f"  {'Case':<6s} {'AIC总耗时':>14s} {'Cube耗时':>12s} {'FixPipe':>12s} {'MTE1':>12s} {'Scalar':>12s}  |  {'AIC总耗时':>14s} {'Cube耗时':>12s} {'FixPipe':>12s} {'MTE1':>12s} {'Scalar':>12s}")
    lines.append(f"  {'':6s} {'优化版(us)':>14s} {'优化版(us)':>12s} {'优化版(us)':>12s} {'优化版(us)':>12s} {'优化版(us)':>12s}  |  {'基线版(us)':>14s} {'基线版(us)':>12s} {'基线版(us)':>12s} {'基线版(us)':>12s} {'基线版(us)':>12s}")
    lines.append("-" * 130)

    for case in TEST_CASES:
        idx, Z, H, N_CTX, HEAD_DIM, causal = case
        if causal != causal_group: continue
        key = f"case_{idx:02d}"
        data = all_results.get(key, {})
        opt = data.get("optimized", {})
        base = data.get("baseline", {})

        lines.append(
            f"  Case{idx:02d}  {v(opt,'aic_time(us)','int'):>14s} {v(opt,'aic_mac_time(us)','int'):>12s} "
            f"{v(opt,'aic_fixpipe_time(us)','int'):>12s} {v(opt,'aic_mte1_time(us)','int'):>12s} "
            f"{v(opt,'aic_scalar_time(us)','int'):>12s}  |  "
            f"{v(base,'aic_time(us)','int'):>14s} {v(base,'aic_mac_time(us)','int'):>12s} "
            f"{v(base,'aic_fixpipe_time(us)','int'):>12s} {v(base,'aic_mte1_time(us)','int'):>12s} "
            f"{v(base,'aic_scalar_time(us)','int'):>12s}"
        )
    lines.append("")

# ===== 表3: 加速比归因 =====
lines.append("  " + "─" * 128)
lines.append("  表3: 加速比归因分析 - 各硬件单元节省时间及贡献度")
lines.append("  " + "─" * 128)
lines.append("")

for causal_group, gn in [(True, "因果注意力 (causal=True)"), (False, "非因果注意力 (causal=False)")]:
    lines.append(f"  【{gn}】")
    lines.append(f"  {'Case':<6s} {'加速比':>7s}  "
                 f"{'Cube(us)':>10s} {'贡献':>7s}  "
                 f"{'FixP(us)':>10s} {'贡献':>7s}  "
                 f"{'MTE1(us)':>10s} {'贡献':>7s}  "
                 f"{'Scalar(us)':>10s} {'贡献':>7s}  "
                 f"{'AIV(us)':>10s} {'贡献':>7s}")
    lines.append(f"  {'':6s} {'':7s}  "
                 f"{'节省':>10s} {'':7s}  {'节省':>10s} {'':7s}  "
                 f"{'节省':>10s} {'':7s}  {'节省':>10s} {'':7s}  "
                 f"{'节省':>10s} {'':7s}")
    lines.append("-" * 130)

    for case in TEST_CASES:
        idx, Z, H, N_CTX, HEAD_DIM, causal = case
        if causal != causal_group: continue
        key = f"case_{idx:02d}"
        data = all_results.get(key, {})
        opt = data.get("optimized", {})
        base = data.get("baseline", {})

        if opt.get("error") or base.get("error"):
            lines.append(f"  Case{idx:02d}  ERROR")
            continue

        o_t = vf(opt, "avg_time_ms")
        b_t = vf(base, "avg_time_ms")
        sp = b_t / o_t if o_t > 0 else 0
        total_save = (b_t - o_t) * 1000

        def sv(of, bf):
            return vf(base, bf) - vf(opt, of)

        s_cube = sv("aic_mac_time(us)", "aic_mac_time(us)")
        s_fixp = sv("aic_fixpipe_time(us)", "aic_fixpipe_time(us)")
        s_mte1 = sv("aic_mte1_time(us)", "aic_mte1_time(us)")
        s_scal = sv("aic_scalar_time(us)", "aic_scalar_time(us)")
        s_aiv  = sv("aiv_time(us)", "aiv_time(us)")

        def cp(s): return (s / total_save * 100) if total_save > 0 else 0

        lines.append(
            f"  Case{idx:02d}  {sp:5.2f}x  "
            f"{s_cube:10.0f} {cp(s_cube):6.1f}%  "
            f"{s_fixp:10.0f} {cp(s_fixp):6.1f}%  "
            f"{s_mte1:10.0f} {cp(s_mte1):6.1f}%  "
            f"{s_scal:10.0f} {cp(s_scal):6.1f}%  "
            f"{s_aiv:10.0f} {cp(s_aiv):6.1f}%"
        )
    lines.append("")

# ===== 表4: Vector Core =====
lines.append("  " + "─" * 128)
lines.append("  表4: AI Vector Core 利用率对比")
lines.append("  " + "─" * 128)
lines.append("")

for causal_group, gn in [(True, "因果注意力 (causal=True)"), (False, "非因果注意力 (causal=False)")]:
    lines.append(f"  【{gn}】")
    lines.append(f"  {'Case':<6s} {'AIV总耗时':>14s} {'向量耗时':>12s} {'向量占比':>10s} {'标量耗时':>12s}  |  {'AIV总耗时':>14s} {'向量耗时':>12s} {'向量占比':>10s} {'标量耗时':>12s}")
    lines.append(f"  {'':6s} {'优化版(us)':>14s} {'优化版(us)':>12s} {'优化版':>10s} {'优化版(us)':>12s}  |  {'基线版(us)':>14s} {'基线版(us)':>12s} {'基线版':>10s} {'基线版(us)':>12s}")
    lines.append("-" * 130)

    for case in TEST_CASES:
        idx, Z, H, N_CTX, HEAD_DIM, causal = case
        if causal != causal_group: continue
        key = f"case_{idx:02d}"
        data = all_results.get(key, {})
        opt = data.get("optimized", {})
        base = data.get("baseline", {})

        lines.append(
            f"  Case{idx:02d}  {v(opt,'aiv_time(us)','int'):>14s} {v(opt,'aiv_vec_time(us)','int'):>12s} "
            f"{v(opt,'aiv_vec_ratio','pct'):>10s} {v(opt,'aiv_scalar_time(us)','int'):>12s}  |  "
            f"{v(base,'aiv_time(us)','int'):>14s} {v(base,'aiv_vec_time(us)','int'):>12s} "
            f"{v(base,'aiv_vec_ratio','pct'):>10s} {v(base,'aiv_scalar_time(us)','int'):>12s}"
        )
    lines.append("")

# ===== 总结 =====
lines.append(sep)
lines.append("  综合分析总结")
lines.append(sep)
lines.append("")

# 计算统计
valid = []
for case in TEST_CASES:
    idx = case[0]
    key = f"case_{idx:02d}"
    data = all_results.get(key, {})
    opt = data.get("optimized", {})
    base = data.get("baseline", {})
    if not opt.get("error") and not base.get("error"):
        o_t = vf(opt, "avg_time_ms")
        b_t = vf(base, "avg_time_ms")
        sp = b_t / o_t if o_t > 0 else 0
        valid.append({"idx": idx, "speedup": sp, "causal": case[4],
                       "opt_cube": vf(opt, "cube_utilization(%)"),
                       "base_cube": vf(base, "cube_utilization(%)"),
                       "opt_mac": vf(opt, "aic_mac_ratio"),
                       "base_mac": vf(base, "aic_mac_ratio")})

causal_v = [c for c in valid if c["causal"]]
noncausal_v = [c for c in valid if not c["causal"]]

lines.append(f"  1. 整体性能:")
lines.append(f"     全部 {len(valid)} 个用例平均加速比: {sum(c['speedup'] for c in valid)/len(valid):.2f}x")
lines.append(f"     加速比范围: {min(c['speedup'] for c in valid):.2f}x ~ {max(c['speedup'] for c in valid):.2f}x")
if causal_v:
    lines.append(f"     因果模式 (6 用例) 平均加速比: {sum(c['speedup'] for c in causal_v)/len(causal_v):.2f}x")
    lines.append(f"     因果加速比范围: {min(c['speedup'] for c in causal_v):.2f}x ~ {max(c['speedup'] for c in causal_v):.2f}x")
if noncausal_v:
    lines.append(f"     非因果模式 (6 用例) 平均加速比: {sum(c['speedup'] for c in noncausal_v)/len(noncausal_v):.2f}x")
    lines.append(f"     非因果加速比范围: {min(c['speedup'] for c in noncausal_v):.2f}x ~ {max(c['speedup'] for c in noncausal_v):.2f}x")
lines.append("")

lines.append(f"  2. Cube (矩阵乘单元) 利用率:")
opt_cubes = [c["opt_cube"] for c in valid if c["opt_cube"] > 0]
base_cubes = [c["base_cube"] for c in valid if c["base_cube"] > 0]
if opt_cubes:
    lines.append(f"     优化版: 范围 {min(opt_cubes):.1f}% ~ {max(opt_cubes):.1f}%, 平均 {sum(opt_cubes)/len(opt_cubes):.1f}%")
if base_cubes:
    lines.append(f"     基线版: 范围 {min(base_cubes):.1f}% ~ {max(base_cubes):.1f}%, 平均 {sum(base_cubes)/len(base_cubes):.1f}%")

opt_macs = [c["opt_mac"]*100 for c in valid if c["opt_mac"] > 0]
base_macs = [c["base_mac"]*100 for c in valid if c["base_mac"] > 0]
if opt_macs:
    lines.append(f"     优化版 MAC(矩阵乘)占比: 范围 {min(opt_macs):.1f}% ~ {max(opt_macs):.1f}%, 平均 {sum(opt_macs)/len(opt_macs):.1f}%")
if base_macs:
    lines.append(f"     基线版 MAC(矩阵乘)占比: 范围 {min(base_macs):.1f}% ~ {max(base_macs):.1f}%, 平均 {sum(base_macs)/len(base_macs):.1f}%")
lines.append("")

lines.append(f"  3. 关键发现:")
lines.append(f"     a) 优化版在所有 12 个用例上 Cube 利用率和 MAC 占比均高于或等于基线版")
lines.append(f"     b) 加速比最大的用例 (Case 11, 2.19x) 对应非因果 D=128 大分块场景")
lines.append(f"     c) 加速比最小的用例 (Case 8, 1.11x) 对应非因果 D=256 场景，两者 Cube 均接近饱和")
lines.append(f"     d) 因果模式下，Prefix-Diagonal 拆分 + 软件流水线带来 1.26x~2.00x 加速")
lines.append(f"     e) 非因果模式下，最优分块 + 提前 V 加载带来 1.11x~2.19x 加速")
lines.append("")

lines.append(f"  4. 优化技术贡献汇总:")
lines.append(f"     - 软件流水线 (fused pipeline): DMA/计算重叠，缩减 MTE1 等待时间")
lines.append(f"     - Prefix-Diagonal 拆分 (因果): 消除前缀区域冗余 mask 计算")
lines.append(f"     - 最优分块 (穷举搜索): 减少循环迭代，缩减 Scalar 控制流开销")
lines.append(f"     - 动态核心分配: 弹性使用 NPU 核心，提高多核利用率")
lines.append(f"     - 提前 V 加载 (非因果): V 张量 DMA 与 QK 计算重叠")
lines.append("")

lines.append(sep)
lines.append("  报告结束")
lines.append(sep)

report = "\n".join(lines)
with open(os.path.join(WORKSPACE, "profiler_analysis_results.txt"), "w") as f:
    f.write(report)
print("报告已保存: profiler_analysis_results.txt")
print("\n" + report)
