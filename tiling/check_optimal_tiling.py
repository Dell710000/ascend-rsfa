#!/usr/bin/env python3
"""检查所有消融用例是否使用了最优分块，对有疑问的用例进行快速验证。"""

import importlib.util, os, sys, time
import torch, torch_npu

DEVICE = "npu"
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NW, NI = 3, 10  # 快速验证用较少迭代

def load_kernel(p):
    n = os.path.splitext(os.path.basename(p))[0].replace(" ","_").replace("-","_").replace("(","").replace(")","")
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    sys.modules[n] = m
    s.loader.exec_module(m)
    return m

def bench(kp, Z,H,N,D,causal,dtype,BM,BN):
    mod = load_kernel(kp)
    fn = mod.attention
    torch.manual_seed(42)
    q = torch.empty((Z,H,N,D), dtype=dtype, device=DEVICE).normal_(0,0.5)
    k = torch.empty((Z,H,N,D), dtype=dtype, device=DEVICE).normal_(0,0.5)
    v = torch.empty((Z,H,N,D), dtype=dtype, device=DEVICE).normal_(0,0.5)
    sm = 0.5
    try:
        for _ in range(NW): fn(q,k,v,causal,sm,BM,BN); torch.npu.synchronize()
        ts=[]
        for _ in range(NI):
            torch.npu.synchronize(); t0=time.perf_counter()
            fn(q,k,v,causal,sm,BM,BN); torch.npu.synchronize()
            ts.append((time.perf_counter()-t0)*1000)
        return sum(ts)/len(ts)
    except Exception as e:
        return f"ERR:{e}"

baseline = os.path.join(WORKSPACE, "baseline", "06-fused-attention.py")
fused_nc = os.path.join(WORKSPACE, "ablation", "04_v3_pipeline_optimized.py")
fused_sp = os.path.join(WORKSPACE, "ablation", "05_ascend_rsfa.py")
dt = torch.float16

checks = []

# === 非因果 Baseline ===
# nc_1024_256: 当前(64,128) vs 候选(64,256)
checks.append(("Baseline nc_1024_256", baseline, 128,8,1024,256,False,dt,
               [(64,128,"当前"), (64,256,"候选最优")]))

# nc_8192_64: 当前(128,256) vs 候选(64,512)
checks.append(("Baseline nc_8192_64", baseline, 128,8,8192,64,False,dt,
               [(128,256,"当前"), (64,512,"候选最优")]))

# === 因果 Split ===
# c_1024_256: 当前(64,64) vs 候选(64,256)
checks.append(("Split c_1024_256", fused_sp, 128,8,1024,256,True,dt,
               [(64,64,"当前"), (64,256,"候选最优")]))

# c_2048_256: 当前(64,64) vs 候选(64,256)
checks.append(("Split c_2048_256", fused_sp, 128,8,2048,256,True,dt,
               [(64,64,"当前"), (64,256,"候选最优")]))

# c_4096_128: 当前(128,64) vs 候选(128,32)
checks.append(("Split c_4096_128", fused_sp, 128,8,4096,128,True,dt,
               [(128,64,"当前"), (128,32,"候选最优")]))

# c_8192_64: 当前(128,64) vs 候选(64,256)
checks.append(("Split c_8192_64", fused_sp, 128,8,8192,64,True,dt,
               [(128,64,"当前"), (64,256,"候选最优")]))

print("="*90)
print("  最优分块验证")
print("="*90)

for label, kp, Z,H,N,D,causal,dtype, configs in checks:
    print(f"\n--- {label} (Z={Z},H={H},N={N},D={D}) ---")
    best_t, best_cfg = float('inf'), None
    for BM, BN, note in configs:
        r = bench(kp, Z,H,N,D,causal,dtype,BM,BN)
        if isinstance(r, str):
            print(f"  BM={BM:3d} BN={BN:3d} ({note}): {r}")
        else:
            print(f"  BM={BM:3d} BN={BN:3d} ({note}): {r:.2f} ms")
            if r < best_t:
                best_t, best_cfg = r, (BM, BN)
    print(f"  => 最优: BM={best_cfg[0]}, BN={best_cfg[1]} ({best_t:.2f} ms)")

print("\n" + "="*90)
print("  完成")
print("="*90)
