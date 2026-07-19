"""
ASCEND-RSFA: FlashAttention v2 with Ascend-specific optimizations.
- K/V load-advance software pipeline
- Dynamic AI core selection with persistent scheduling
- Compile pipeline configuration
- Causal region split (Prefix/Diagonal dual-kernel)
"""

import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver

DEVICE = "npu"


def get_num_compute_cores():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)["num_aicore"]


# ---------- Tiling configuration ----------
# ---------- Small sequence kernel (N_CTX <= 128) ----------
@triton.jit
def _attn_fwd_small(Q, K, V, Out, sm_scale,
                    stride_qz: tl.constexpr, stride_qh: tl.constexpr,
                    stride_qm: tl.constexpr, stride_qk: tl.constexpr,
                    stride_kz: tl.constexpr, stride_kh: tl.constexpr,
                    stride_kn: tl.constexpr, stride_kk: tl.constexpr,
                    stride_vz: tl.constexpr, stride_vh: tl.constexpr,
                    stride_vn: tl.constexpr, stride_vk: tl.constexpr,
                    stride_oz: tl.constexpr, stride_oh: tl.constexpr,
                    stride_om: tl.constexpr, stride_on: tl.constexpr,
                    H: tl.constexpr,
                    N_CTX: tl.constexpr,
                    HEAD_DIM: tl.constexpr):
    task_hz_idx = tl.program_id(0)
    off_z = task_hz_idx // H
    off_h = task_hz_idx % H

    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(N_CTX, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(N_CTX, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(N_CTX, HEAD_DIM),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(0, 0),
        block_shape=(N_CTX, HEAD_DIM),
        order=(1, 0),
    )

    q = tl.load(Q_block_ptr)
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    qk = tl.dot(q, tl.trans(k)) * sm_scale
    qk = qk - tl.max(qk, 1)[:, None]
    p = tl.math.exp(qk)
    p = p / tl.sum(p, 1)[:, None]
    out = tl.dot(p.to(v.dtype), v)
    tl.store(O_block_ptr, out.to(Out.type.element_ty))


@triton.jit
def _attn_fwd_small_causal(Q, K, V, Out, sm_scale,
                           stride_qz: tl.constexpr, stride_qh: tl.constexpr,
                           stride_qm: tl.constexpr, stride_qk: tl.constexpr,
                           stride_kz: tl.constexpr, stride_kh: tl.constexpr,
                           stride_kn: tl.constexpr, stride_kk: tl.constexpr,
                           stride_vz: tl.constexpr, stride_vh: tl.constexpr,
                           stride_vn: tl.constexpr, stride_vk: tl.constexpr,
                           stride_oz: tl.constexpr, stride_oh: tl.constexpr,
                           stride_om: tl.constexpr, stride_on: tl.constexpr,
                           H: tl.constexpr,
                           N_CTX: tl.constexpr,
                           HEAD_DIM: tl.constexpr):
    task_hz_idx = tl.program_id(0)
    off_z = task_hz_idx // H
    off_h = task_hz_idx % H

    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(N_CTX, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(N_CTX, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(N_CTX, HEAD_DIM),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(0, 0),
        block_shape=(N_CTX, HEAD_DIM),
        order=(1, 0),
    )

    q = tl.load(Q_block_ptr)
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    qk = tl.dot(q, tl.trans(k)) * sm_scale
    offs_m = tl.arange(0, N_CTX)
    offs_n = tl.arange(0, N_CTX)
    mask = offs_m[:, None] >= offs_n[None, :]
    qk = qk + tl.where(mask, 0, -1.0e6)
    qk = qk - tl.max(qk, 1)[:, None]
    p = tl.math.exp(qk)
    p = p / tl.sum(p, 1)[:, None]
    out = tl.dot(p.to(v.dtype), v)
    tl.store(O_block_ptr, out.to(Out.type.element_ty))


# ---------- 内层循环（软件流水：load 后立即 advance）----------
@triton.jit
def _attn_fwd_inner(
    acc_ptr, l_i, m_i, q,
    K_block_ptr, V_block_ptr,
    start_m, qk_scale,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
    N_CTX: tl.constexpr, fp8_v: tl.constexpr,
):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        hi = tl.minimum(hi, N_CTX)
    else:
        lo, hi = 0, N_CTX

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        qk = tl.dot(q, k)
        if HEAD_DIM >= 128 and STAGE != 2:
            v = tl.load(V_block_ptr)
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            qk = qk * qk_scale
            m_ij = tl.maximum(
                m_i, tl.max(qk, 1, propagate_nan=tl.PropagateNan.ALL),
                propagate_nan=tl.PropagateNan.ALL,
            )
            qk = qk - m_ij[:, None]

        p = tl.math.exp(qk)

        if fp8_v:
            p_cast = p.to(tl.float8e5)
        else:
            p_cast = p.to(k.dtype)

        if HEAD_DIM < 128 or STAGE == 2:
            v = tl.load(V_block_ptr)
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        acc_ptr = acc_ptr * alpha[:, None]
        pv = tl.dot(p_cast, v)
        acc_ptr = acc_ptr + pv
        m_i = m_ij

    return acc_ptr, l_i, m_i


@triton.jit
def _attn_fwd_inner_d64_nc(
    acc_ptr, l_i, m_i, q,
    K_block_ptr, V_block_ptr,
    qk_scale,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    N_CTX: tl.constexpr, fp8_v: tl.constexpr,
):
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        qk = tl.dot(q, k)
        v = tl.load(V_block_ptr)
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

        qk = qk * qk_scale
        m_ij = tl.maximum(
            m_i, tl.max(qk, 1, propagate_nan=tl.PropagateNan.ALL),
            propagate_nan=tl.PropagateNan.ALL,
        )
        qk = qk - m_ij[:, None]
        p = tl.math.exp(qk)

        if fp8_v:
            p_cast = p.to(tl.float8e5)
        else:
            p_cast = p.to(k.dtype)

        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        acc_ptr = acc_ptr * alpha[:, None]
        pv = tl.dot(p_cast, v)
        acc_ptr = acc_ptr + pv
        m_i = m_ij

    return acc_ptr, l_i, m_i


# ---------- 主 kernel（pipeline 路径 + split state 复用同一 _attn_fwd）----------
@triton.jit
def _attn_fwd(
    Q, K, V, M, Out, sm_scale,
    stride_qz: tl.constexpr, stride_qh: tl.constexpr,
    stride_qm: tl.constexpr, stride_qk: tl.constexpr,
    stride_kz: tl.constexpr, stride_kh: tl.constexpr,
    stride_kn: tl.constexpr, stride_kk: tl.constexpr,
    stride_vz: tl.constexpr, stride_vh: tl.constexpr,
    stride_vn: tl.constexpr, stride_vk: tl.constexpr,
    stride_oz: tl.constexpr, stride_oh: tl.constexpr,
    stride_om: tl.constexpr, stride_on: tl.constexpr,
    Z: tl.constexpr, H: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    NUM_BLOCKS_M = tl.cdiv(N_CTX, BLOCK_M)
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset,
            shape=(HEAD_DIM, N_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, 0),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(0, 1),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        q = tl.load(Q_block_ptr)

        if STAGE == 4:
            m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
            l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
            acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
            acc, l_i, m_i = _attn_fwd_split_inner(
                acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                task_m_idx, sm_scale,
                BLOCK_M, HEAD_DIM, BLOCK_N,
                1, offs_m, offs_n, N_CTX,
                V.dtype.element_ty == tl.float8e5
            )
            state_size = Z * H * NUM_BLOCKS_M * BLOCK_M
            state_ptrs = M + task_hz_idx * NUM_BLOCKS_M * BLOCK_M + task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
            tl.store(state_ptrs, l_i)
            tl.store(state_ptrs + state_size, m_i)
            tl.store(O_block_ptr, acc.to(Out.type.element_ty))
        elif STAGE == 5:
            state_size = Z * H * NUM_BLOCKS_M * BLOCK_M
            state_ptrs = M + task_hz_idx * NUM_BLOCKS_M * BLOCK_M + task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
            l_i = tl.load(state_ptrs)
            m_i = tl.load(state_ptrs + state_size)
            acc = tl.load(O_block_ptr).to(tl.float32)
            acc, l_i, m_i = _attn_fwd_split_inner(
                acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                task_m_idx, sm_scale,
                BLOCK_M, HEAD_DIM, BLOCK_N,
                2, offs_m, offs_n, N_CTX,
                V.dtype.element_ty == tl.float8e5
            )
            accumulator = acc / l_i[:, None]
            tl.store(O_block_ptr, accumulator.to(Out.type.element_ty))
        else:
            m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
            l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
            acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

            if STAGE & 1:
                if STAGE == 1 and HEAD_DIM == 64:
                    acc, l_i, m_i = _attn_fwd_inner_d64_nc(
                        acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                        sm_scale,
                        BLOCK_M, HEAD_DIM, BLOCK_N,
                        N_CTX,
                        V.dtype.element_ty == tl.float8e5
                    )
                else:
                    acc, l_i, m_i = _attn_fwd_inner(
                        acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                        task_m_idx, sm_scale,
                        BLOCK_M, HEAD_DIM, BLOCK_N,
                        4 - STAGE, offs_m, offs_n, N_CTX,
                        V.dtype.element_ty == tl.float8e5
                    )
            if STAGE & 2:
                acc, l_i, m_i = _attn_fwd_inner(
                    acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                    task_m_idx, sm_scale,
                    BLOCK_M, HEAD_DIM, BLOCK_N,
                    2, offs_m, offs_n, N_CTX,
                    V.dtype.element_ty == tl.float8e5
                )

            m_i += tl.math.log(l_i)
            accumulator = acc / l_i[:, None]
            m_ptrs = M + task_hz_idx * N_CTX + offs_m
            tl.store(m_ptrs, m_i)
            tl.store(O_block_ptr, accumulator.to(Out.type.element_ty))


# ---------- Causal split kernel（平台计分形状走师妹版 prefix/diag）----------
@triton.jit
def _attn_fwd_split_inner(
    acc_ptr, l_i, m_i, q,
    K_block_ptr, V_block_ptr,
    start_m, qk_scale,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
    N_CTX: tl.constexpr, fp8_v: tl.constexpr,
):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        hi = tl.minimum(hi, N_CTX)
    else:
        lo, hi = 0, N_CTX

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        qk = tl.dot(q, k)
        # 建议5: 始终早 load V（无 HEAD_DIM 判断），DMA 与 QK matmul + softmax 重叠
        v = tl.load(V_block_ptr)
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            qk = qk * qk_scale
            m_ij = tl.maximum(
                m_i, tl.max(qk, 1, propagate_nan=tl.PropagateNan.ALL),
                propagate_nan=tl.PropagateNan.ALL,
            )
            qk = qk - m_ij[:, None]

        p = tl.math.exp(qk)

        if fp8_v:
            p_cast = p.to(tl.float8e5)
        else:
            p_cast = p.to(k.dtype)

        # l_ij = tl.sum(p, 1)
        # alpha = tl.math.exp(m_i - m_ij)
        # l_i = l_i * alpha + l_ij

        # acc_ptr = acc_ptr * alpha[:, None]
        # acc_ptr = tl.dot(p_cast, v, acc_ptr)
        # m_i = m_ij

        # ---------- FA4 rescale skipping (threshold = 0.0) ----------
        l_ij = tl.sum(p, 1)
        delta = m_ij - m_i
        need_rescale = delta > 0.0
        alpha = tl.where(
            need_rescale,
            tl.math.exp(-delta),
            1.0
        )
        l_i = tl.where(
            need_rescale,
            l_i * alpha,
            l_i
        ) + l_ij

        acc_ptr = tl.where(
            need_rescale[:, None],
            acc_ptr * alpha[:, None],
            acc_ptr
        )
        acc_ptr = tl.dot(p_cast, v, acc_ptr)
        m_i = m_ij

    return acc_ptr, l_i, m_i



def _run_causal_split(q, k, v, o, sm_scale, BM, BN, Z, H, N_CTX, HEAD_DIM):
    max_cores = get_num_compute_cores()
    grid = lambda meta: (
        min(max_cores, Z * H * triton.cdiv(N_CTX, meta["BLOCK_M"])),
    )
    bn_diag = BN
    bn_prefix = BM if BM > BN else BN
    num_blocks_m = triton.cdiv(N_CTX, BM)
    state = torch.empty((2, Z, H, num_blocks_m, BM), dtype=torch.float32, device=q.device)
    launch_kwargs = {}
    launch_kwargs.update({
        "unit_flag": True,
        "limit_auto_multi_buffer_only_for_local_buffer": False,
        "set_workspace_multibuffer": 4,
        "tile_mix_vector_loop": 4,
        "tile_mix_cube_loop": 2,
    })

    _attn_fwd[grid](
        q, k, v, state, o, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z=Z, H=H, N_CTX=N_CTX, HEAD_DIM=HEAD_DIM,
        BLOCK_M=BM,
        BLOCK_N=bn_prefix,
        STAGE=4,
        multibuffer=True,
        num_warps=4,
        num_stages=2,
        **launch_kwargs,
    )

    _attn_fwd[grid](
        q, k, v, state, o, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z=Z, H=H, N_CTX=N_CTX, HEAD_DIM=HEAD_DIM,
        BLOCK_M=BM,
        BLOCK_N=bn_diag,
        STAGE=5,
        multibuffer=True,
        num_warps=4,
        num_stages=2,
        **launch_kwargs,
    )


# ============================================================================
# IMPORTANT -- DO NOT CHANGE CLASS NAME: _attention
# ============================================================================
class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, BM, BN):
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        Z, H, N_CTX, HEAD_DIM = q.shape
        use_causal_split = causal
        o = torch.empty_like(q)
        if use_causal_split:
            M = torch.empty((1,), dtype=torch.float32, device=q.device)
        else:
            M = torch.empty((Z, H, N_CTX), dtype=torch.float32, device=q.device)

        if not causal and N_CTX <= 128 and HEAD_DIM <= 128:
            _attn_fwd_small[(Z * H,)](
                q, k, v, o, sm_scale,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                H=H, N_CTX=N_CTX, HEAD_DIM=HEAD_DIM,
            )
        elif causal and N_CTX <= 64 and HEAD_DIM <= 128:
            _attn_fwd_small_causal[(Z * H,)](
                q, k, v, o, sm_scale,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                H=H, N_CTX=N_CTX, HEAD_DIM=HEAD_DIM,
            )
        elif use_causal_split:
            _run_causal_split(q, k, v, o, sm_scale, BM, BN, Z, H, N_CTX, HEAD_DIM)
        else:
            stage = 3 if causal else 1
            max_cores = get_num_compute_cores()
            grid = lambda meta: (
                min(max_cores, Z * H * triton.cdiv(N_CTX, meta["BLOCK_M"])),
            )
            launch_kwargs = {}
            launch_kwargs.update({
                "unit_flag": True,
                "multibuffer": True,
                "limit_auto_multi_buffer_only_for_local_buffer": False,
                "set_workspace_multibuffer": 4,
                "tile_mix_vector_loop": 4,
                "tile_mix_cube_loop": 2,
            })
            _attn_fwd[grid](
                q, k, v, M, o, sm_scale,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                Z=Z, H=H, N_CTX=N_CTX, HEAD_DIM=HEAD_DIM,
                BLOCK_M=BM,
                BLOCK_N=BN,
                STAGE=stage,
                **launch_kwargs,
            )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o


# ============================================================================
# IMPORTANT -- DO NOT CHANGE VARIABLE NAME: attention
# ============================================================================
attention = _attention.apply

def get_tiling(Z, H, N_CTX, HEAD_DIM, causal):
    """
    Return (BLOCK_M, BLOCK_N) tiling parameters for the given problem shape.
    """
    _DEFAULTS = {
        # (Z,    H,   N_CTX, HEAD_DIM, causal) : (BM,  BN)
        # --- correctness tests ---
        (1,   1,   64,   64,  False): (64,  16),
        (1,   1,   64,   64,  True):  (64,  16),
        (1,   1,   128,  128, False): (32,  128),
        (1,   1,   128,  128, True):  (64,  32),
        (1,   2,   256,  256, False): (32,  128),
        (2,   2,   128,  256, False): (64,  64),
        (1,   2,   1024, 64,  False): (128, 32),
        (1,   2,   1024, 64,  True):  (64,  32),
        (4,   32,  64,   64,  False): (64,  32),
        (4,   32,  64,   64,  True):  (64,  32),
        (4,   32,  128,  128, False): (32,  64),
        (4,   32,  256,  128, True):  (64,  32),
        (4,   32,  512,  64,  True):  (64,  64),
        (4,   32,  1024, 64,  False): (128, 256),
        (4,   32,  1024, 64,  True):  (64,  64),
        (4,   32,  1024, 128, False): (128, 128),
        (4,   32,  2048, 64,  True):  (64,  64),
        (4,   32,  2048, 128, False): (128, 128),
        (4,   32,  4096, 64,  False): (128, 256),
        (128, 8,   1024, 64,  False): (128, 128),
        # --- performance tests ---
        (128, 8,   1024, 128, True):  (128, 64),
        (128, 8,   1024, 256, True):  (64,  64),
        (128, 8,   2048, 128, True):  (128, 64),
        (128, 8,   2048, 256, True):  (64,  64),
        (128, 8,   4096, 128, True):  (128, 64),
        (128, 8,   8192, 64,  True):  (128, 64),
        (128, 8,   1024, 128, False): (128, 256),
        (128, 8,   1024, 256, False): (64,  256),
        (128, 8,   2048, 128, False): (128, 256),
        (128, 8,   2048, 256, False): (64,  256),
        (128, 8,   4096, 128, False): (128, 256),
        (128, 8,   8192, 64,  False): (128, 512),
    }
    return _DEFAULTS[(Z, H, N_CTX, HEAD_DIM, causal)]


@pytest.mark.parametrize(
    "Z, H, N_CTX, HEAD_DIM, causal, dtype, BM, BN",
    [
        (1, 2, 1024, 64, False, torch.float16, 64, 32),
        (1, 1, 64, 64, False, torch.float16, 64, 16),
        (4, 32, 64, 64, False, torch.float16, 64, 32),
        (4, 32, 64, 64, False, torch.float16, 64, 16),
        (1, 1, 128, 128, False, torch.float16, 16, 16),
        (1, 1, 128, 128, False, torch.float16, 64, 64),
        (4, 32, 128, 128, False, torch.float16, 32, 64),
        (1, 2, 1024, 64, False, torch.float16, 128, 32),

        (1, 1, 64, 64, True, torch.float16, 64, 16),
        (4, 32, 64, 64, True, torch.float16, 64, 32),
        (4, 32, 64, 64, True, torch.float16, 64, 16),
        (1, 1, 128, 128, True, torch.float16, 16, 16),
        (1, 1, 128, 128, True, torch.float16, 64, 64),
        (1, 2, 1024, 64, True, torch.float16, 64, 32),
        (1, 1, 128, 128, True, torch.float16, 32, 32),
        (1, 2, 1024, 64, True, torch.float16, 128, 32),

        (4, 32, 2048, 128, False, torch.float16, 128, 128),
        (4, 32, 4096, 64, False, torch.float16, 128, 128),
        (4, 32, 1024, 128, False, torch.float16, 128, 128),
        (4, 32, 1024, 64, False, torch.float16, 128, 128),
        (4, 32, 2048, 64, True, torch.float16, 64, 64),
        (4, 32, 1024, 64, True, torch.float16, 64, 64),
        (4, 32, 512, 64, True, torch.float16, 64, 64),
        (4, 32, 256, 128, True, torch.float16, 64, 64),

        (128, 8, 1024, 64, False, torch.bfloat16, 128, 128),
    ]
)

def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype, BM, BN):
    """Test your Triton kernel for correctness."""
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())

    sm_scale = 0.5

    compressed_len = 2048
    atten_mask = None
    sparse_mode = 0
    atten_golden_mask = None

    if causal:
        atten_mask = torch.triu(torch.ones(N_CTX, N_CTX, device=DEVICE), diagonal=1)
        atten_golden_mask = torch.triu(torch.ones(compressed_len, compressed_len, device=DEVICE), diagonal=1).bool()
        sparse_mode = 2

    if atten_mask is None:
        atten_mask = torch.zeros((1, 1), device=DEVICE)

    ref_out = torch_npu.npu_fusion_attention(
        q, k, v, H,
        padding_mask=None,
        atten_mask=atten_golden_mask,
        scale=sm_scale,
        keep_prob=1.0,
        input_layout='BNSD',
        pre_tockens=65535,
        next_tockens=65535,
        sparse_mode=sparse_mode,
        )[0]

    tri_out = attention(q, k, v, causal, sm_scale, BM, BN).to(dtype)
    rtol = 1e-2
    atol = 1e-2
    assert torch.allclose(ref_out, tri_out, atol=atol, rtol=rtol)
    print("function pass!")

if __name__ == "__main__":
    # Correctness test with fixed tiling
    test_op(128, 8, 1024, 128, causal=True, dtype=torch.float16, BM=128, BN=64)
    test_op(128, 8, 1024, 256, causal=True, dtype=torch.float16, BM=64, BN=64)
    test_op(128, 8, 2048, 128, causal=True, dtype=torch.float16, BM=128, BN=64)
    test_op(128, 8, 2048, 256, causal=True, dtype=torch.float16, BM=64, BN=64)

    test_op(128, 8, 4096, 128, causal=True, dtype=torch.float16, BM=128, BN=64)
    test_op(128, 8, 8192, 64, causal=True, dtype=torch.float16, BM=128, BN=64)
    #=================================================================================
    test_op(128, 8, 1024, 128, causal=False, dtype=torch.float16, BM=64, BN=128)
    test_op(128, 8, 1024, 256, causal=False, dtype=torch.float16, BM=64, BN=128)
    test_op(128, 8, 2048, 128, causal=False, dtype=torch.float16, BM=64, BN=128)
    test_op(128, 8, 2048, 256, causal=False, dtype=torch.float16, BM=64, BN=256)
    test_op(128, 8, 4096, 128, causal=False, dtype=torch.float16, BM=128, BN=128)
    test_op(128, 8, 8192, 64, causal=False, dtype=torch.float16, BM=128, BN=256)
