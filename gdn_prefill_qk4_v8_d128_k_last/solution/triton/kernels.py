"""
GDN Prefill - Self-contained FLA chunked delta rule (all 6 sub-kernels inlined).
Original code: flash-linear-attention (fla) by Songlin Yang, Yu Zhang.
Inlined here for standalone optimization via cuda-evolve.

Pipeline: precompute -> chunk_local_cumsum -> chunk_scaled_dot_kkt ->
          solve_tril -> recompute_w_u -> chunk_fwd_h -> chunk_fwd_o
"""

import inspect
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

KERNEL_TYPE = "gdn_prefill"

# ── Hardware detection ────────────────────────────────────────────────────────

_IS_NVIDIA = torch.cuda.is_available() and "NVIDIA" in torch.cuda.get_device_name(0)
_CAP = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
_IS_HOPPER_PLUS = _IS_NVIDIA and _CAP[0] >= 9
_IS_BLACKWELL = _IS_NVIDIA and _CAP[0] == 10

def _check_shared_mem(arch='none'):
    try:
        smem = torch.cuda.get_device_properties(0).max_shared_memory_per_block_optin
        thresholds = {'none': 0, 'ampere': 166912, 'ada': 166912, 'hopper': 232448}
        return smem >= thresholds.get(arch, 0)
    except Exception:
        return True

_HAS_LARGE_SMEM = _check_shared_mem()
_HAS_ADA_SMEM = _check_shared_mem('ada')
_HAS_AMPERE_SMEM = _check_shared_mem('ampere')
_HAS_HOPPER_SMEM = _check_shared_mem('hopper')

_at_sig = inspect.signature(triton.autotune)
_AT_KW = {"cache_results": True} if "cache_results" in _at_sig.parameters else {}
_AT_KW_CG = {**_AT_KW}
if "use_cuda_graph" in _at_sig.parameters:
    _AT_KW_CG["use_cuda_graph"] = False

# ── Triton JIT helpers ────────────────────────────────────────────────────────

@triton.jit
def _exp(x):
    return tl.exp(x.to(tl.float32))

@triton.jit
def _exp2(x):
    return tl.math.exp2(x.to(tl.float32))

# ── Python utilities ──────────────────────────────────────────────────────────

def _prepare_chunk_indices(cu_seqlens, chunk_size):
    lens = torch.diff(cu_seqlens)
    num_chunks = triton.cdiv(lens, chunk_size)
    indices = torch.cat([torch.arange(n, device=cu_seqlens.device)
                         for n in num_chunks.tolist()])
    seq_ids = indices.eq(0).cumsum(0) - 1
    return torch.stack([seq_ids, indices], 1).to(cu_seqlens)

def _prepare_chunk_offsets(cu_seqlens, chunk_size):
    lens = torch.diff(cu_seqlens)
    return F.pad(triton.cdiv(lens, chunk_size), (1, 0), value=0).cumsum(-1)

# ═════════════════════════════════════════════════════════════════════════════
# SUB-KERNEL 1: chunk_local_cumsum (scalar)
# ═════════════════════════════════════════════════════════════════════════════

@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in [1, 2, 4, 8]],
    key=['B', 'H', 'BT', 'IS_VARLEN', 'REVERSE'],
    **_AT_KW,
)
@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_scalar_kernel(
    s, o, scale, cu_seqlens, chunk_indices, T,
    B: tl.constexpr, H: tl.constexpr, BT: tl.constexpr,
    REVERSE: tl.constexpr, HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr, HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_s = tl.make_block_ptr(s + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def _chunk_local_cumsum(g, chunk_size, cu_seqlens=None, chunk_indices=None):
    B, T, H = g.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = _prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    g_out = torch.empty_like(g, dtype=torch.float32)
    chunk_local_cumsum_scalar_kernel[(NT, B * H)](
        s=g, o=g_out, scale=None, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, T=T, B=B, H=H, BT=BT,
        HEAD_FIRST=False, REVERSE=False,
    )
    return g_out

# ═════════════════════════════════════════════════════════════════════════════
# SUB-KERNEL 2: chunk_scaled_dot_kkt_fwd
# ═════════════════════════════════════════════════════════════════════════════

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=w, num_stages=s)
        for BK in [32, 64, 128] for w in [2, 4, 8] for s in [2, 3, 4]
    ],
    key=['H', 'K', 'BT', 'IS_VARLEN'],
    **_AT_KW,
)
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel(
    k, g, beta, A, cu_seqlens, chunk_indices, T,
    H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
    IS_VARLEN: tl.constexpr, USE_G: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p_b = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, tl.trans(b_k))
    if USE_G:
        p_g = tl.make_block_ptr(g + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_A *= _exp(b_g[:, None] - b_g[None, :])
    b_A *= b_b[:, None]
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (T, BT), (BT*H, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def _chunk_scaled_dot_kkt_fwd(k, g=None, beta=None, cu_seqlens=None,
                               chunk_size=64, output_dtype=torch.float32,
                               chunk_indices=None):
    B, T, H, K = k.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = _prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
    chunk_scaled_dot_kkt_fwd_kernel[(NT, B * H)](
        k=k, g=g, beta=beta, A=A, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, T=T, H=H, K=K, BT=BT,
    )
    return A

# ═════════════════════════════════════════════════════════════════════════════
# SUB-KERNEL 3: solve_tril (64x64 block inverse)
# ═════════════════════════════════════════════════════════════════════════════

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'DOT_PRECISION': 'ieee'}, num_warps=w, num_stages=s)
        for w in [2, 4, 8] for s in [2, 3, 4, 5]
    ],
    key=['H', 'BT', 'IS_VARLEN'],
    **_AT_KW,
)
@triton.jit(do_not_specialize=['T'])
def solve_tril_64x64_kernel(
    A, Ai, cu_seqlens, chunk_indices, T,
    H: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr, DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    # Load 4 diagonal 16x16 blocks from A
    p11 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t*BT, 0), (16, 16), (1, 0))
    p22 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t*BT+16, 16), (16, 16), (1, 0))
    p33 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t*BT+32, 32), (16, 16), (1, 0))
    p44 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t*BT+48, 48), (16, 16), (1, 0))
    b11 = -tl.where(m_A, tl.load(p11, boundary_check=(0, 1)).to(tl.float32), 0)
    b22 = -tl.where(m_A, tl.load(p22, boundary_check=(0, 1)).to(tl.float32), 0)
    b33 = -tl.where(m_A, tl.load(p33, boundary_check=(0, 1)).to(tl.float32), 0)
    b44 = -tl.where(m_A, tl.load(p44, boundary_check=(0, 1)).to(tl.float32), 0)

    # Invert each 16x16 block inline
    for i in range(2, min(16, T - i_t * BT)):
        b_a = -tl.load(A + (i_t*BT + i)*H*BT + o_i)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a = b_a + tl.sum(b_a[:, None] * b11, 0)
        b11 = tl.where((o_i == i)[:, None], b_a, b11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a = -tl.load(A + (i_t*BT + i)*H*BT + o_i + 16)
        b_a = tl.where(o_i < i - 16, b_a, 0.)
        b_a = b_a + tl.sum(b_a[:, None] * b22, 0)
        b22 = tl.where((o_i == i - 16)[:, None], b_a, b22)
    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a = -tl.load(A + (i_t*BT + i)*H*BT + o_i + 32)
        b_a = tl.where(o_i < i - 32, b_a, 0.)
        b_a = b_a + tl.sum(b_a[:, None] * b33, 0)
        b33 = tl.where((o_i == i - 32)[:, None], b_a, b33)
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a = -tl.load(A + (i_t*BT + i)*H*BT + o_i + 48)
        b_a = tl.where(o_i < i - 48, b_a, 0.)
        b_a = b_a + tl.sum(b_a[:, None] * b44, 0)
        b44 = tl.where((o_i == i - 48)[:, None], b_a, b44)
    b11 += m_I; b22 += m_I; b33 += m_I; b44 += m_I

    # Off-diagonal blocks from A
    p21 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t*BT+16, 0), (16, 16), (1, 0))
    p31 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t*BT+32, 0), (16, 16), (1, 0))
    p32 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t*BT+32, 16), (16, 16), (1, 0))
    p41 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t*BT+48, 0), (16, 16), (1, 0))
    p42 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t*BT+48, 16), (16, 16), (1, 0))
    p43 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t*BT+48, 32), (16, 16), (1, 0))
    bA21 = tl.load(p21, boundary_check=(0, 1)).to(tl.float32)
    bA31 = tl.load(p31, boundary_check=(0, 1)).to(tl.float32)
    bA32 = tl.load(p32, boundary_check=(0, 1)).to(tl.float32)
    bA41 = tl.load(p41, boundary_check=(0, 1)).to(tl.float32)
    bA42 = tl.load(p42, boundary_check=(0, 1)).to(tl.float32)
    bA43 = tl.load(p43, boundary_check=(0, 1)).to(tl.float32)

    # Schur complement: compute off-diagonal blocks of inverse
    bi21 = -tl.dot(tl.dot(b22, bA21, input_precision=DOT_PRECISION), b11, input_precision=DOT_PRECISION)
    bi32 = -tl.dot(tl.dot(b33, bA32, input_precision=DOT_PRECISION), b22, input_precision=DOT_PRECISION)
    bi43 = -tl.dot(tl.dot(b44, bA43, input_precision=DOT_PRECISION), b33, input_precision=DOT_PRECISION)
    bi31 = -tl.dot(b33,
        tl.dot(bA31, b11, input_precision=DOT_PRECISION) +
        tl.dot(bA32, bi21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION)
    bi42 = -tl.dot(b44,
        tl.dot(bA42, b22, input_precision=DOT_PRECISION) +
        tl.dot(bA43, bi32, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION)
    bi41 = -tl.dot(b44,
        tl.dot(bA41, b11, input_precision=DOT_PRECISION) +
        tl.dot(bA42, bi21, input_precision=DOT_PRECISION) +
        tl.dot(bA43, bi31, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION)

    # Store all 10 non-zero blocks of the inverse
    po = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t*BT, 0), (16, 16), (1, 0))
    tl.store(po, b11.to(po.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    po = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t*BT+16, 16), (16, 16), (1, 0))
    tl.store(po, b22.to(po.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    po = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t*BT+32, 32), (16, 16), (1, 0))
    tl.store(po, b33.to(po.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    po = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t*BT+48, 48), (16, 16), (1, 0))
    tl.store(po, b44.to(po.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    po = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t*BT+16, 0), (16, 16), (1, 0))
    tl.store(po, bi21.to(po.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    po = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t*BT+32, 0), (16, 16), (1, 0))
    tl.store(po, bi31.to(po.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    po = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t*BT+32, 16), (16, 16), (1, 0))
    tl.store(po, bi32.to(po.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    po = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t*BT+48, 0), (16, 16), (1, 0))
    tl.store(po, bi41.to(po.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    po = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t*BT+48, 16), (16, 16), (1, 0))
    tl.store(po, bi42.to(po.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    po = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t*BT+48, 32), (16, 16), (1, 0))
    tl.store(po, bi43.to(po.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


def _solve_tril(A, cu_seqlens=None, output_dtype=torch.float, chunk_indices=None):
    B, T, H, BT = A.shape
    assert BT == 64
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = _prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    Ai = torch.zeros_like(A, dtype=output_dtype)
    solve_tril_64x64_kernel[NT, B * H](
        A=A, Ai=Ai, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        T=T, H=H, BT=BT,
    )
    return Ai

# ═════════════════════════════════════════════════════════════════════════════
# SUB-KERNEL 4: recompute_w_u_fwd
# ═════════════════════════════════════════════════════════════════════════════

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [2, 4, 8] for s in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **_AT_KW,
)
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kernel(
    k, v, beta, w, u, A, g, cu_seqlens, chunk_indices, T,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_b = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))
    p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))
    if USE_G:
        p_g = tl.make_block_ptr(g + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = _exp(tl.load(p_g, boundary_check=(0,)))
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_w = tl.make_block_ptr(w + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_b[:, None]
        if USE_G:
            b_kb *= b_g[:, None]
        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def _recompute_w_u_fwd(k, v, beta, A, g=None, cu_seqlens=None, chunk_indices=None):
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = A.shape[-1]
    BK, BV = 64, 64
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = _prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    recompute_w_u_fwd_kernel[(NT, B*H)](
        k=k, v=v, beta=beta, w=w, u=u, A=A, g=g,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        T=T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV,
    )
    return w, u

# ═════════════════════════════════════════════════════════════════════════════
# SUB-KERNEL 5: chunk_gated_delta_rule_fwd_h (state propagation)
# ═════════════════════════════════════════════════════════════════════════════

_H_FWD_STAGES = [4, 3, 2] if _HAS_AMPERE_SMEM else [2, 1]
_H_FWD_BV = [32, 64] if _HAS_ADA_SMEM else [32]
_H_FWD_WARPS = [2, 4] if _IS_HOPPER_PLUS else [2, 4, 8, 16]

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': bv}, num_warps=w, num_stages=s)
        for w in _H_FWD_WARPS for s in _H_FWD_STAGES for bv in _H_FWD_BV
    ],
    key=['H', 'K', 'V', 'BT', 'USE_EXP2', 'TRANSPOSE_STATE'],
    **_AT_KW_CG,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_kernel_h(
    k, v, w, v_new, g, gk, h, h0, ht, cu_seqlens, chunk_offsets, T,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr, USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    if TRANSPOSE_STATE:
        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    h += (boh * H + i_h).to(tl.int64) * K*V
    v += (bos * H + i_h).to(tl.int64) * V
    k += (bos * H + i_h).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h).to(tl.int64) * V
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K*V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K*V

    # Load initial state
    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p0 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p0 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p0, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            if TRANSPOSE_STATE:
                p0 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p0 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_h2 += tl.load(p0, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            if TRANSPOSE_STATE:
                p0 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p0 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_h3 += tl.load(p0, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            if TRANSPOSE_STATE:
                p0 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p0 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_h4 += tl.load(p0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        i_t64 = i_t.to(tl.int64)
        # Store h for this chunk
        if TRANSPOSE_STATE:
            ph = tl.make_block_ptr(h + i_t64*H*K*V, (V, K), (K, 1), (i_v*BV, 0), (BV, 64), (1, 0))
        else:
            ph = tl.make_block_ptr(h + i_t64*H*K*V, (K, V), (V, 1), (0, i_v*BV), (64, BV), (1, 0))
        tl.store(ph, b_h1.to(ph.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                ph = tl.make_block_ptr(h + i_t64*H*K*V, (V, K), (K, 1), (i_v*BV, 64), (BV, 64), (1, 0))
            else:
                ph = tl.make_block_ptr(h + i_t64*H*K*V, (K, V), (V, 1), (64, i_v*BV), (64, BV), (1, 0))
            tl.store(ph, b_h2.to(ph.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                ph = tl.make_block_ptr(h + i_t64*H*K*V, (V, K), (K, 1), (i_v*BV, 128), (BV, 64), (1, 0))
            else:
                ph = tl.make_block_ptr(h + i_t64*H*K*V, (K, V), (V, 1), (128, i_v*BV), (64, BV), (1, 0))
            tl.store(ph, b_h3.to(ph.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                ph = tl.make_block_ptr(h + i_t64*H*K*V, (V, K), (K, 1), (i_v*BV, 192), (BV, 64), (1, 0))
            else:
                ph = tl.make_block_ptr(h + i_t64*H*K*V, (K, V), (V, 1), (192, i_v*BV), (64, BV), (1, 0))
            tl.store(ph, b_h4.to(ph.dtype.element_ty), boundary_check=(0, 1))

        # Compute v_new = u - w @ h  (delta rule correction)
        p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        if TRANSPOSE_STATE:
            b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        else:
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h4.to(b_w.dtype))

        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_vn = tl.make_block_ptr(v_new, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_vn, b_v.to(p_vn.dtype.element_ty), boundary_check=(0, 1))

        # Apply gating
        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + (bos * H + last_idx * H + i_h).to(tl.int64)).to(tl.float32)
            p_g = tl.make_block_ptr(g + (bos * H + i_h).to(tl.int64), (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            if USE_EXP2:
                b_v = b_v * tl.where(m_t, _exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = _exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, _exp(b_g_last - b_g), 0)[:, None]
                b_g_last = _exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk1 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
            if TRANSPOSE_STATE:
                b_h1 *= _exp(b_gk1)[None, :]
            else:
                b_h1 *= _exp(b_gk1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk2 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                if TRANSPOSE_STATE:
                    b_h2 *= _exp(b_gk2)[None, :]
                else:
                    b_h2 *= _exp(b_gk2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk3 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                if TRANSPOSE_STATE:
                    b_h3 *= _exp(b_gk3)[None, :]
                else:
                    b_h3 *= _exp(b_gk3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk4 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                if TRANSPOSE_STATE:
                    b_h4 *= _exp(b_gk4)[None, :]
                else:
                    b_h4 *= _exp(b_gk4)[:, None]

        b_v = b_v.to(k.dtype.element_ty)

        # Rank-1 state update: h += k^T @ v_new
        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if TRANSPOSE_STATE:
            b_h1 += tl.trans(tl.dot(b_k, b_v))
        else:
            b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h2 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h3 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h4 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h4 += tl.dot(b_k, b_v)

    # Store final state
    if STORE_FINAL_STATE:
        if TRANSPOSE_STATE:
            pt = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            pt = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(pt, b_h1.to(pt.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                pt = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                pt = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(pt, b_h2.to(pt.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                pt = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                pt = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(pt, b_h3.to(pt.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                pt = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                pt = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(pt, b_h4.to(pt.dtype.element_ty), boundary_check=(0, 1))


def _chunk_gated_delta_rule_fwd_h(
    k, w, u, g=None, gk=None, initial_state=None,
    output_final_state=False, chunk_size=64, save_new_value=True,
    cu_seqlens=None, chunk_indices=None, transpose_state_layout=False,
):
    B, T, H, K = k.shape
    V = u.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = _prepare_chunk_indices(cu_seqlens, BT)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(cu_seqlens) - 1
        NT = len(chunk_indices)
        chunk_offsets = _prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256

    if transpose_state_layout:
        h = k.new_empty(B, NT, H, V, K)
        final_state = k.new_zeros(N, H, V, K, dtype=torch.float32) if output_final_state else None
    else:
        h = k.new_empty(B, NT, H, K, V)
        final_state = k.new_zeros(N, H, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u) if save_new_value else None
    def grid(meta): return (triton.cdiv(V, meta['BV']), N*H)
    chunk_gated_delta_rule_fwd_kernel_h[grid](
        k=k, v=u, w=w, v_new=v_new, g=g, gk=gk,
        h=h, h0=initial_state, ht=final_state,
        cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets,
        T=T, H=H, K=K, V=V, BT=BT,
        USE_EXP2=False, TRANSPOSE_STATE=transpose_state_layout,
    )
    return h, v_new, final_state

# ═════════════════════════════════════════════════════════════════════════════
# SUB-KERNEL 6: chunk_fwd_o (output computation)
# ═════════════════════════════════════════════════════════════════════════════

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': 128, 'BV': 128}, num_warps=8, num_stages=3),
        triton.Config({'BK': 64, 'BV': 64}, num_warps=4, num_stages=3),
        triton.Config({'BK': 32, 'BV': 32}, num_warps=2, num_stages=3),
    ],
    key=['H', 'K', 'V', 'BT', 'TRANSPOSE_STATE'],
    **_AT_KW,
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o(
    q, k, v, h, g, g_gamma, o, cu_seqlens, chunk_indices, scale, T,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, USE_G_GAMMA: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K*V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        if TRANSPOSE_STATE:
            p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
        else:
            p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        if TRANSPOSE_STATE:
            b_o += tl.dot(b_q, tl.trans(b_h))
        else:
            b_o += tl.dot(b_q, b_h)
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * _exp(b_g)[:, None]
        b_A = b_A * _exp(b_g[:, None] - b_g[None, :])

    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_o = b_o * _exp(b_g)[:, None]
        b_A = b_A * _exp(b_g[:, None] - b_g[None, :])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def _chunk_fwd_o(q, k, v, h, g=None, g_gamma=None, scale=None,
                 cu_seqlens=None, chunk_size=64, chunk_indices=None,
                 transpose_state_layout=False):
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = _prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = K ** -0.5
    o = torch.empty_like(v)
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * H)
    chunk_fwd_kernel_o[grid](
        q=q, k=k, v=v, h=h, g=g, g_gamma=g_gamma, o=o,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        scale=scale, T=T, H=H, K=K, V=V, BT=BT,
        TRANSPOSE_STATE=transpose_state_layout,
    )
    return o

# ═════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION: Full chunked gated delta rule forward
# ═════════════════════════════════════════════════════════════════════════════

def _chunk_gated_delta_rule_fwd(
    q, k, v, g, beta, scale,
    initial_state=None, output_final_state=False,
    cu_seqlens=None, transpose_state_layout=False,
):
    chunk_indices = _prepare_chunk_indices(cu_seqlens, 64) if cu_seqlens is not None else None

    g = _chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    A = _chunk_scaled_dot_kkt_fwd(
        k=k, g=g, beta=beta, cu_seqlens=cu_seqlens,
        output_dtype=torch.float32, chunk_indices=chunk_indices,
    )
    A = _solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype, chunk_indices=chunk_indices)
    w, u = _recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g=g,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )
    h, v_new, final_state = _chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g,
        initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        transpose_state_layout=transpose_state_layout,
    )
    o = _chunk_fwd_o(
        q=q, k=k, v=v_new, h=h, g=g, scale=scale,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        transpose_state_layout=transpose_state_layout,
    )
    return g, o, A, final_state, initial_state

# ═════════════════════════════════════════════════════════════════════════════
# FUSED: cumsum + kkt (eliminates 1 kernel launch + g_cs HBM read in kkt)
# ═════════════════════════════════════════════════════════════════════════════

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=w, num_stages=s)
        for BK in [32, 64, 128] for w in [2, 4, 8] for s in [2, 3, 4]
    ],
    key=['H', 'K', 'BT', 'IS_VARLEN'],
    **_AT_KW,
)
@triton.jit(do_not_specialize=['T'])
def fused_cumsum_kkt_kernel(
    s, g_cs_out, k, beta, A, cu_seqlens, chunk_indices, T,
    H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_s = tl.make_block_ptr(s + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_g = tl.cumsum(b_s, axis=0)

    p_g = tl.make_block_ptr(g_cs_out + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    tl.store(p_g, b_g.to(p_g.dtype.element_ty), boundary_check=(0,))

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p_b = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, tl.trans(b_k))
    b_A *= _exp(b_g[:, None] - b_g[None, :])
    b_A *= b_b[:, None]
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (T, BT), (BT*H, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


# ═════════════════════════════════════════════════════════════════════════════
# PRECOMPUTE KERNEL: fused head expansion + gating
# ═════════════════════════════════════════════════════════════════════════════

@triton.jit
def _fused_precompute_kernel(
    q_ptr, k_ptr, a_ptr, b_ptr,
    A_log_ptr, dt_bias_ptr,
    q_out_ptr, k_out_ptr, g_out_ptr, beta_out_ptr,
    T, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
    RATIO: tl.constexpr,
):
    pid = tl.program_id(0)
    i_t = pid
    k_offs = tl.arange(0, K)

    for i_h in range(H):
        qk_base = i_t * H * K + i_h * K
        q_raw = tl.load(q_ptr + qk_base + k_offs)
        k_raw = tl.load(k_ptr + qk_base + k_offs)
        for r in range(RATIO):
            i_hv = i_h * RATIO + r
            out_base = i_t * HV * K + i_hv * K
            tl.store(q_out_ptr + out_base + k_offs, q_raw)
            tl.store(k_out_ptr + out_base + k_offs, k_raw)

    for i_hv in range(HV):
        A_log_val = tl.load(A_log_ptr + i_hv)
        dt_bias_val = tl.load(dt_bias_ptr + i_hv)
        a_val = tl.load(a_ptr + i_t * HV + i_hv).to(tl.float32)
        b_val = tl.load(b_ptr + i_t * HV + i_hv).to(tl.float32)
        x = a_val + dt_bias_val
        softplus_x = tl.where(x <= 20.0, tl.math.log(1.0 + tl.math.exp(x)), x)
        g = -tl.math.exp(A_log_val) * softplus_x
        beta = 1.0 / (1.0 + tl.math.exp(-b_val))
        tl.store(g_out_ptr + i_t * HV + i_hv, g)
        tl.store(beta_out_ptr + i_t * HV + i_hv, beta.to(tl.bfloat16))

# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT (optimized: inlined orchestration + buffer cache)
# ═════════════════════════════════════════════════════════════════════════════

_buf_cache = {}
_chunk_cache = {}
_chunk_fast = [None, None]  # [cu_seqlens_ref, (indices, offsets)]

def _get_buf(key, shape, dtype, device, zero=False):
    if key not in _buf_cache or _buf_cache[key].shape != shape:
        _buf_cache[key] = torch.zeros(shape, dtype=dtype, device=device) if zero else torch.empty(shape, dtype=dtype, device=device)
    return _buf_cache[key]

def _get_chunk_info(cu_seqlens, BT):
    """Cache chunk_indices and chunk_offsets; fast path avoids D2H copy."""
    if _chunk_fast[0] is cu_seqlens:
        return _chunk_fast[1]
    key = (tuple(cu_seqlens.tolist()), BT)
    if key not in _chunk_cache:
        _chunk_cache[key] = (
            _prepare_chunk_indices(cu_seqlens, BT),
            _prepare_chunk_offsets(cu_seqlens, BT),
        )
    result = _chunk_cache[key]
    _chunk_fast[0] = cu_seqlens
    _chunk_fast[1] = result
    return result


def kernel_fn(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    T, H, K = q.shape
    HV, V = v.shape[1], v.shape[2]
    N = cu_seqlens.shape[0] - 1
    ratio = HV // H
    BT = 64
    dev = q.device

    if isinstance(scale, torch.Tensor):
        scale = scale.item()

    # Cached buffers
    q_exp = _get_buf('q_exp', (1, T, HV, K), torch.bfloat16, dev)
    k_exp = _get_buf('k_exp', (1, T, HV, K), torch.bfloat16, dev)
    g_raw = _get_buf('g_raw', (1, T, HV), torch.float32, dev)
    beta = _get_buf('beta', (1, T, HV), torch.bfloat16, dev)
    g_cs = _get_buf('g_cs', (1, T, HV), torch.float32, dev)

    _fused_precompute_kernel[(T,)](
        q, k, a, b, A_log, dt_bias,
        q_exp, k_exp, g_raw, beta,
        T, H, HV, K, ratio, num_warps=1,
    )

    chunk_indices, chunk_offsets = _get_chunk_info(cu_seqlens, BT)
    NT = len(chunk_indices)

    A_mat = _get_buf('A_mat', (1, T, HV, BT), torch.float32, dev)
    fused_cumsum_kkt_kernel[(NT, HV)](
        s=g_raw, g_cs_out=g_cs, k=k_exp, beta=beta, A=A_mat,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        T=T, H=HV, K=K, BT=BT,
    )

    Ai = _get_buf('Ai', (1, T, HV, BT), torch.bfloat16, dev)
    Ai.zero_()
    solve_tril_64x64_kernel[NT, HV](
        A=A_mat, Ai=Ai, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        T=T, H=HV, BT=BT,
    )

    w_buf = _get_buf('w_buf', (1, T, HV, K), torch.bfloat16, dev)
    u_buf = _get_buf('u_buf', (1, T, HV, V), torch.bfloat16, dev)
    v_4d = v.unsqueeze(0)
    recompute_w_u_fwd_kernel[(NT, HV)](
        k=k_exp, v=v_4d, beta=beta, w=w_buf, u=u_buf, A=Ai, g=g_cs,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        T=T, H=HV, K=K, V=V, BT=BT, BK=64, BV=64,
    )

    h_buf = _get_buf('h_buf', (1, NT, HV, V, K), k_exp.dtype, dev)
    v_new = _get_buf('v_new', (1, T, HV, V), u_buf.dtype, dev)
    final_state = _get_buf('final_state', (N, HV, V, K), torch.float32, dev)
    def grid_h(meta): return (triton.cdiv(V, meta['BV']), N*HV)
    chunk_gated_delta_rule_fwd_kernel_h[grid_h](
        k=k_exp, v=u_buf, w=w_buf, v_new=v_new, g=g_cs, gk=None,
        h=h_buf, h0=state, ht=final_state,
        cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets,
        T=T, H=HV, K=K, V=V, BT=BT,
        USE_EXP2=False, TRANSPOSE_STATE=True,
    )

    o_buf = _get_buf('o_buf', (1, T, HV, V), torch.bfloat16, dev)
    def grid_o(meta): return (triton.cdiv(V, meta['BV']), NT, HV)
    chunk_fwd_kernel_o[grid_o](
        q=q_exp, k=k_exp, v=v_new, h=h_buf, g=g_cs, g_gamma=None, o=o_buf,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        scale=scale, T=T, H=HV, K=K, V=V, BT=BT,
        TRANSPOSE_STATE=True,
    )

    return o_buf.squeeze(0), final_state


def kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state):
    """DPS entry point for FlashInfer-Bench."""
    o, fs = kernel_fn(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
    output.copy_(o)
    new_state.copy_(fs)
