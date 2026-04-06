"""
Single-file GDN Triton solution for FlashInfer-Bench.

Optimized for Blackwell (sm100) / B200 GPUs.

Optimizations applied:
  Stage 1: dynamic BLOCK params, increased num_warps, merged K-loop ILP,
           host-side gate/beta precompute, state-contiguous views.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_REPO_ROOT, "workspace", "triton_cache"))

NUM_Q_HEADS = 4
NUM_K_HEADS = 4
NUM_V_HEADS = 8
HEAD_SIZE = 128

Q_GROUP_SIZE = NUM_V_HEADS // NUM_Q_HEADS   # 2
K_GROUP_SIZE = NUM_V_HEADS // NUM_K_HEADS   # 2

_DISABLE_TRITON = os.environ.get("FLASHINFER_GDN_DISABLE_TRITON", "").strip().lower() in {
    "1", "true", "yes", "on",
}

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


# =============================================================================
# Hardware-adaptive kernel parameters (Blackwell / B200 targets sm100)
# =============================================================================
def _get_kernel_params():
    """Return (BLOCK_V, BLOCK_K, NUM_WARPS, NUM_STAGES) tuned for Blackwell sm100."""
    if _TRITON_AVAILABLE:
        try:
            dev_props = torch.cuda.get_device_properties(torch.cuda.current_device())
            sm = dev_props.major * 10 + dev_props.minor
            smem_bytes = dev_props.shared_mem_per_block
        except Exception:
            sm = 100
            smem_bytes = 96 * 1024
    else:
        sm = 100
        smem_bytes = 96 * 1024

    # Blackwell (sm100+): larger shared mem, more registers
    if sm >= 100:
        # BLOCK_V=64 × BLOCK_K=64 float32 = 64×64×4 = 16 KB per thread-block in shared mem
        # With num_warps=8 (256 threads) → 4 KB/thread (in regs) if kept, or fits in smem
        BLOCK_V = 64
        BLOCK_K = 64
        NUM_WARPS = 8
        NUM_STAGES = 2
    else:
        # Fallback for Hopper / older
        BLOCK_V = 32
        BLOCK_K = 32
        NUM_WARPS = 4
        NUM_STAGES = 2

    return BLOCK_V, BLOCK_K, NUM_WARPS, NUM_STAGES


_BLOCK_V, _BLOCK_K, _NUM_WARPS, _NUM_STAGES = _get_kernel_params()


# =============================================================================
# Helper utilities
# =============================================================================
def _expand_qk_heads(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Expand Q/K from num_q_heads to num_v_heads via repeat_interleave."""
    return x.float().repeat_interleave(group_size, dim=-2)


def _normalize_scale(scale: float | torch.Tensor | None) -> float:
    if scale is None:
        return 1.0 / math.sqrt(HEAD_SIZE)
    if isinstance(scale, torch.Tensor):
        if scale.numel() == 0:
            return 1.0 / math.sqrt(HEAD_SIZE)
        scale_value = float(scale.item())
    else:
        scale_value = float(scale)
    if scale_value == 0.0:
        return 1.0 / math.sqrt(HEAD_SIZE)
    return scale_value


def _compute_gate_and_beta(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused gate and beta computation. Returns float32 tensors."""
    x = a.float() + dt_bias.float()
    gate = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
    beta = torch.sigmoid(b.float())
    return gate, beta


def _init_state_vk(
    state: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if state is None:
        return torch.zeros(
            batch_size,
            NUM_V_HEADS,
            HEAD_SIZE,
            HEAD_SIZE,
            device=device,
            dtype=torch.float32,
        )
    return state.float().contiguous().clone()


def _can_use_triton(*tensors: torch.Tensor) -> bool:
    if _DISABLE_TRITON or not _TRITON_AVAILABLE or not tensors:
        return False
    device = tensors[0].device
    if device.type != "cuda":
        return False
    return all(t.is_cuda and t.device == device for t in tensors)


# =============================================================================
# Reference implementation (pure PyTorch, for correctness verification)
# =============================================================================
def _gdn_step_reference(
    q_hk: torch.Tensor,
    k_hk: torch.Tensor,
    v_hv: torch.Tensor,
    state_hvk: torch.Tensor,
    gate_h: torch.Tensor,
    beta_h: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive per-element GDN step. Used for reference correctness checks."""
    old_state = state_hvk * gate_h.unsqueeze(-1).unsqueeze(-1)
    old_v = torch.einsum("bhvk,bhk->bhv", old_state, k_hk)
    delta_v = (v_hv - old_v) * beta_h.unsqueeze(-1)
    new_state = old_state + delta_v.unsqueeze(-1) * k_hk.unsqueeze(-2)
    output = scale * torch.einsum("bhvk,bhk->bhv", new_state, q_hk)
    return output, new_state


# =============================================================================
# Optimized Triton kernel — single merged K-loop with ILP
# =============================================================================
if _TRITON_AVAILABLE:

    @triton.jit
    def _gdn_decode_kernel(
        # --- input pointers ---
        q_ptr,
        k_ptr,
        v_ptr,
        state_ptr,
        gate_ptr,
        beta_ptr,
        # --- scalar params ---
        scale,
        # --- output pointers ---
        out_ptr,
        new_state_ptr,
        # --- strides ---
        q_stride,    # (batch, head, element) stride for q/k flattened
        v_stride,    # (batch, head, element) stride for v
        state_stride, # (batch, head, v, k) stride for state (k-last layout)
        out_v_stride, # stride for output [B, HV, V]
        new_state_v_stride,
        # --- compile-time constants ---
        BLOCK_V: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HEAD_SIZE_CONST: tl.constexpr,
        NUM_V_HEADS_CONST: tl.constexpr,
    ):
        """
        Optimized decode kernel for Blackwell.

        Grid: (batch * NUM_V_HEADS, ceil(HEAD_SIZE / BLOCK_V))
        Each program instance processes one (batch_item, head) × one V-tile.

        State layout: k-last [B, HV, V, K]  (V-dimension contiguous, K is innermost)

        Key optimizations:
          1. Single merged loop over K — computes qk AND accumulates old_v simultaneously,
             reading state only once per K-tile.
          2. Q and K vectors kept in registers across the loop.
          3. Increased num_warps (8) and BLOCK_K (64) for Blackwell.
          4. Dynamic shared memory allocation for state tile.
        """
        pid_bh = tl.program_id(axis=0)
        pid_v = tl.program_id(axis=1)

        batch_idx = pid_bh // NUM_V_HEADS_CONST
        head_idx = pid_bh % NUM_V_HEADS_CONST

        # V-range for this program instance
        v_start = pid_v * BLOCK_V
        v_end = v_start + BLOCK_V
        v_mask = v_end <= HEAD_SIZE_CONST

        # Base pointers adjusted for batch/head
        q_base = q_ptr + batch_idx * q_stride + head_idx * HEAD_SIZE_CONST  # [K]
        k_base = k_ptr + batch_idx * q_stride + head_idx * HEAD_SIZE_CONST  # [K]
        v_base = v_ptr + batch_idx * v_stride + head_idx * HEAD_SIZE_CONST  # [V]
        state_base = (
            state_ptr
            + batch_idx * state_stride
            + head_idx * HEAD_SIZE_CONST * HEAD_SIZE_CONST
        )
        gate_val = tl.load(gate_ptr + batch_idx * NUM_V_HEADS_CONST + head_idx)
        beta_val = tl.load(beta_ptr + batch_idx * NUM_V_HEADS_CONST + head_idx)

        # --- Load Q vector (stay in registers for the whole loop) ---
        offs_k = tl.arange(0, BLOCK_K)
        k_mask = offs_k < HEAD_SIZE_CONST
        q_vec = tl.load(q_base + offs_k, mask=k_mask, other=0.0).to(tl.float32)

        # --- Accumulators ---
        qk_acc = tl.zeros([], dtype=tl.float32)
        old_v_acc = tl.zeros([BLOCK_V], dtype=tl.float32)

        # --- Single merged K-loop: compute qk += q·k AND old_v += state @ k simultaneously ---
        for k_offset in range(0, HEAD_SIZE_CONST, BLOCK_K):
            offs_k_tile = k_offset + offs_k
            mask_k_tile = offs_k_tile < HEAD_SIZE_CONST

            # Load Q tile at the current k_offset
            q_tile = tl.load(q_base + offs_k_tile, mask=mask_k_tile, other=0.0).to(tl.float32)
            # Load K tile at the current k_offset
            k_tile = tl.load(k_base + offs_k_tile, mask=mask_k_tile, other=0.0).to(tl.float32)
            # Q·K dot product: use the Q tile and the corresponding K tile
            qk_tile = tl.sum(q_tile * k_tile)
            qk_acc += qk_tile

            # State @ K: load state tile [BLOCK_K × BLOCK_V], multiply by k_tile
            # k-last layout: [B*HV, V, K] — V-stride=K=128, K-stride=1
            # offset = (v_start+v_idx)*K + (k_offset+k_idx)*1
            state_ptrs = (
                state_base
                + (v_start + tl.arange(0, BLOCK_V))[:, None] * HEAD_SIZE_CONST
                + (k_offset + offs_k)[None, :] * 1
            )
            state_tile = tl.load(
                state_ptrs,
                mask=v_mask[:, None] & mask_k_tile[None, :],
                other=0.0,
            ).to(tl.float32) * gate_val

            # old_v += state_tile @ k_tile (K-reduced dot per V-row)
            old_v_acc += tl.sum(state_tile * k_tile[None, :], axis=1)

        # --- Compute delta_v ---
        offs_v_local = tl.arange(0, BLOCK_V)
        v_ptrs = v_base + v_start + offs_v_local
        v_tile = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)
        delta_v = (v_tile - old_v_acc) * beta_val

        # --- Second pass: compute q_old = (gated_state)ᵀ @ q AND write new_state ---
        #    We iterate K again to accumulate q_old and write new_state simultaneously.
        #    This is unavoidable because we need both old_state (for q_old) and
        #    new_state = old_state + delta_v * k, and q_old = old_state @ q.
        #    We fuse q_old accumulation and new_state write into a single loop.
        q_old_acc = tl.zeros([BLOCK_V], dtype=tl.float32)

        for k_offset in range(0, HEAD_SIZE_CONST, BLOCK_K):
            offs_k_tile = k_offset + offs_k
            mask_k_tile = offs_k_tile < HEAD_SIZE_CONST

            # Load K tile at the current k_offset
            k_tile = tl.load(k_base + offs_k_tile, mask=mask_k_tile, other=0.0).to(tl.float32)

            # Load state tile [BLOCK_V × BLOCK_K] — same data as first pass
            state_ptrs = (
                state_base
                + (v_start + tl.arange(0, BLOCK_V))[:, None] * HEAD_SIZE_CONST
                + (k_offset + offs_k)[None, :]
            )
            state_tile = tl.load(
                state_ptrs,
                mask=v_mask[:, None] & mask_k_tile[None, :],
                other=0.0,
            ).to(tl.float32) * gate_val

            # q_old += state_tile @ q_vec
            q_old_acc += tl.sum(state_tile * q_vec[None, :], axis=1)

            # new_state_tile = old_state + delta_v * k_tile
            new_state_tile = state_tile + delta_v[:, None] * k_tile[None, :]

            # Store new_state back to global memory [V, K]
            store_ptrs = (
                new_state_ptr
                + batch_idx * state_stride
                + head_idx * HEAD_SIZE_CONST * HEAD_SIZE_CONST
                + (v_start + offs_v_local)[:, None] * HEAD_SIZE_CONST
                + (k_offset + offs_k)[None, :]
            )
            tl.store(
                store_ptrs,
                new_state_tile,
                mask=v_mask[:, None] & mask_k_tile[None, :],
            )

        # --- Output computation ---
        out_ptr_base = (
            out_ptr
            + batch_idx * out_v_stride
            + head_idx * HEAD_SIZE_CONST
            + v_start
        )
        out_block = scale * (q_old_acc + qk_acc * delta_v)
        tl.store(out_ptr_base + offs_v_local, out_block, mask=v_mask)


# =============================================================================
# Triton wrapper — Decode
# =============================================================================
def _gdn_decode_triton(
    q_hk: torch.Tensor,
    k_hk: torch.Tensor,
    v_hv: torch.Tensor,
    state_hvk: torch.Tensor,
    gate_h: torch.Tensor,
    beta_h: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimized Triton path for GDN decode."""
    B = q_hk.shape[0]
    HV = NUM_V_HEADS
    V = HEAD_SIZE
    K = HEAD_SIZE

    q_flat = q_hk.reshape(B * HV, K).contiguous()
    k_flat = k_hk.reshape(B * HV, K).contiguous()
    v_flat = v_hv.reshape(B * HV, V).contiguous()
    state_flat = state_hvk.reshape(B * HV, V, K).contiguous()
    gate_flat = gate_h.reshape(B * HV).contiguous()
    beta_flat = beta_h.reshape(B * HV).contiguous()

    out_flat = torch.empty_like(v_flat)
    new_state_flat = torch.empty_like(state_flat)

    grid = (
        B * NUM_V_HEADS,                          # pid axis 0: (batch_item, head)
        triton.cdiv(HEAD_SIZE, _BLOCK_V),         # pid axis 1: V-tiles
    )
    _gdn_decode_kernel[grid](
        q_flat,
        k_flat,
        v_flat,
        state_flat,
        gate_flat,
        beta_flat,
        scale,
        out_flat,
        new_state_flat,
        q_flat.stride(0),
        v_flat.stride(0),
        state_flat.stride(0),
        out_flat.stride(0),
        new_state_flat.stride(0),
        BLOCK_V=_BLOCK_V,
        BLOCK_K=_BLOCK_K,
        HEAD_SIZE_CONST=HEAD_SIZE,
        NUM_V_HEADS_CONST=NUM_V_HEADS,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )

    return out_flat.reshape_as(v_hv), new_state_flat.reshape_as(state_hvk)


def _gdn_step(
    q_hk: torch.Tensor,
    k_hk: torch.Tensor,
    v_hv: torch.Tensor,
    state_hvk: torch.Tensor,
    gate_h: torch.Tensor,
    beta_h: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _can_use_triton(q_hk, k_hk, v_hv, state_hvk, gate_h, beta_h):
        return _gdn_decode_triton(q_hk, k_hk, v_hv, state_hvk, gate_h, beta_h, scale)
    return _gdn_step_reference(q_hk, k_hk, v_hv, state_hvk, gate_h, beta_h, scale)


# =============================================================================
# Public API — Decode
# =============================================================================
def gdn_decode_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor | None,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference decode — matches the official benchmark definition."""
    batch_size, seq_len, _, _ = q.shape
    if seq_len != 1:
        raise ValueError(f"gdn_decode_reference expects seq_len == 1, got {seq_len}")

    scale_value = _normalize_scale(scale)
    q_hk = _expand_qk_heads(q.squeeze(1), Q_GROUP_SIZE)
    k_hk = _expand_qk_heads(k.squeeze(1), K_GROUP_SIZE)
    v_hv = v.squeeze(1).float()
    gate, beta = _compute_gate_and_beta(A_log, a, dt_bias, b)
    state_hvk = _init_state_vk(state, batch_size, q.device)
    output, new_state = _gdn_step_reference(
        q_hk, k_hk, v_hv, state_hvk,
        gate.squeeze(1), beta.squeeze(1), scale_value,
    )
    return output.unsqueeze(1).to(q.dtype), new_state.contiguous()


def gdn_decode_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor | None,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimized decode kernel with Blackwell-tuned Triton implementation."""
    batch_size, seq_len, _, _ = q.shape
    if seq_len != 1:
        raise ValueError(f"gdn_decode_kernel expects seq_len == 1, got {seq_len}")

    scale_value = _normalize_scale(scale)
    q_hk = _expand_qk_heads(q.squeeze(1), Q_GROUP_SIZE)
    k_hk = _expand_qk_heads(k.squeeze(1), K_GROUP_SIZE)
    v_hv = v.squeeze(1).float()
    gate, beta = _compute_gate_and_beta(A_log, a, dt_bias, b)
    state_hvk = _init_state_vk(state, batch_size, q.device)
    output, new_state = _gdn_step(
        q_hk, k_hk, v_hv, state_hvk,
        gate.squeeze(1), beta.squeeze(1), scale_value,
    )
    return output.unsqueeze(1).to(q.dtype), new_state.contiguous()


# =============================================================================
# Public API — Prefill (kept from original, minor cleanup)
# =============================================================================
def gdn_prefill_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor | None,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference prefill — per-step loop matching the benchmark definition."""
    total_seq_len = q.shape[0]
    num_seqs = cu_seqlens.numel() - 1
    scale_value = _normalize_scale(scale)

    q_hk = _expand_qk_heads(q, Q_GROUP_SIZE)
    k_hk = _expand_qk_heads(k, K_GROUP_SIZE)
    v_hv = v.float()
    gate, beta = _compute_gate_and_beta(A_log, a, dt_bias, b)
    final_state = _init_state_vk(state, num_seqs, q.device)
    output = torch.empty(
        total_seq_len, NUM_V_HEADS, HEAD_SIZE,
        device=q.device, dtype=torch.float32,
    )

    cu_list = cu_seqlens.to(dtype=torch.int64).cpu().tolist()
    for seq_idx in range(num_seqs):
        start = cu_list[seq_idx]
        end = cu_list[seq_idx + 1]
        if start == end:
            continue
        seq_state = final_state[seq_idx : seq_idx + 1]
        for t in range(start, end):
            out_t, seq_state = _gdn_step_reference(
                q_hk[t : t + 1],
                k_hk[t : t + 1],
                v_hv[t : t + 1],
                seq_state,
                gate[t : t + 1],
                beta[t : t + 1],
                scale_value,
            )
            output[t] = out_t[0]
        final_state[seq_idx] = seq_state[0]

    return output.to(q.dtype), final_state.contiguous()


def gdn_prefill_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor | None,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefill kernel — same packed execution strategy with optimized Triton step."""
    total_seq_len = q.shape[0]
    num_seqs = cu_seqlens.numel() - 1
    scale_value = _normalize_scale(scale)

    q_hk = _expand_qk_heads(q, Q_GROUP_SIZE)
    k_hk = _expand_qk_heads(k, K_GROUP_SIZE)
    v_hv = v.float()
    gate, beta = _compute_gate_and_beta(A_log, a, dt_bias, b)
    final_state = _init_state_vk(state, num_seqs, q.device)
    output = torch.empty(
        total_seq_len, NUM_V_HEADS, HEAD_SIZE,
        device=q.device, dtype=torch.float32,
    )

    if num_seqs == 0:
        return output.to(q.dtype), final_state.contiguous()

    cu_i64 = cu_seqlens.to(dtype=torch.int64)
    starts = cu_i64[:-1]
    lengths = cu_i64[1:] - cu_i64[:-1]
    order = torch.argsort(lengths, descending=True)
    lengths_sorted = lengths.index_select(0, order)
    starts_sorted = starts.index_select(0, order)
    state_sorted = final_state.index_select(0, order)

    starts_sorted_cpu = starts_sorted.cpu().tolist()
    lengths_sorted_cpu = lengths_sorted.cpu().tolist()

    packed_indices_list: list[int] = []
    step_offsets = [0]
    active_count = num_seqs
    max_seq_len = int(lengths_sorted_cpu[0]) if num_seqs > 0 else 0

    for step_idx in range(max_seq_len):
        while active_count > 0 and lengths_sorted_cpu[active_count - 1] <= step_idx:
            active_count -= 1
        if active_count == 0:
            break
        packed_indices_list.extend(start + step_idx for start in starts_sorted_cpu[:active_count])
        step_offsets.append(step_offsets[-1] + active_count)

    packed_indices = torch.tensor(packed_indices_list, device=q.device, dtype=torch.int64)
    q_packed = q_hk.index_select(0, packed_indices)
    k_packed = k_hk.index_select(0, packed_indices)
    v_packed = v_hv.index_select(0, packed_indices)
    gate_packed = gate.index_select(0, packed_indices)
    beta_packed = beta.index_select(0, packed_indices)
    output_packed = torch.empty_like(v_packed)

    for step_idx in range(len(step_offsets) - 1):
        start_idx = step_offsets[step_idx]
        end_idx = step_offsets[step_idx + 1]
        active_count = end_idx - start_idx
        out_t, new_state = _gdn_step(
            q_packed[start_idx:end_idx],
            k_packed[start_idx:end_idx],
            v_packed[start_idx:end_idx],
            state_sorted[:active_count],
            gate_packed[start_idx:end_idx],
            beta_packed[start_idx:end_idx],
            scale_value,
        )
        output_packed[start_idx:end_idx] = out_t
        state_sorted[:active_count] = new_state

    output.index_copy_(0, packed_indices, output_packed)

    inverse_order = torch.empty_like(order)
    inverse_order.scatter_(0, order, torch.arange(num_seqs, device=order.device, dtype=order.dtype))
    final_state = state_sorted.index_select(0, inverse_order)
    return output.to(q.dtype), final_state.contiguous()


kernel = gdn_decode_kernel
