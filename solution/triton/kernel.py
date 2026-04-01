"""Single-file GDN Triton solution for FlashInfer-Bench.

The official benchmark definitions still use names such as
`gdn_decode_qk4_v8_d128_k_last`, but local helper names here stay generic.
This module keeps one maintainable implementation file under `solution/triton/`.
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

Q_GROUP_SIZE = NUM_V_HEADS // NUM_Q_HEADS
K_GROUP_SIZE = NUM_V_HEADS // NUM_K_HEADS

_DISABLE_TRITON = os.environ.get("FLASHINFER_GDN_DISABLE_TRITON", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def _expand_qk_heads(x: torch.Tensor, group_size: int) -> torch.Tensor:
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
    x = a.float() + dt_bias.float()
    gate = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
    beta = torch.sigmoid(b.float())
    return gate, beta


def _init_state_vk(state: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
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


def _gdn_step_reference(
    q_hk: torch.Tensor,
    k_hk: torch.Tensor,
    v_hv: torch.Tensor,
    state_hvk: torch.Tensor,
    gate_h: torch.Tensor,
    beta_h: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    old_state = state_hvk * gate_h.unsqueeze(-1).unsqueeze(-1)
    old_v = torch.einsum("bhvk,bhk->bhv", old_state, k_hk)
    delta_v = (v_hv - old_v) * beta_h.unsqueeze(-1)
    new_state = old_state + delta_v.unsqueeze(-1) * k_hk.unsqueeze(-2)
    output = scale * torch.einsum("bhvk,bhk->bhv", new_state, q_hk)
    return output, new_state


if _TRITON_AVAILABLE:

    @triton.jit
    def _gdn_step_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        state_ptr,
        gate_ptr,
        beta_ptr,
        scale,
        out_ptr,
        new_state_ptr,
        q_stride,
        v_stride,
        state_stride,
        BLOCK_V: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HEAD_SIZE_CONST: tl.constexpr,
    ):
        pid_item = tl.program_id(axis=0)
        pid_v = tl.program_id(axis=1)

        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = offs_v < HEAD_SIZE_CONST

        q_item_ptr = q_ptr + pid_item * q_stride
        k_item_ptr = k_ptr + pid_item * q_stride
        v_item_ptr = v_ptr + pid_item * v_stride
        state_item_ptr = state_ptr + pid_item * state_stride
        out_item_ptr = out_ptr + pid_item * v_stride
        new_state_item_ptr = new_state_ptr + pid_item * state_stride

        gate = tl.load(gate_ptr + pid_item)
        beta = tl.load(beta_ptr + pid_item)

        qk = tl.zeros([], dtype=tl.float32)
        old_v = tl.zeros([BLOCK_V], dtype=tl.float32)
        v_block = tl.load(v_item_ptr + offs_v, mask=mask_v, other=0).to(tl.float32)

        for k_offset in tl.static_range(0, HEAD_SIZE_CONST, BLOCK_K):
            offs_k = k_offset + tl.arange(0, BLOCK_K)
            mask_k = offs_k < HEAD_SIZE_CONST

            q_block = tl.load(q_item_ptr + offs_k, mask=mask_k, other=0).to(tl.float32)
            k_block = tl.load(k_item_ptr + offs_k, mask=mask_k, other=0).to(tl.float32)
            qk += tl.sum(q_block * k_block, axis=0)

            state_ptrs = state_item_ptr + offs_v[:, None] * HEAD_SIZE_CONST + offs_k[None, :]
            old_state_block = tl.load(
                state_ptrs,
                mask=mask_v[:, None] & mask_k[None, :],
                other=0,
            ).to(tl.float32) * gate
            old_v += tl.sum(old_state_block * k_block[None, :], axis=1)

        delta_v = (v_block - old_v) * beta

        q_old = tl.zeros([BLOCK_V], dtype=tl.float32)
        for k_offset in tl.static_range(0, HEAD_SIZE_CONST, BLOCK_K):
            offs_k = k_offset + tl.arange(0, BLOCK_K)
            mask_k = offs_k < HEAD_SIZE_CONST

            q_block = tl.load(q_item_ptr + offs_k, mask=mask_k, other=0).to(tl.float32)
            k_block = tl.load(k_item_ptr + offs_k, mask=mask_k, other=0).to(tl.float32)
            state_ptrs = state_item_ptr + offs_v[:, None] * HEAD_SIZE_CONST + offs_k[None, :]
            old_state_block = tl.load(
                state_ptrs,
                mask=mask_v[:, None] & mask_k[None, :],
                other=0,
            ).to(tl.float32) * gate
            q_old += tl.sum(old_state_block * q_block[None, :], axis=1)
            new_state_block = old_state_block + delta_v[:, None] * k_block[None, :]
            tl.store(new_state_item_ptr + offs_v[:, None] * HEAD_SIZE_CONST + offs_k[None, :], new_state_block, mask=mask_v[:, None] & mask_k[None, :])

        out_block = scale * (q_old + qk * delta_v)
        tl.store(out_item_ptr + offs_v, out_block, mask=mask_v)


def _gdn_step_triton(
    q_hk: torch.Tensor,
    k_hk: torch.Tensor,
    v_hv: torch.Tensor,
    state_hvk: torch.Tensor,
    gate_h: torch.Tensor,
    beta_h: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_flat = q_hk.reshape(-1, HEAD_SIZE).contiguous()
    k_flat = k_hk.reshape(-1, HEAD_SIZE).contiguous()
    v_flat = v_hv.reshape(-1, HEAD_SIZE).contiguous()
    state_flat = state_hvk.reshape(-1, HEAD_SIZE, HEAD_SIZE).contiguous()
    gate_flat = gate_h.reshape(-1).contiguous()
    beta_flat = beta_h.reshape(-1).contiguous()

    out_flat = torch.empty_like(v_flat)
    new_state_flat = torch.empty_like(state_flat)

    grid = lambda meta: (q_flat.shape[0], triton.cdiv(HEAD_SIZE, meta["BLOCK_V"]))
    _gdn_step_kernel[grid](
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
        BLOCK_V=32,
        BLOCK_K=32,
        HEAD_SIZE_CONST=HEAD_SIZE,
        num_warps=4,
        num_stages=2,
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
        return _gdn_step_triton(q_hk, k_hk, v_hv, state_hvk, gate_h, beta_h, scale)
    return _gdn_step_reference(q_hk, k_hk, v_hv, state_hvk, gate_h, beta_h, scale)


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
        q_hk,
        k_hk,
        v_hv,
        state_hvk,
        gate.squeeze(1),
        beta.squeeze(1),
        scale_value,
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
        q_hk,
        k_hk,
        v_hv,
        state_hvk,
        gate.squeeze(1),
        beta.squeeze(1),
        scale_value,
    )
    return output.unsqueeze(1).to(q.dtype), new_state.contiguous()


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
    total_seq_len = q.shape[0]
    num_seqs = cu_seqlens.numel() - 1
    scale_value = _normalize_scale(scale)

    q_hk = _expand_qk_heads(q, Q_GROUP_SIZE)
    k_hk = _expand_qk_heads(k, K_GROUP_SIZE)
    v_hv = v.float()
    gate, beta = _compute_gate_and_beta(A_log, a, dt_bias, b)
    final_state = _init_state_vk(state, num_seqs, q.device)
    output = torch.empty(total_seq_len, NUM_V_HEADS, HEAD_SIZE, device=q.device, dtype=torch.float32)

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
    total_seq_len = q.shape[0]
    num_seqs = cu_seqlens.numel() - 1
    scale_value = _normalize_scale(scale)

    q_hk = _expand_qk_heads(q, Q_GROUP_SIZE)
    k_hk = _expand_qk_heads(k, K_GROUP_SIZE)
    v_hv = v.float()
    gate, beta = _compute_gate_and_beta(A_log, a, dt_bias, b)
    final_state = _init_state_vk(state, num_seqs, q.device)
    output = torch.empty(total_seq_len, NUM_V_HEADS, HEAD_SIZE, device=q.device, dtype=torch.float32)

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
