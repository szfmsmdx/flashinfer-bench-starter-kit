"""FlashInfer GDN solution entrypoints.

This file intentionally mirrors the validated PyTorch baselines in
`cuda-evolve-oss/kernels/` so the starter kit can package and benchmark the
same semantics. The current default in `config.toml` points to
`gdn_decode_kernel`. To benchmark prefill instead, switch:

- `solution.definition` -> `gdn_prefill_qk4_v8_d128_k_last`
- `build.entry_point` -> `kernel.py::gdn_prefill_kernel`
- keep `destination_passing_style = false`
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

NUM_Q_HEADS = 4
NUM_K_HEADS = 4
NUM_V_HEADS = 8
HEAD_SIZE = 128
Q_GROUP_SIZE = NUM_V_HEADS // NUM_Q_HEADS
K_GROUP_SIZE = NUM_V_HEADS // NUM_K_HEADS


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


def _expand_q_heads(q: torch.Tensor) -> torch.Tensor:
    return q.float().repeat_interleave(Q_GROUP_SIZE, dim=-2)


def _expand_k_heads(k: torch.Tensor) -> torch.Tensor:
    return k.float().repeat_interleave(K_GROUP_SIZE, dim=-2)


def _compute_gate_and_beta(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = a.float() + dt_bias.float()
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
    beta = torch.sigmoid(b.float())
    return g, beta


def _init_state_hkv(state: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
    if state is None:
        return torch.zeros(
            batch_size,
            NUM_V_HEADS,
            HEAD_SIZE,
            HEAD_SIZE,
            device=device,
            dtype=torch.float32,
        )
    return state.float().transpose(-1, -2).contiguous()


def _gdn_step(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    state_hkv: torch.Tensor,
    g_t: torch.Tensor,
    beta_t: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    old_state = g_t.unsqueeze(-1).unsqueeze(-1) * state_hkv
    flat_state = old_state.reshape(-1, HEAD_SIZE, HEAD_SIZE)
    flat_k = k_t.reshape(-1, HEAD_SIZE)
    flat_q = q_t.reshape(-1, HEAD_SIZE)

    old_v = torch.bmm(flat_k.unsqueeze(1), flat_state).squeeze(1).reshape_as(v_t)
    delta_v = beta_t.unsqueeze(-1) * (v_t - old_v)
    new_state = torch.baddbmm(
        flat_state,
        flat_k.unsqueeze(-1),
        delta_v.reshape(-1, 1, HEAD_SIZE),
    ).reshape_as(old_state)
    output = scale * torch.bmm(flat_q.unsqueeze(1), new_state.reshape(-1, HEAD_SIZE, HEAD_SIZE))
    return output.squeeze(1).reshape_as(v_t), new_state


def _gdn_step_decode(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    state_hkv: torch.Tensor,
    g_t: torch.Tensor,
    beta_t: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    old_state = g_t.unsqueeze(-1).unsqueeze(-1) * state_hkv
    flat_state = old_state.reshape(-1, HEAD_SIZE, HEAD_SIZE)
    flat_k = k_t.reshape(-1, HEAD_SIZE)
    flat_q = q_t.reshape(-1, HEAD_SIZE)

    # Decode only: reuse the old-state read for both k @ state and q @ state,
    # then apply the rank-1 update analytically in the output space.
    qk_state = torch.stack((flat_k, flat_q), dim=1)
    kv_proj = torch.bmm(qk_state, flat_state)
    old_v = kv_proj[:, 0, :].reshape_as(v_t)
    q_old = kv_proj[:, 1, :].reshape_as(v_t)
    delta_v = beta_t.unsqueeze(-1) * (v_t - old_v)
    new_state = torch.baddbmm(
        flat_state,
        flat_k.unsqueeze(-1),
        delta_v.reshape(-1, 1, HEAD_SIZE),
    ).reshape_as(old_state)
    qk = (flat_q * flat_k).sum(dim=-1, keepdim=True).reshape(*v_t.shape[:-1], 1)
    output = scale * (q_old + qk * delta_v)
    return output, new_state


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
    """Value-returning GDN decode implementation for `gdn_decode_qk4_v8_d128_k_last`."""
    batch_size, seq_len, _, _ = q.shape
    if seq_len != 1:
        raise ValueError(f"gdn_decode_kernel expects seq_len == 1, got {seq_len}")

    scale_value = _normalize_scale(scale)
    q_exp = _expand_q_heads(q.squeeze(1))
    k_exp = _expand_k_heads(k.squeeze(1))
    v_f = v.squeeze(1).float()
    g, beta = _compute_gate_and_beta(A_log, a, dt_bias, b)
    state_hkv = _init_state_hkv(state, batch_size, q.device)
    output, new_state_hkv = _gdn_step_decode(
        q_exp,
        k_exp,
        v_f,
        state_hkv,
        g.squeeze(1),
        beta.squeeze(1),
        scale_value,
    )
    return output.unsqueeze(1).to(q.dtype), new_state_hkv.transpose(-1, -2).contiguous()


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
    """Value-returning GDN prefill implementation for `gdn_prefill_qk4_v8_d128_k_last`."""
    total_seq_len = q.shape[0]
    num_seqs = cu_seqlens.numel() - 1
    scale_value = _normalize_scale(scale)

    q_exp = _expand_q_heads(q)
    k_exp = _expand_k_heads(k)
    v_f = v.float()
    g, beta = _compute_gate_and_beta(A_log, a, dt_bias, b)
    state_hkv = _init_state_hkv(state, num_seqs, q.device)

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(dtype=torch.int64)
    max_seq_len = int(lengths.max().item()) if num_seqs > 0 else 0
    order = torch.argsort(lengths, descending=True)
    lengths_sorted = lengths.index_select(0, order)
    state_hkv = state_hkv.index_select(0, order)

    padded_q = torch.zeros(num_seqs, max_seq_len, NUM_V_HEADS, HEAD_SIZE, device=q.device, dtype=torch.float32)
    padded_k = torch.zeros_like(padded_q)
    padded_v = torch.zeros_like(padded_q)
    padded_g = torch.zeros(num_seqs, max_seq_len, NUM_V_HEADS, device=q.device, dtype=torch.float32)
    padded_beta = torch.zeros_like(padded_g)

    cu_list = cu_seqlens.to(dtype=torch.int64).cpu().tolist()
    order_list = order.cpu().tolist()
    lengths_sorted_list = lengths_sorted.cpu().tolist()
    for sorted_idx, seq_idx in enumerate(order_list):
        start = cu_list[seq_idx]
        end = cu_list[seq_idx + 1]
        seq_len = end - start
        if seq_len <= 0:
            continue
        padded_q[sorted_idx, :seq_len] = q_exp[start:end]
        padded_k[sorted_idx, :seq_len] = k_exp[start:end]
        padded_v[sorted_idx, :seq_len] = v_f[start:end]
        padded_g[sorted_idx, :seq_len] = g[start:end]
        padded_beta[sorted_idx, :seq_len] = beta[start:end]

    output_sorted = torch.zeros_like(padded_v)
    active_count = num_seqs
    for t in range(max_seq_len):
        while active_count > 0 and lengths_sorted_list[active_count - 1] <= t:
            active_count -= 1
        if active_count == 0:
            break
        output_t, new_state = _gdn_step(
            padded_q[:active_count, t],
            padded_k[:active_count, t],
            padded_v[:active_count, t],
            state_hkv[:active_count],
            padded_g[:active_count, t],
            padded_beta[:active_count, t],
            scale_value,
        )
        output_sorted[:active_count, t] = output_t
        state_hkv[:active_count] = new_state

    output = torch.empty(total_seq_len, NUM_V_HEADS, HEAD_SIZE, device=q.device, dtype=torch.float32)
    final_state = torch.empty(num_seqs, NUM_V_HEADS, HEAD_SIZE, HEAD_SIZE, device=q.device, dtype=torch.float32)
    for sorted_idx, seq_idx in enumerate(order_list):
        start = cu_list[seq_idx]
        end = cu_list[seq_idx + 1]
        seq_len = end - start
        if seq_len > 0:
            output[start:end] = output_sorted[sorted_idx, :seq_len]
        final_state[seq_idx] = state_hkv[sorted_idx].transpose(-1, -2)

    return output.to(q.dtype), final_state.contiguous()


# Default alias for ad-hoc imports.
kernel = gdn_decode_kernel
