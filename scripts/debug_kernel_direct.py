"""
Direct kernel-replication debug: reproduce the kernel's exact data flow,
then compare with reference, to find where the 2.69 error comes from.
"""
import torch
import math

torch.manual_seed(42)
device = "cuda"
dtype_bf = torch.bfloat16
dtype_f = torch.float32

# Same dims as kernel
B, HV, V, K = 1, 8, 128, 128
BLOCK_V = 64
BLOCK_K = 64
NUM_V_HEADS = 8
HEAD_SIZE = 128

# Input data (same as debug_kernel.py)
q = torch.randn(B, 1, 4, HEAD_SIZE, dtype=dtype_bf, device=device)
k = torch.randn(B, 1, 4, HEAD_SIZE, dtype=dtype_bf, device=device)
v = torch.randn(B, 1, 8, HEAD_SIZE, dtype=dtype_bf, device=device)
state = torch.randn(B, 8, HEAD_SIZE, HEAD_SIZE, dtype=dtype_f, device=device)
A_log = torch.randn(8, dtype=dtype_f, device=device)
a = torch.randn(B, 1, 8, dtype=dtype_bf, device=device)
dt_bias = torch.randn(8, dtype=dtype_f, device=device)
b = torch.randn(B, 1, 8, dtype=dtype_bf, device=device)
scale = 1.0 / math.sqrt(HEAD_SIZE)

Q_GROUP_SIZE = 2
K_GROUP_SIZE = 2

# Exactly replicate _gdn_decode_triton pre-processing
q_hk = q.squeeze(1).float().repeat_interleave(Q_GROUP_SIZE, dim=1)
k_hk = k.squeeze(1).float().repeat_interleave(K_GROUP_SIZE, dim=1)
v_hv = v.squeeze(1).float()

import torch.nn.functional as F
x = a.float() + dt_bias.float()
gate = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
beta = torch.sigmoid(b.float())

state_hvk = state.clone().contiguous()

# Flatten exactly like the kernel wrapper does
q_flat = q_hk.reshape(B * HV, K).contiguous()
k_flat = k_hk.reshape(B * HV, K).contiguous()
v_flat = v_hv.reshape(B * HV, V).contiguous()
state_flat = state_hvk.reshape(B * HV, V, K).contiguous()
gate_flat = gate.reshape(B * HV).contiguous()
beta_flat = beta.reshape(B * HV).contiguous()

print(f"q_flat shape: {q_flat.shape}")
print(f"state_flat shape: {state_flat.shape}")
print(f"state_flat stride: {state_flat.stride()}")

# Kernel-replicated computation for each program instance
# Grid: (B * NUM_V_HEADS, ceil(HEAD_SIZE / BLOCK_V)) = (8, 2)
# Each pid_bh = (batch_idx * NUM_V_HEADS + head_idx) = head_idx (B=1)
# pid_v = 0 or 1

out_flat = torch.empty_like(v_flat)
new_state_flat = torch.empty_like(state_flat)

for pid_bh in range(B * HV):
    for pid_v in range(math.ceil(HEAD_SIZE / BLOCK_V)):
        batch_idx = pid_bh // NUM_V_HEADS
        head_idx = pid_bh % NUM_V_HEADS

        v_start = pid_v * BLOCK_V
        v_end = min(v_start + BLOCK_V, HEAD_SIZE)
        v_mask = v_end <= HEAD_SIZE

        q_base = q_flat[pid_bh]      # [K]
        k_base = k_flat[pid_bh]      # [K]
        v_base = v_flat[pid_bh]      # [V]
        state_base = state_flat[pid_bh]  # [V, K]
        gate_val = gate_flat[pid_bh].item()
        beta_val = beta_flat[pid_bh].item()

        # --- Load Q and K vectors ---
        # In Triton: offs_k = tl.arange(0, BLOCK_K), k_mask = offs_k < HEAD_SIZE_CONST
        offs_k = torch.arange(0, BLOCK_K, device=device)
        k_mask = offs_k < HEAD_SIZE
        q_vec = q_base[offs_k].masked_fill(~k_mask, 0.0)

        # --- First K-loop ---
        qk_acc = 0.0
        old_v_acc = torch.zeros(BLOCK_V, dtype=dtype_f, device=device)

        for k_offset in range(0, HEAD_SIZE, BLOCK_K):
            offs_k_tile = k_offset + offs_k
            mask_k_tile = offs_k_tile < HEAD_SIZE

            q_tile = q_base[offs_k_tile].masked_fill(~mask_k_tile, 0.0)
            k_tile = k_base[offs_k_tile].masked_fill(~mask_k_tile, 0.0)
            qk_tile = (q_tile * k_tile).sum()
            qk_acc += qk_tile.item()

            # state tile [BLOCK_V × BLOCK_K]
            state_tile = state_base[v_start:v_end, k_offset:k_offset + BLOCK_K]
            state_tile_masked = state_tile.clone()
            # Pad with zeros if needed
            if state_tile.shape[1] < BLOCK_K:
                pad = torch.zeros(BLOCK_V, BLOCK_K - state_tile.shape[1], dtype=dtype_f, device=device)
                state_tile_masked = torch.cat([state_tile, pad], dim=1)
            state_tile_gated = state_tile_masked * gate_val

            k_tile_for_matvec = k_base[k_offset:k_offset + BLOCK_K]
            if k_tile_for_matvec.shape[0] < BLOCK_K:
                pad = torch.zeros(BLOCK_K - k_tile_for_matvec.shape[0], dtype=dtype_f, device=device)
                k_tile_for_matvec = torch.cat([k_tile_for_matvec, pad])

            old_v_tile = state_tile_gated @ k_tile_for_matvec
            old_v_acc += old_v_tile

        print(f"\npid_bh={pid_bh}, pid_v={pid_v}")
        print(f"  qk_acc = {qk_acc:.6f}")
        print(f"  old_v_acc[:4] = {old_v_acc[:4].tolist()}")

        # --- delta_v ---
        v_tile = v_base[v_start:v_end]
        delta_v = (v_tile - old_v_acc) * beta_val
        print(f"  delta_v[:4] = {delta_v[:4].tolist()}")

        # --- Second K-loop ---
        q_old_acc = torch.zeros(BLOCK_V, dtype=dtype_f, device=device)

        for k_offset in range(0, HEAD_SIZE, BLOCK_K):
            offs_k_tile = k_offset + offs_k
            mask_k_tile = offs_k_tile < HEAD_SIZE

            k_tile = k_base[offs_k_tile].masked_fill(~mask_k_tile, 0.0)

            state_tile = state_base[v_start:v_end, k_offset:k_offset + BLOCK_K]
            state_tile_masked = state_tile.clone()
            if state_tile.shape[1] < BLOCK_K:
                pad = torch.zeros(BLOCK_V, BLOCK_K - state_tile.shape[1], dtype=dtype_f, device=device)
                state_tile_masked = torch.cat([state_tile, pad], dim=1)
            state_tile_gated = state_tile_masked * gate_val

            q_old_tile = state_tile_gated @ q_vec
            q_old_acc += q_old_tile

            # new_state
            k_tile_for_state = k_base[k_offset:k_offset + BLOCK_K]
            if k_tile_for_state.shape[0] < BLOCK_K:
                pad = torch.zeros(BLOCK_K - k_tile_for_state.shape[0], dtype=dtype_f, device=device)
                k_tile_for_state = torch.cat([k_tile_for_state, pad])

            new_state_tile = state_tile_gated + delta_v[:, None] * k_tile_for_state[None, :]

            if v_end - v_start == BLOCK_V:
                new_state_flat[pid_bh, v_start:v_end, k_offset:k_offset + BLOCK_K] = new_state_tile
            else:
                new_state_flat[pid_bh, v_start:v_end, k_offset:k_offset + BLOCK_K] = new_state_tile[:v_end - v_start]

        print(f"  q_old_acc[:4] = {q_old_acc[:4].tolist()}")

        # --- Output ---
        out_block = scale * (q_old_acc + qk_acc * delta_v)
        out_flat[pid_bh, v_start:v_end] = out_block[:v_end - v_start]
        print(f"  out[:4] = {out_block[:4].tolist()}")

# Reference
old_state = state_hvk * gate.unsqueeze(-1).unsqueeze(-1)
old_v_ref = torch.einsum("bhvk,bhk->bhv", old_state, k_hk)
delta_v_ref = (v_hv - old_v_ref) * beta.unsqueeze(-1)
new_state_ref = old_state + delta_v_ref.unsqueeze(-1) * k_hk.unsqueeze(-2)
out_ref = scale * torch.einsum("bhvk,bhk->bhv", new_state_ref, q_hk)

print(f"\n=== ERRORS ===")
print(f"out_ref shape: {out_ref.shape}")
print(f"out_flat shape: {out_flat.shape}")
out_err = (out_ref.reshape_as(out_flat) - out_flat).abs().max().item()
print(f"Max output err: {out_err:.6f}")

state_err = (new_state_ref - new_state_flat).abs().max().item()
print(f"Max state err: {state_err:.6f}")

# Per head
for h in range(HV):
    out_h_ref = out_ref[0, h]
    out_h_kern = out_flat[h]
    err = (out_h_ref - out_h_kern).abs().max().item()
    print(f"  Head {h}: output err = {err:.6f}")

# --- Key question: what does q·k look like per tile? ---
print(f"\n=== Q·K breakdown ===")
for h in range(HV):
    qh = q_hk[0, h]
    kh = k_hk[0, h]
    qk_full = torch.dot(qh, kh).item()
    qk_tile0 = torch.dot(qh[:64], kh[:64]).item()
    qk_tile1 = torch.dot(qh[64:], kh[64:]).item()
    print(f"  Head {h}: full={qk_full:.6f}, tile0={qk_tile0:.6f}, tile1={qk_tile1:.6f}, sum={qk_tile0+qk_tile1:.6f}")
