"""
逐行追踪 debug：把 Triton kernel 里的每一行计算都在 PyTorch 里完整模拟，
找 exact first-diff 的位置。
"""
import torch
import torch.nn.functional as F
import math
torch.manual_seed(42)
device = "cuda"
dtype_bf = torch.bfloat16
dtype_f = torch.float32
B, HV, V, K = 1, 8, 128, 128
BLOCK_V = 64
BLOCK_K = 64
# ------------------------------------------------------------------
# Input data
# ------------------------------------------------------------------
q_bhk = torch.randn(B, HV, K, dtype=dtype_bf, device=device).float()
k_bhk = torch.randn(B, HV, K, dtype=dtype_bf, device=device).float()
v_bhv = torch.randn(B, HV, V, dtype=dtype_bf, device=device).float()
state_bhvk = torch.randn(B, HV, V, K, dtype=dtype_f, device=device)
gate_bhv = torch.rand(B, HV, dtype=dtype_f, device=device)
beta_bhv = torch.rand(B, HV, dtype=dtype_f, device=device)
scale = 0.08838834764831843

# Simulate what the Triton kernel does for a single program instance
# pid_bh = 0 (batch=0, head=0), pid_v = 0 (v_start=0)
batch_idx, head_idx = 0, 0
pid_v = 0
v_start = pid_v * BLOCK_V  # 0
v_end = v_start + BLOCK_V   # 64
v_mask = v_end <= V          # True
# Gate and beta for this head
g_val = gate_bhv[batch_idx, head_idx].item()
b_val = beta_bhv[batch_idx, head_idx].item()
# Q, K vectors (full K=128)
q_vec = q_bhk[batch_idx, head_idx]          # [K]
k_vec = k_bhk[batch_idx, head_idx]          # [K]
v_vec = v_bhv[batch_idx, head_idx]          # [V]
# Accumulators
qk_acc = 0.0
old_v_acc = torch.zeros(BLOCK_V, dtype=dtype_f, device=device)

# ------------------------------------------------------------------
# Triton K-loop (BLOCK_K = 64, so 2 iterations: k_offset=0, k_offset=64)
# ------------------------------------------------------------------
print("=== K-LOOP (old_v accumulation) ===")
for k_offset in range(0, K, BLOCK_K):
    print(f"\n  k_offset = {k_offset}")
    k_end = min(k_offset + BLOCK_K, K)
    k_slice = slice(k_offset, k_end)
    k_tile = k_vec[k_slice]  # [BLOCK_K]
    # Q·K tile
    qk_tile = torch.dot(q_vec[k_slice], k_tile).item()
    qk_acc += qk_tile
    print(f"  q·k_tile = {qk_tile:.6f}, qk_acc = {qk_acc:.6f}")
    # state tile [BLOCK_V × BLOCK_K], rows v_start:v_end, cols k_offset:k_end
    state_tile = state_bhvk[batch_idx, head_idx, v_start:v_end, k_slice]  # [64, 64]
    state_tile_gated = state_tile * g_val
    print(f"  state_tile mean = {state_tile.mean().item():.6f}, gated mean = {state_tile_gated.mean().item():.6f}")
    # old_v_acc += state_tile_gated @ k_tile (K-reduced, per V-row)
    old_v_tile = state_tile_gated @ k_tile  # [64] = [64,64] @ [64]
    old_v_acc += old_v_tile
    print(f"  old_v_tile mean = {old_v_tile.mean().item():.6f}, old_v_acc mean = {old_v_acc.mean().item():.6f}")
print(f"\nFinal qk_acc = {qk_acc:.6f}")
print(f"Final old_v_acc (first 8): {old_v_acc[:8].tolist()}")

# ------------------------------------------------------------------
# Reference: what should old_v_acc be?
# ------------------------------------------------------------------
print("\n=== REFERENCE old_v ===")
# state_k_last = state_bhvk[batch_idx, head_idx]  # [V, K] k-last
# Reference: old_state = gate * state_k_last
#            old_v = old_state @ k_vec = (gate * state_k_last) @ k_vec
old_state_ref = gate_bhv[batch_idx, head_idx] * state_bhvk[batch_idx, head_idx]  # [V, K]
old_v_ref = old_state_ref @ k_bhk[batch_idx, head_idx]  # [V]
print(f"old_v_ref (first 8): {old_v_ref[:8].tolist()}")
print(f"old_v_ref mean: {old_v_ref.mean().item():.6f}")
# Are they close?
err_old_v = (old_v_acc - old_v_ref[:BLOCK_V]).abs().max().item()
print(f"Max abs err in old_v (first 64 elements): {err_old_v:.6f}")

# ------------------------------------------------------------------
# Compute delta_v
# ------------------------------------------------------------------
v_tile = v_vec[v_start:v_end]  # [64]
delta_v = (v_tile - old_v_acc) * b_val
print(f"\n=== delta_v ===")
print(f"delta_v (first 8): {delta_v[:8].tolist()}")
delta_v_ref = (v_bhv[batch_idx, head_idx] - old_v_ref) * beta_bhv[batch_idx, head_idx]
print(f"delta_v_ref (first 8): {delta_v_ref[:8].tolist()}")
print(f"delta_v_ref mean: {delta_v_ref.mean().item():.6f}")
err_delta_v = (delta_v - delta_v_ref[:BLOCK_V]).abs().max().item()
print(f"Max abs err in delta_v: {err_delta_v:.6f}")

# ------------------------------------------------------------------
# Second K-loop: q_old accumulation + new_state write
# ------------------------------------------------------------------
print("\n=== K-LOOP 2 (q_old + new_state) ===")
q_old_acc = torch.zeros(BLOCK_V, dtype=dtype_f, device=device)
for k_offset in range(0, K, BLOCK_K):
    k_end = min(k_offset + BLOCK_K, K)
    k_slice = slice(k_offset, k_end)
    state_tile = state_bhvk[batch_idx, head_idx, v_start:v_end, k_slice]  # [64, 64]
    state_tile_gated = state_tile * g_val
    # q_old_acc += state_tile_gated @ q_vec (K-reduced, per V-row)
    q_old_tile = state_tile_gated @ q_vec[k_slice]  # [64]
    q_old_acc += q_old_tile
print(f"q_old_acc (first 8): {q_old_acc[:8].tolist()}")
q_old_ref = old_state_ref @ q_bhk[batch_idx, head_idx]  # [V]
print(f"q_old_ref (first 8): {q_old_ref[:8].tolist()}")
err_q_old = (q_old_acc - q_old_ref[:BLOCK_V]).abs().max().item()
print(f"Max abs err in q_old: {err_q_old:.6f}")

# ------------------------------------------------------------------
# Output computation
# ------------------------------------------------------------------
print("\n=== OUTPUT ===")
out_block = scale * (q_old_acc + qk_acc * delta_v)
out_ref_head = scale * (q_old_ref + qk_acc * delta_v_ref)
print(f"out_block (first 8): {out_block[:8].tolist()}")
print(f"out_ref_head (first 8): {out_ref_head[:8].tolist()}")
err_out = (out_block - out_ref_head[:BLOCK_V]).abs().max().item()
print(f"Max abs err in output: {err_out:.6f}")

# ------------------------------------------------------------------
# new_state computation
# ------------------------------------------------------------------
print("\n=== NEW STATE ===")
new_state_block_list = []
for k_offset in range(0, K, BLOCK_K):
    k_end = min(k_offset + BLOCK_K, K)
    k_slice = slice(k_offset, k_end)
    state_tile = state_bhvk[batch_idx, head_idx, v_start:v_end, k_slice]  # [64, 64]
    state_tile_gated = state_tile * g_val
    new_state_tile = state_tile_gated + delta_v[:, None] * k_vec[k_slice]  # [64, 64]
    new_state_block_list.append(new_state_tile)
new_state_block = torch.cat(new_state_block_list, dim=1)  # [64, 128]
print(f"new_state_block shape: {new_state_block.shape}")
# Reference new_state for this head
state_old = state_bhvk[batch_idx, head_idx]  # [V, K] k-last
gated_state_old = gate_bhv[batch_idx, head_idx] * state_old  # [V, K]
delta_v_2d = (v_bhv[batch_idx, head_idx] - (gated_state_old @ k_bhk[batch_idx, head_idx]))[:, None] * beta_bhv[batch_idx, head_idx]
new_state_ref_head = gated_state_old + delta_v_2d * k_bhk[batch_idx, head_idx][None, :]  # [V, K]
print(f"new_state_ref_head shape: {new_state_ref_head.shape}")
print(f"new_state_ref_head (v=0, k=0..7): {new_state_ref_head[0, :8].tolist()}")
print(f"new_state_block    (v=0, k=0..7): {new_state_block[0, :8].tolist()}")
err_state = (new_state_block - new_state_ref_head[:BLOCK_V, :]).abs().max().item()
print(f"Max abs err in new_state (first 64 V rows): {err_state:.6f}")

# ------------------------------------------------------------------
# Full end-to-end check
# ------------------------------------------------------------------
print("\n\n=== FULL END-TO-END (all heads, all V) ===")
for h in range(HV):
    b_idx, h_idx = 0, h
    g_v = gate_bhv[b_idx, h_idx].item()
    b_v = beta_bhv[b_idx, h_idx].item()

    # Kernel-style: two K-loops
    old_v_acc_h = torch.zeros(V, dtype=dtype_f, device=device)
    for k_offset in range(0, K, BLOCK_K):
        k_end = min(k_offset + BLOCK_K, K)
        state_tile = state_bhvk[b_idx, h_idx, :, k_offset:k_end]
        gated = state_tile * g_v
        old_v_acc_h += gated @ k_bhk[b_idx, h_idx, k_offset:k_end]
    v_full = v_bhv[b_idx, h_idx]
    dv = (v_full - old_v_acc_h) * b_v
    q_old_acc_h = torch.zeros(V, dtype=dtype_f, device=device)
    for k_offset in range(0, K, BLOCK_K):
        k_end = min(k_offset + BLOCK_K, K)
        state_tile = state_bhvk[b_idx, h_idx, :, k_offset:k_end]
        gated = state_tile * g_v
        q_old_acc_h += gated @ q_bhk[b_idx, h_idx, k_offset:k_end]
    qk_full = torch.dot(q_bhk[b_idx, h_idx], k_bhk[b_idx, h_idx]).item()
    out_h = scale * (q_old_acc_h + qk_full * dv)
    
    # Reference
    old_s = g_v * state_bhvk[b_idx, h_idx]
    old_v_ref_h = old_s @ k_bhk[b_idx, h_idx]
    dv_ref = (v_full - old_v_ref_h) * b_v
    q_old_ref_h = old_s @ q_bhk[b_idx, h_idx]
    out_ref_h = scale * (q_old_ref_h + qk_full * dv_ref)
    err = (out_h - out_ref_h).abs().max().item()
    print(f"  Head {h}: max err = {err:.8f}")

print("\n=== KEY QUESTION: what does q_old_acc_h look like vs q_old_ref_h ===")
print(f"q_old_acc_h (kernel): {q_old_acc_h[:8].tolist()}")
print(f"q_old_ref_h (ref):     {q_old_ref_h[:8].tolist()}")
print(f"Difference: {(q_old_acc_h - q_old_ref_h)[:8].tolist()}")
print("\n=== KEY QUESTION 2: old_v_acc_h vs old_v_ref_h ===")
print(f"old_v_acc_h (kernel): {old_v_acc_h[:8].tolist()}")
print(f"old_v_ref_h (ref):    {old_v_ref_h[:8].tolist()}")
print(f"Difference: {(old_v_acc_h - old_v_ref_h)[:8].tolist()}")