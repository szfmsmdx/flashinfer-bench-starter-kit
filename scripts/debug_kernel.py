"""
Debug script: compare intermediate values between reference and kernel.
"""
import torch
import torch.nn.functional as F
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(PROJECT_ROOT, "workspace", "triton_cache"))

from solution.triton.kernel import (
    gdn_decode_kernel,
    gdn_decode_reference,
    _compute_gate_and_beta,
    _expand_qk_heads,
    _init_state_vk,
)

NUM_Q_HEADS, NUM_K_HEADS, NUM_V_HEADS = 4, 4, 8
HEAD_SIZE = 128
Q_GROUP_SIZE = NUM_V_HEADS // NUM_Q_HEADS   # 2
K_GROUP_SIZE = NUM_V_HEADS // NUM_K_HEADS   # 2

torch.manual_seed(42)

B = 1
q = torch.randn(B, 1, NUM_Q_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, 1, NUM_K_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, 1, NUM_V_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device="cuda")
state = torch.randn(B, NUM_V_HEADS, HEAD_SIZE, HEAD_SIZE, dtype=torch.float32, device="cuda")
A_log = torch.randn(NUM_V_HEADS, dtype=torch.float32, device="cuda")
a = torch.randn(B, 1, NUM_V_HEADS, dtype=torch.bfloat16, device="cuda")
dt_bias = torch.randn(NUM_V_HEADS, dtype=torch.float32, device="cuda")
b = torch.randn(B, 1, NUM_V_HEADS, dtype=torch.bfloat16, device="cuda")
scale = torch.tensor(1.0 / math.sqrt(HEAD_SIZE), dtype=torch.float32)

# Run both
out_ref, state_ref = gdn_decode_reference(q, k, v, state.clone(), A_log, a, dt_bias, b, scale)
out_kern, state_kern = gdn_decode_kernel(q, k, v, state.clone(), A_log, a, dt_bias, b, scale)

print("=== OUTPUT ===")
print(f"out_ref shape: {out_ref.shape}")
print(f"out_kern shape: {out_kern.shape}")
print(f"out_ref max: {out_ref.float().abs().max().item():.6f}")
print(f"out_kern max: {out_kern.float().abs().max().item():.6f}")
print(f"MaxAbsErr: {(out_ref.float() - out_kern.float()).abs().max().item():.6f}")

# Check a specific element
print(f"\nout_ref[0,0,0,:8]: {out_ref[0,0,0,:8].float().tolist()}")
print(f"out_kern[0,0,0,:8]: {out_kern[0,0,0,:8].float().tolist()}")

print("\n=== STATE ===")
print(f"state_ref shape: {state_ref.shape}")
print(f"state_kern shape: {state_kern.shape}")
print(f"state_ref max: {state_ref.abs().max().item():.6f}")
print(f"state_kern max: {state_kern.abs().max().item():.6f}")
print(f"MaxAbsErr: {(state_ref - state_kern).abs().max().item():.6f}")

# Detailed per-head comparison
print("\n=== PER-HEAD comparison ===")
for h in range(NUM_V_HEADS):
    err = (out_ref[0,0,h] - out_kern[0,0,h]).float().abs().max().item()
    print(f"  head {h}: max err = {err:.6f}")

# Intermediate values comparison
print("\n=== Intermediate gate/beta ===")
q_hk = _expand_qk_heads(q.squeeze(1), Q_GROUP_SIZE)
k_hk = _expand_qk_heads(k.squeeze(1), K_GROUP_SIZE)
v_hv = v.squeeze(1).float()
gate, beta = _compute_gate_and_beta(A_log, a, dt_bias, b)
print(f"q_hk shape: {q_hk.shape}")
print(f"k_hk shape: {k_hk.shape}")
print(f"v_hv shape: {v_hv.shape}")
print(f"gate shape: {gate.shape}, first 4: {gate[0,:4].tolist()}")
print(f"beta shape: {beta.shape}, first 4: {beta[0,:4].tolist()}")

# Check what reference does step by step
print("\n=== Reference step by step (B=0, H=0) ===")
x = a.float() + dt_bias.float()
g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
b_ref = torch.sigmoid(b.float())
q_exp = q.squeeze(1).float().repeat_interleave(2, dim=1)
k_exp = k.squeeze(1).float().repeat_interleave(2, dim=1)
print(f"gate[0,0]: {gate[0,0].item():.6f} vs ref: {g[0,0].item():.6f} match={torch.allclose(gate, g)}")
print(f"beta[0,0]: {beta[0,0].item():.6f} vs ref: {b_ref[0,0].item():.6f} match={torch.allclose(beta, b_ref)}")

# Reference state step
h_state_ref = state_ref[0, 0].clone().transpose(-1, -2)  # [V,K] -> [K,V]
h_state_kern = state_kern[0, 0].clone()  # [V,K]
print(f"\nstate_ref[0,0] (kern style [V,K]):")
print(f"  shape: {h_state_kern.shape}")
print(f"  first row: {h_state_kern[0,:8].tolist()}")
print(f"state_ref[0,0] (ref style [K,V]):")
print(f"  shape: {h_state_ref.shape}")
print(f"  first col: {h_state_ref[:8,0].tolist()}")
print(f"  state_ref[0,0,K=0,V=0] = {state_ref[0,0,0,0].item():.6f}")
print(f"  state_ref[0,0,V=0,K=0] = {state_ref[0,0,0,0].item():.6f}")

# Check v loading
print(f"\nv input: {v[0,0,0,:4].bfloat16().tolist()}")
print(f"v_hv: {v_hv[0,0,:4].tolist()}")

# Check q loading
print(f"\nq input: {q[0,0,0,:4].bfloat16().tolist()}")
print(f"q_hk (head 0): {q_hk[0,0,:4].tolist()}")
print(f"q_hk (head 1): {q_hk[0,1,:4].tolist()}")
