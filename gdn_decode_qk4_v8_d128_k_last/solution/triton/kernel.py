"""
Triton GDN Decode Kernel - gdn_decode_qk4_v8_d128_k_last

Gated Delta Net decode with GVA configuration and k-last state layout.
Single-token generation with recurrent state update.

Config: num_q_heads=4, num_k_heads=4, num_v_heads=8, head_size=128
State layout: k-last [B, HV, V, K]

Algorithm per (batch, v_head):
  1. g = exp(-exp(A_log) * softplus(a + dt_bias))  -- decay factor
  2. beta = sigmoid(b)
  3. h *= g                  -- decay state
  4. pred = h @ k            -- [V,K] @ [K] -> [V]
  5. v_new = (v - pred)*beta -- delta rule + gate
  6. h += outer(v_new, k)    -- rank-1 update
  7. out = scale * h @ q     -- readout
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_decode_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr,
    output_ptr, new_state_ptr,
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    scale,
    B, H: tl.constexpr, HV: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    num_v_tiles = V // BLOCK_V
    i_v = pid % num_v_tiles
    tmp = pid // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV

    i_h = i_hv * H // HV

    A_log_val = tl.load(A_log_ptr + i_hv)
    dt_bias_val = tl.load(dt_bias_ptr + i_hv)
    a_val = tl.load(a_ptr + i_n * HV + i_hv).to(tl.float32)
    b_val = tl.load(b_ptr + i_n * HV + i_hv).to(tl.float32)

    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= 20.0, tl.math.log(1.0 + tl.math.exp(x)), x)
    g = -tl.math.exp(A_log_val) * softplus_x
    decay = tl.math.exp(g)
    beta = 1.0 / (1.0 + tl.math.exp(-b_val))

    k_offs = tl.arange(0, K)
    qk_base = i_n * H * K + i_h * K
    q_vec = tl.load(q_ptr + qk_base + k_offs, eviction_policy="evict_last").to(tl.float32)
    k_vec = tl.load(k_ptr + qk_base + k_offs, eviction_policy="evict_last").to(tl.float32)

    v_range = i_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_base = i_n * HV * V + i_hv * V
    v_vals = tl.load(v_ptr + v_base + v_range).to(tl.float32)

    s_base = (i_n * HV + i_hv) * V * K
    s_offs = v_range[:, None] * K + k_offs[None, :]
    h = tl.load(state_ptr + s_base + s_offs)

    h = h * decay

    pred = tl.sum(h * k_vec[None, :], axis=1)

    v_new = (v_vals - pred) * beta

    h = h + v_new[:, None] * k_vec[None, :]

    out = tl.sum(h * q_vec[None, :], axis=1) * scale

    o_base = (i_n * HV + i_hv) * V
    tl.store(output_ptr + o_base + v_range, out.to(tl.bfloat16))

    ns_base = (i_n * HV + i_hv) * V * K
    tl.store(new_state_ptr + ns_base + s_offs, h)


def kernel(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    B_size = q.shape[0]
    H = q.shape[2]
    K = q.shape[3]
    HV = v.shape[2]
    V = v.shape[3]

    if isinstance(scale, torch.Tensor):
        scale = scale.item()

    BLOCK_V = 8
    num_v_tiles = V // BLOCK_V
    grid = (B_size * HV * num_v_tiles,)

    q_c = q.reshape(B_size, H, K).contiguous()
    k_c = k.reshape(B_size, H, K).contiguous()
    v_c = v.reshape(B_size, HV, V).contiguous()
    a_c = a.reshape(B_size, HV).contiguous()
    b_c = b.reshape(B_size, HV).contiguous()

    _gdn_decode_kernel[grid](
        q_c, k_c, v_c, state,
        output.view(B_size, HV, V), new_state,
        A_log, a_c, dt_bias, b_c,
        scale,
        B_size, H, HV, K, V,
        BLOCK_V,
        num_warps=1,
    )
