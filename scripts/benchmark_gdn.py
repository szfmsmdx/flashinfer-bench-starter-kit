"""
Standalone GDN benchmark runner.

Generates synthetic inputs matching the dataset definition shapes
(gdn_decode_qk4_v8_d128_k_last) and runs the Triton kernel vs reference.
"""

import os
import sys
import json
import math
import time
import gc
import torch
import torch.nn.functional as F
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Triton kernel
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(PROJECT_ROOT, "workspace", "triton_cache"))

from solution.triton.kernel import (
    gdn_decode_kernel,
    gdn_decode_reference,
    gdn_prefill_kernel,
    gdn_prefill_reference,
)


# ==============================================================================
# Shape constants (Qwen3-Next GVA, TP=4)
# ==============================================================================
NUM_Q_HEADS = 4
NUM_K_HEADS = 4
NUM_V_HEADS = 8
HEAD_SIZE = 128
Q_GROUP_SIZE = NUM_V_HEADS // NUM_Q_HEADS   # 2
K_GROUP_SIZE = NUM_V_HEADS // NUM_K_HEADS   # 2

# State layout: [B, HV, V, K] (k-last)
STATE_SHAPE_FN = lambda B: (B, NUM_V_HEADS, HEAD_SIZE, HEAD_SIZE)

# ==============================================================================
# Reference implementation (exact copy from flashinfer_bench dataset definition)
# ==============================================================================
def reference_run(q, k, v, state, A_log, a, dt_bias, b, scale):
    """
    Gated Delta Net decode reference implementation (k-last layout).
    State layout: [B, H, V, K] (k-last, K dimension at the end)

    Gate computation:
    g = exp(-exp(A_log) * softplus(a + dt_bias))
    beta = sigmoid(b)

    Delta rule update:
    state_new = g * state_old + k^T @ (beta * v + (1-beta) * k @ state_old) - k^T @ (k @ state_old)
    output = scale * q @ state_new
    """
    B, T, num_q_heads, K = q.shape
    _, _, num_k_heads, _ = k.shape
    _, _, num_v_heads, V = v.shape
    num_heads = num_v_heads
    device = q.device

    assert num_q_heads == 4
    assert num_k_heads == 4
    assert num_v_heads == 8
    assert K == 128 and V == 128
    assert T == 1

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    # Compute g and beta from raw parameters
    x = a.float() + dt_bias.float()  # [B, 1, HV]
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [B, 1, HV]
    beta = torch.sigmoid(b.float())  # [B, 1, HV]

    q_f32 = q.squeeze(1).float()
    k_f32 = k.squeeze(1).float()
    v_f32 = v.squeeze(1).float()
    g_f32 = g.squeeze(1).float()
    beta_f32 = beta.squeeze(1).float()

    if state is not None:
        state_f32 = state.float()
    else:
        state_f32 = torch.zeros(B, num_heads, V, K, dtype=torch.float32, device=device)

    q_exp = q_f32.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k_f32.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    new_state = torch.zeros_like(state_f32)
    output = torch.zeros(B, num_heads, V, dtype=torch.float32, device=device)

    for b_idx in range(B):
        for h_idx in range(num_heads):
            q_h = q_exp[b_idx, h_idx]
            k_h = k_exp[b_idx, h_idx]
            v_h = v_f32[b_idx, h_idx]
            h_state = state_f32[b_idx, h_idx].clone().transpose(-1, -2)  # [V,K] -> [K,V]
            g_val = g_f32[b_idx, h_idx]
            beta_val = beta_f32[b_idx, h_idx]

            old_state = g_val * h_state
            old_v = k_h @ old_state
            new_v = beta_val * v_h + (1 - beta_val) * old_v
            state_remove = k_h.unsqueeze(1) @ old_v.unsqueeze(0)
            state_update = k_h.unsqueeze(1) @ new_v.unsqueeze(0)
            h_state = old_state - state_remove + state_update

            output[b_idx, h_idx] = scale * (q_h @ h_state)
            new_state[b_idx, h_idx] = h_state.transpose(-1, -2)  # [K,V] -> [V,K]

    output = output.unsqueeze(1).to(q.dtype)
    return output, new_state


# ==============================================================================
# Input generation
# ==============================================================================
def generate_inputs(batch_size: int, dtype=torch.bfloat16, device="cuda"):
    """Generate synthetic inputs matching the definition shapes."""
    torch.manual_seed(42)

    q = torch.randn(batch_size, 1, NUM_Q_HEADS, HEAD_SIZE, dtype=dtype, device=device)
    k = torch.randn(batch_size, 1, NUM_K_HEADS, HEAD_SIZE, dtype=dtype, device=device)
    v = torch.randn(batch_size, 1, NUM_V_HEADS, HEAD_SIZE, dtype=dtype, device=device)
    state = torch.randn(batch_size, NUM_V_HEADS, HEAD_SIZE, HEAD_SIZE, dtype=torch.float32, device=device)

    # Learned parameters (normally from safetensors, here synthesized)
    A_log = torch.randn(NUM_V_HEADS, dtype=torch.float32, device=device)
    a = torch.randn(batch_size, 1, NUM_V_HEADS, dtype=dtype, device=device)
    dt_bias = torch.randn(NUM_V_HEADS, dtype=torch.float32, device=device)
    b = torch.randn(batch_size, 1, NUM_V_HEADS, dtype=dtype, device=device)
    scale = torch.tensor(1.0 / math.sqrt(HEAD_SIZE), dtype=torch.float32)

    return q, k, v, state, A_log, a, dt_bias, b, scale


# ==============================================================================
# Benchmark harness
# ==============================================================================
def benchmark_fn(fn, q, k, v, state, A_log, a, dt_bias, b, scale, warmup=10, iters=100):
    """Benchmark a kernel function, returns (mean_ms, std_ms)."""
    for _ in range(warmup):
        fn(q, k, v, state, A_log, a, dt_bias, b, scale)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        out, new_state = fn(q, k, v, state, A_log, a, dt_bias, b, scale)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    return sum(times) / len(times), torch.std(torch.tensor(times)).item()


def compute_memory_mb():
    """Peak allocated GPU memory in MB."""
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def run_decode_benchmark(batch_sizes, dtype=torch.bfloat16):
    """Run decode benchmarks across batch sizes."""
    results = []

    for B in batch_sizes:
        print(f"\n{'='*60}")
        print(f"  Decode Benchmark  B={B}")
        print(f"{'='*60}")

        q, k, v, state, A_log, a, dt_bias, b, scale = generate_inputs(B, dtype)
        torch.cuda.reset_peak_memory_stats()

        # --- Reference ---
        ref_mean, ref_std = benchmark_fn(
            lambda *args: reference_run(*args), q, k, v, state.clone(),
            A_log, a, dt_bias, b, scale
        )
        ref_mem = compute_memory_mb()

        # --- Kernel ---
        torch.cuda.reset_peak_memory_stats()
        kernel_mean, kernel_std = benchmark_fn(
            gdn_decode_kernel, q, k, v, state.clone(),
            A_log, a, dt_bias, b, scale
        )
        kernel_mem = compute_memory_mb()

        # --- Correctness ---
        out_ref, state_ref = reference_run(
            q, k, v, state.clone(),
            A_log, a, dt_bias, b, scale
        )
        out_kern, state_kern = gdn_decode_kernel(
            q, k, v, state.clone(),
            A_log, a, dt_bias, b, scale
        )
        max_abs = (out_ref.float() - out_kern.float()).abs().max().item()
        max_rel = ((out_ref.float() - out_kern.float()) / (out_ref.float().abs() + 1e-8)).abs().max().item()

        speedup = ref_mean / kernel_mean

        print(f"  Reference: {ref_mean:.4f} ± {ref_std:.4f} ms")
        print(f"  Kernel:   {kernel_mean:.4f} ± {kernel_std:.4f} ms")
        print(f"  Speedup:  {speedup:.2f}x")
        print(f"  MaxAbsErr: {max_abs:.2e}, MaxRelErr: {max_rel:.2e}")
        print(f"  PeakVRAM: {max(ref_mem, kernel_mem):.1f} MB")

        results.append({
            "batch_size": B,
            "ref_ms": round(ref_mean, 4),
            "ref_std": round(ref_std, 4),
            "kernel_ms": round(kernel_mean, 4),
            "kernel_std": round(kernel_std, 4),
            "speedup": round(speedup, 4),
            "max_abs_err": max_abs,
            "max_rel_err": max_rel,
            "peak_vram_mb": round(max(ref_mem, kernel_mem), 1),
        })

    return results


def run_prefill_benchmark(batch_sizes, seq_lens, dtype=torch.bfloat16):
    """Run prefill benchmarks."""
    results = []

    for B in batch_sizes:
        for T in seq_lens:
            print(f"\n{'='*60}")
            print(f"  Prefill Benchmark  B={B} T={T}")
            print(f"{'='*60}")

            # Generate prefill tensors: [total_seq, H, K] flattened per sequence
            # cu_seqlens = cumulative sequence lengths
            total_len = B * T
            q = torch.randn(total_len, NUM_Q_HEADS, HEAD_SIZE, dtype=dtype, device="cuda")
            k = torch.randn(total_len, NUM_K_HEADS, HEAD_SIZE, dtype=dtype, device="cuda")
            v = torch.randn(total_len, NUM_V_HEADS, HEAD_SIZE, dtype=dtype, device="cuda")
            state = torch.randn(B, NUM_V_HEADS, HEAD_SIZE, HEAD_SIZE, dtype=torch.float32, device="cuda")
            A_log = torch.randn(NUM_V_HEADS, dtype=torch.float32, device="cuda")
            a = torch.randn(total_len, NUM_V_HEADS, dtype=dtype, device="cuda")
            dt_bias = torch.randn(NUM_V_HEADS, dtype=torch.float32, device="cuda")
            b = torch.randn(total_len, NUM_V_HEADS, dtype=dtype, device="cuda")
            scale = torch.tensor(1.0 / math.sqrt(HEAD_SIZE), dtype=torch.float32)
            cu_seqlens = torch.tensor([i * T for i in range(B + 1)], dtype=torch.int32, device="cuda")

            torch.cuda.reset_peak_memory_stats()

            ref_mean, ref_std = benchmark_fn(
                lambda *args: gdn_prefill_reference(*args),
                q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale
            )
            ref_mem = compute_memory_mb()

            torch.cuda.reset_peak_memory_stats()
            kernel_mean, kernel_std = benchmark_fn(
                gdn_prefill_kernel,
                q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale
            )
            kernel_mem = compute_memory_mb()

            # Correctness
            out_ref, state_ref = gdn_prefill_reference(
                q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale
            )
            out_kern, state_kern = gdn_prefill_kernel(
                q, k, v, state.clone(), A_log, a, dt_bias, b, cu_seqlens, scale
            )
            max_abs = (out_ref.float() - out_kern.float()).abs().max().item()
            max_rel = ((out_ref.float() - out_kern.float()) / (out_ref.float().abs() + 1e-8)).abs().max().item()

            speedup = ref_mean / kernel_mean

            print(f"  Reference: {ref_mean:.4f} ± {ref_std:.4f} ms")
            print(f"  Kernel:   {kernel_mean:.4f} ± {kernel_std:.4f} ms")
            print(f"  Speedup:  {speedup:.2f}x")
            print(f"  MaxAbsErr: {max_abs:.2e}, MaxRelErr: {max_rel:.2e}")
            print(f"  PeakVRAM: {max(ref_mem, kernel_mem):.1f} MB")

            results.append({
                "batch_size": B, "seq_len": T,
                "ref_ms": round(ref_mean, 4),
                "ref_std": round(ref_std, 4),
                "kernel_ms": round(kernel_mean, 4),
                "kernel_std": round(kernel_std, 4),
                "speedup": round(speedup, 4),
                "max_abs_err": max_abs,
                "max_rel_err": max_rel,
                "peak_vram_mb": round(max(ref_mem, kernel_mem), 1),
            })

    return results


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Standalone GDN benchmark")
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="decode")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128")
    parser.add_argument("--seq-lens", type=str, default="4,8,16,32")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print(f"GDN Benchmark — dtype={dtype}, iters={args.iters}, warmup={args.warmup}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"SM: {torch.cuda.get_device_capability(0)}")
    print(f"TRITON_CACHE_DIR: {os.environ.get('TRITON_CACHE_DIR')}")

    torch.cuda.empty_cache()
    gc.collect()

    decode_results = []
    prefill_results = []

    if args.mode in ("decode", "all"):
        decode_results = run_decode_benchmark(batch_sizes, dtype)
        torch.cuda.empty_cache()
        gc.collect()

    if args.mode in ("prefill", "all"):
        prefill_results = run_prefill_benchmark(batch_sizes, seq_lens, dtype)

    # Print summary table
    print("\n\n" + "=" * 80)
    print("  SUMMARY TABLE")
    print("=" * 80)
    print(f"{'ExpID':<20} {'B':>4} {'T':>3} {'Ref(ms)':>8} {'Kern(ms)':>9} {'Speedup':>7} {'MaxAbsErr':>10} {'Kept':>5}")
    print("-" * 80)

    for r in decode_results:
        print(f"baseline_v0_decode    {r['batch_size']:>4} {1:>3} {r['ref_ms']:>8.4f} {r['kernel_ms']:>9.4f} {r['speedup']:>7.2f}x {r['max_abs_err']:>10.2e} {'?':>5}")

    for r in prefill_results:
        print(f"baseline_v0_prefill  {r['batch_size']:>4} {r['seq_len']:>3} {r['ref_ms']:>8.4f} {r['kernel_ms']:>9.4f} {r['speedup']:>7.2f}x {r['max_abs_err']:>10.2e} {'?':>5}")
