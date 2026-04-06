# GDN Kernel Optimization — Program

## Project Overview

**Goal**: Optimize the Gated Delta Network (GDN) Triton kernel for Track C of the MLSys 2026 FlashInfer Kernel Generation Challenge on NVIDIA B200 (Blackwell / sm100) GPUs.

**Competition**: [mlsys26.flashinfer.ai](http://mlsys26.flashinfer.ai/)
**Evaluation**: Speedup over reference Python implementation, arithmetic mean across all workloads.

---

## Environment

### Conda Environment

```bash
conda activate fi-bench
```

| Package | Version |
|---------|---------|
| torch | 2.12.0.dev20260307+cu130 |
| triton | 3.6.0+git9844da95 |
| flashinfer-bench | 0.1.2 |
| flashinfer-python | 0.6.5 |
| CUDA compute | sm100 (Blackwell) |

### GPU Target

- **Competition machine**: NVIDIA B200, sm100, HBM3e 192 GB, 8 TB/s
- **Local dev machine**: NVIDIA Thor, sm_110 (Blackwell-class driver)
- Both are Blackwell arch — all Triton code targets sm_100 / `torch.cuda.get_device_capability() >= (10, 0)`

### Key Paths

| Item | Path |
|------|------|
| Dataset | `/home/mlsys/mlsys26-contest` |
| Dataset env var | `FIB_DATASET_PATH=/home/mlsys/mlsys26-contest` |
| Starter kit | `/home/szf/flashinfer-bench-starter-kit` |
| Active solution | `solution/triton/kernel.py` |
| Config | `config.toml` |
| Local runner | `scripts/run_local.py` |
| Modal runner | `scripts/run_modal.py` |
| History | `history.md` |

---

## Baseline Definition

**Definition name** (decode): `gdn_decode_qk4_v8_d128_k_last`
**Definition name** (prefill): `gdn_prefill_qk4_v8_d128_k_last`

### Decode Shape (Qwen3-Next, GVA, TP=4)

| Tensor | Shape | Dtype |
|--------|-------|-------|
| q | [B, 1, 4, 128] | bfloat16 |
| k | [B, 1, 4, 128] | bfloat16 |
| v | [B, 1, 8, 128] | bfloat16 |
| state | [B, 8, 128, 128] | float32 |
| A_log | [8] | float32 |
| a | [B, 1, 8] | bfloat16 |
| dt_bias | [8] | float32 |
| b | [B, 1, 8] | bfloat16 |
| scale | scalar | float32 |

- GVA: num_q_heads=4, num_k_heads=4, num_v_heads=8
- State layout: **k-last** `[B, HV, V, K]`
- seq_len = 1 (single-token decode)

### Compute Flow

```
gate = exp(-exp(A_log) * softplus(a + dt_bias))      # [B, HV]
beta = sigmoid(b)                                       # [B, HV]
out, new_state = gdn_step(q_expanded, k_expanded, v, state, gate, beta, scale)
```

---

## Reference: FlashInfer GDN Optimization History

Based on studying flashinfer-ai/flashinfer PRs #2370, #2405, #2618, #2619, #2679:

### Key Findings

1. **CuTe DSL + CoopRow BF16 state** (PR #2679): Each warp processes one V-row, V-tile=8, K-tile=128, state stored as BF16 in memory / FP32 in registers. Peak **13.8 TFLOPS** on B200 (vs 7.9 TFLOPS FP32). BS=4-16 → 2.21x speedup over FP32.
2. **Pretranspose kernel** (PR #2370): FlashInfer pretranspose decode is 1.16-1.47x faster than vLLM Triton on B200 across batch sizes 1-512.
3. **cp.async pipeline**: Used in MTP (multi-token prediction) for hiding memory latency.
4. **cudaGetDeviceProperties caching** (PR #2509): Avoid per-call overhead.
5. **Qwen3-Next Triton comparison** (from vadim's benchmarks): Triton in vLLM/SGLang is still competitive; FlashInfer CuTe wins by 1.16-1.47x for decode.

---

## Architecture: Blackwell (sm100) Key Properties

| Property | Value | Implication |
|----------|-------|-------------|
| Shared mem | 228 KB / SM | Can hold larger state tiles |
| Registers | 2048 / SM | More register pressure OK |
| L2 cache | 192 MB (GB300) | Larger than H100's 80 MB |
| Tensor Core | FP8/FP16/BF16/FP32 | BF16 state + FP32 compute ideal |
| Warp size | 32 | Standard |
| Memory BW | 8 TB/s (B200) | ~2.4x H100 |
| SMs | 8 GPCs, ~140 SMs | Massive parallelism |

---

## Optimization Roadmap

### Stage 1 — Quick wins, no kernel rewrite
- [ ] Dynamic BLOCK params based on `torch.cuda.get_device_properties()`
- [ ] Increase `num_warps` (4 → 8), increase `BLOCK_K` (32 → 64)
- [ ] Host-side precomputation: fuse gate/beta computation into Triton kernel or pre-compute before kernel call
- [ ] Remove unnecessary `reshape` / `.contiguous()` calls in hot path
- [ ] Pre-allocate output tensors to avoid per-call allocation

### Stage 2 — Kernel architecture redesign
- [ ] **State tiling with CoopRow**: Multiple warps cooperatively process V dimension, reduce state HBM reads by ~16x
- [ ] **BF16 state storage**: State in BF16 on HBM, FP32 in registers (FlashInfer PR #2679 pattern)
- [ ] **Block layout**: Each program processes 1 batch item × 1 V-tile, iterates over K with static range
- [ ] **ILP (Instruction-Level Parallelism)**: Overlap the two static_range loops for qk and q_old computation

### Stage 3 — Prefill optimization
- [ ] Packed execution for prefill (already in place via length-sorting)
- [ ] Add cp.async pipeline for K/V prefetch
- [ ] Multi-stage Triton pipeline (producer-consumer overlap)

### Stage 4 — CUDA/CuTe DSL (optional if Triton plateaus)
- [ ] Rewrite decode kernel in CuTe DSL
- [ ] Use `cute.compile()` with JIT caching
- [ ] Leverage `cp.async` for state loads

---

## Current Baseline Kernel (`solution/triton/kernel.py`)

The existing kernel (`_gdn_step_kernel`) processes:
- **pid_item × pid_v** = `batch * ceil(HV * V / BLOCK_V)` program instances
- **BLOCK_V = 32, BLOCK_K = 32, num_warps = 4, num_stages = 2**
- Two separate `static_range` loops over K (for qk accumulation + state loads)
- State `[HV, V, K]` read once per K-tile per program instance

**Known bottlenecks**:
1. State matrix `[8, 128, 128]` read twice per step per batch item (once for qk, once for q_old)
2. `gate` and `beta` computed on CPU-side Python before kernel call
3. Fixed block params, no Blackwell-specific tuning
4. `repeat_interleave` + `float()` conversion in `_expand_qk_heads` adds host overhead

---

## Expected Outcomes

| Stage | Target speedup vs baseline |
|-------|---------------------------|
| Stage 1 | +20-40% |
| Stage 2 (CoopRow + BF16 state) | +100-200% |
| Stage 3 (Prefill) | +30-50% |
| Stage 4 (CuTe) | Competitive with FlashInfer |

**Reference**: FlashInfer B200 decode is 1.16-1.47x faster than vLLM Triton baseline.
