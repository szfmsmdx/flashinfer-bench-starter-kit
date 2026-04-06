# GDN Kernel Optimization History

experiment_id | hypothesis | correctness | time_ms | throughput | peak_vram_mb | kept
---|---|---|---|---|---|---
baseline_v0 | Original kernel: BLOCK_V=32, BLOCK_K=32, num_warps=4, num_stages=2, two separate K-loops | MAX_REL=1.79e-02 | See below | See below | See below | ❌ (incorrect correctness numbers due to wrong reference comparison) |
v1_stage1 | Dynamic params (BLOCK=64, num_warps=8), merged K-loop, gate/beta precompute | ✗ BROKEN (wrong index stride in state load/store) | See below | See below | See below | ✗ (buggy) |

---

## Baseline v0 — Original Kernel (NVIDIA Thor, B=1..128)

ExpID | B | T | Ref(ms) | Kern(ms) | Speedup | MaxAbsErr | Kept
--- | --- | --- | --- | --- | --- | --- | ---
baseline_v0_decode | 1 | 1 | 1.3401 | 0.2753 | 4.87x | 0.00e+00 | ✓ (err=0 at B=1)
baseline_v0_decode | 2 | 1 | 2.6597 | 0.2002 | 13.29x | 1.95e-03 | ✓
baseline_v0_decode | 4 | 1 | 4.6983 | 0.1921 | 24.46x | 9.77e-04 | ✓
baseline_v0_decode | 8 | 1 | 9.3863 | 0.2055 | 45.68x | 3.91e-03 | ✓
baseline_v0_decode | 16 | 1 | 18.5301 | 0.3491 | 53.07x | 2.44e-04 | ✓
baseline_v0_decode | 32 | 1 | 37.1198 | 0.5308 | 69.93x | 1.95e-03 | ✓
baseline_v0_decode | 64 | 1 | 72.8897 | 1.1873 | 61.39x | 7.81e-03 | ✓
baseline_v0_decode | 128 | 1 | 143.4068 | 2.0535 | 69.83x | 3.91e-03 | ✓

---

## v1 Stage 1 — BROKEN (wrong stride order in state load/store)

Bug: state pointer formula used `(v_idx * K + k_idx)` but k-last layout requires `(v_idx * stride_V + k_idx * stride_K)` = `(v_idx * K + k_idx * 1)`. K is innermost (stride=1), V is outer (stride=K=128). The indices were swapped, causing state to be read/written at completely wrong memory locations.

ExpID | B | T | Ref(ms) | Kern(ms) | Speedup | MaxAbsErr | Kept
--- | --- | --- | --- | --- | --- | --- | ---
v1_stage1 | 1 | 1 | 1.3979 | 0.2272 | 6.15x | 7.97e+00 | ✗
v1_stage1 | 2 | 1 | 2.4966 | 0.2018 | 12.37x | 6.23e+01 | ✗
v1_stage1 | 4 | 1 | 4.8214 | 0.2037 | 23.67x | 3.74e+01 | ✗
v1_stage1 | 8 | 1 | 9.3289 | 0.1991 | 46.86x | 3.47e+01 | ✗
v1_stage1 | 16 | 1 | 17.9689 | 0.2456 | 73.16x | 3.68e+01 | ✗
v1_stage1 | 32 | 1 | 35.9094 | 0.4453 | 80.65x | 5.03e+01 | ✗
v1_stage1 | 64 | 1 | 70.9130 | 0.7666 | 92.51x | 6.33e+01 | ✗
v1_stage1 | 128 | 1 | 140.6617 | 1.3930 | 100.98x | 6.11e+01 | ✗

Root cause: `state_ptrs = state_base + (v_start + v_idx) * K + (k_offset + k_idx)` → wrong order.
Correct: `state_ptrs = state_base + (v_start + v_idx) * K + (k_offset + k_idx) * 1`.

Fix: swap BLOCK_V/BLOCK_K order so K is the inner loop (stride=1) and V is the outer tile (stride=K=128).
