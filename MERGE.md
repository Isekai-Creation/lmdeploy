# EAGLE3 Branch & Merge Workflow

This file defines how we develop and maintain the `eagle3` branch for TurboMind EAGLE3 speculative decoding while keeping `main` on the latest engine / KV / DriftEngine work.

## 1. Branch Roles

- `main`
  - Tracks upstream `origin/main` plus all new features (engine scheduler, KV cache manager, FP4/NVFP4, DriftEngine, etc.).
  - May contain experimental / partially tuned EAGLE3 work.
- `eagle3`
  - Pinned to `8da9555dee0609647d368455dc4dfbd4f8e109ac` as the base.
  - Dedicated branch for **fully hardening EAGLE3 speculative decoding to 2–3×** (32K single, 16K batch4) with stable behaviour.
  - Only EAGLE3-related work should be merged into `main` once validated by perf gates.

## 2. Creating and Updating the `eagle3` Branch

- Initial branch (already created):
  - `git -C LM/lmdeploy branch eagle3 8da9555dee0609647d368455dc4dfbd4f8e109ac`
- To work on `eagle3`:
  - `cd LM/lmdeploy`
  - `git checkout eagle3`
  - Keep changes focused on:
    - EAGLE3 speculative decoding (LlamaV2, LlamaBatch, EagleModule, Eagle buffers / kernels, speculative_decoding kernels).
    - GEMM/FMHA tuning and perf gating for EAGLE3.
    - Minimal supporting changes in KV / attention needed to keep EAGLE3 compatible with future FP4/NVFP4 and DriftEngine.

## 3. Syncing From `main` Into `eagle3`

We treat `main` as the integration branch and periodically pull in its changes into `eagle3` so EAGLE3 stays compatible with new engine features.

Recommended workflow:

1. Update local `main`:
   - `git checkout main`
   - `git pull origin main`
2. Merge `main` into `eagle3`:
   - `git checkout eagle3`
   - `git merge main`
3. Resolve conflicts with the following priorities:
   - **EAGLE3-critical paths** (prefer `eagle3` implementation unless the `main` change is required for compatibility):
     - `src/turbomind/models/llama/LlamaV2.cc/.h`
     - `src/turbomind/models/llama/LlamaBatch.cc/.h`
     - `src/turbomind/models/llama/EagleModule.cc/.h`
     - `lmdeploy/turbomind/kernels/speculative_decoding/*`
     - `src/turbomind/models/llama/eagle3_attention_*`
   - **Shared infrastructure** (prefer `main` where possible, then re-integrate EAGLE3 behaviour on top):
     - `src/turbomind/kernels/attention/*` (including FP4/NVFP4 / kv_cache_utils_v2)
     - `src/turbomind/models/llama/SequenceManager.h` and KV / BlockManager wiring
     - DriftEngine / KVCacheManager / EngineScheduler
   - **Python API and benchmark harness**:
     - `lmdeploy/api.py`, `lmdeploy/messages.py`
     - `benchmark/profile_throughput.py`, `benchmark_speculative.py`, `run_spec_suite.sh`
     - Ensure EAGLE3 knobs and perf-gate scripts still work against the latest APIs on `main`.

After a merge:

- Rebuild TurboMind from `eagle3`.
- Run the **smallest** EAGLE3 smoke tests (e.g. tests/test_eagle_e2e.py, short-context benchmarks) before continuing deeper work.

## 4. Merging EAGLE3 Work Back Into `main`

We only merge from `eagle3` → `main` when:

1. The EAGLE3 code on `eagle3` passes:
   - Core correctness tests (baseline vs EAGLE3 equality where expected).
   - Multi-token invariants (LMDEPLOY_EAGLE_INVARIANTS_DEBUG) on small tests.
2. Perf gates on target hardware:
   - 32K single spec‑3 ≥ 2–3× baseline throughput.
   - 16K batch4 spec‑3 ≥ 1× baseline throughput.
   - Verified via `run_spec_suite.sh` (micro and full) in PERF_MODE.

Merge process:

1. Update `eagle3` from `main` (Section 3).
2. Run full EAGLE3 tests + perf gates on `eagle3` and fix regressions until clean.
3. Merge `eagle3` into `main`:
   - `git checkout main`
   - `git merge --no-ff eagle3`
4. Re-run a reduced perf suite on `main` to ensure nothing regressed.

## 5. High-Churn EAGLE3 Files to Watch

The following files are high-risk during merges and should be reviewed carefully:

- EAGLE3 logic:
  - `src/turbomind/models/llama/LlamaV2.cc/.h`
  - `src/turbomind/models/llama/LlamaBatch.cc/.h`
  - `src/turbomind/models/llama/EagleModule.cc/.h`
  - `lmdeploy/turbomind/kernels/speculative_decoding/common.{cu,h}`
  - `lmdeploy/turbomind/kernels/speculative_decoding/target_tree_decode.{cu,h}`
  - `src/turbomind/models/llama/eagle3_attention_*`
- KV / FP4 / NVFP4 / SequenceManager:
  - `src/turbomind/kernels/attention/kv_cache_utils_v2.*`
  - `src/turbomind/kernels/attention/fp4_kv_utils.h`, `fp4_kv_probe.cu`
  - `src/turbomind/models/llama/SequenceManager.h`
- Tests & harness:
  - `tests/test_eagle_e2e.py`
  - `tools/sweep_fmha_sm120.py`
  - `benchmark_speculative.py`, `run_spec_suite.sh`

When in doubt, prefer to:

- Keep `main`’s infra changes.
- Reapply EAGLE3-specific logic on top, guided by the EAGLE3 design doc and the perf/invariant TODO list.

## 6. EAGLE3 TODOs (Summary for This Branch)

On the `eagle3` branch, the goal is to bring EAGLE3 to **100% of the 2–3× target**. The active implementation plan (tracked in this conversation’s plan) covers:

1. Stabilize baseline (no spec) behaviour and ensure benchmarks run.
2. Rewire and harden the EAGLE3 core path (dynamicDecodeWithSpecMulti, EagleModule, buffers).
3. Implement strict multi-token invariants and LMDEPLOY_EAGLE_INVARIANTS_DEBUG checks.
4. Implement active-slot compaction, planner, and KV budgets.
5. Finalize draft CUDA graph fusion with geometry guards.
6. Strengthen multi-q EagleNet context prep aligned with TRT semantics.
7. Design and wire truncated draft vocab + d2t mapping.
8. Complete GEMM tuning loop (shape logging → offline tuning → TM_GEMM_IMPORT).
9. Complete FMHA tuning loop (env knobs → sweep → tuned defaults).
10. Tighten KvCacheMode/FP4/NVFP4 semantics and KV rewind assertions for EAGLE3.
11. Reduce PERF_MODE noise and unnecessary synchronizations; validate NVTX CPU–GPU balance.
12. Harden batch4 acceptance metrics, ALIGN_DEBUG checks, and per-reason fallbacks.
13. Implement micro perf gates and full `run_spec_suite.sh` PERF_MODE gates.
14. Finalize tests, docs, safe defaults, and fallback instrumentation.
15. Run full operational tuning on target hardware and confirm 2–3× EAGLE3 perf gates.

This file should be updated when:

- Branch roles change, or
- The EAGLE3 TODO set materially changes (e.g., new perf gates or invariants).

