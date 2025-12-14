# NVFP4 / MXFP4 KV Cache – TurboMind Progress & TODOs (UPDATED)

This file is the **single source of truth** for FP4 KV-cache work in TurboMind/LMDeploy.

We are implementing **two FP4 KV cache formats**:

- **MXFP4_KV (SM89/SM90 / Hopper)**: FP4(E2M1) payload (packed) + **per-16 exponent byte** scales (uint8, biased by 127).
- **NVFP4_KV (Blackwell SM100/101/120/121)**: FP4(E2M1) payload (packed) + **per-16 FP8(E4M3) scales** + optional **second-level global K/V scales**.

**Non-negotiables**
- No fake NVFP4 (do NOT implement NVFP4 using exponent bytes).
- No flatten-before-decode “Phase 1 workaround”.
- FP4 decode must read the **scale pool** (for MXFP4) and must not use affine KVp.
- Keep `kFp4Nv` gated until true NVFP4 semantics are implemented and tested.
- Keep SpecPV / partial-KV disabled for FP4 until FP4 flatten is implemented and verified.

---

## 0) Milestones / Critical Path

0.1 [x] **M0 – Plumbing & dual-pool allocation compiles**
- ✅ Dual pools, pointers, signatures, gating, ProcessKV FP4Mx plumbing compiles.

0.2 [ ] **M1 – MXFP4 end-to-end on SM90 (prefill + decode)**
- FP4Mx decode reads FP4 payload + exponent scales directly (no flatten).
- Verified runtime generation works, output is sane, memory footprint reduced.

0.3 [ ] **M2 – MXFP4 flatten support (debug/compat only)**
- Flatten FP4Mx works for debugging/tests. SpecPV stays disabled unless explicitly enabled later.

0.4 [ ] **M3 – NVFP4 end-to-end on Blackwell (prefill + decode)**
- True NVFP4 semantics: FP8(E4M3) block scales + optional global K/V scales.
- Strict arch gating and tests.

---

## 1) Global Config & Gating (TARGET: 100%)

**Goal:** expose FP4 KV cache as a distinct mode, with correct bitmask semantics and arch-aware gating.

1.1 [x] **Add FP4 bit in QuantPolicy**
- `QuantPolicy::kCacheKVFp4 = 0x10`

1.2 [x] **Add KvCacheMode enum**
- `KvCacheMode { kNone, kInt4, kInt8, kFp4Mx, kFp4Nv }`

1.3 [x] **Implement `GetKvCacheMode(quant_policy, sm_version)`**
- Must interpret `quant_policy` as a **bitmask**, not “bit width”.
- Rules:
  - FP4 bit set + Blackwell SM (100/101/120/121) + `ENABLE_FP4` → `kFp4Nv`
  - FP4 bit set + Hopper SM (89/90) + `ENABLE_FP4` → `kFp4Mx`
  - else int8/int4/none.

1.4 [x] **Compile-time gating**
- FP4 modes only compile when `ENABLE_FP4` is defined (CUDA headers expose FP4 types).

1.5 [x] **Runtime gating**
- If FP4 requested but arch unsupported → fail fast or fallback (explicit, documented behavior).

1.6 [x] **NVFP4 remains gated**
- `kFp4Nv` must not write/read scale pool until true NVFP4 is implemented.
- Current behavior: `kFp4Nv → dispatch(T{})` (base KV).

1.7 [ ] **Python/CLI config surface**
1.7 [x] **Python/CLI config surface**
- `--quant-policy 16` is exposed in LMDeploy CLI for TurboMind backend and validated in `TurbomindEngineConfig`.
- Help text describes:
  - `0: no quant`, `4: 4-bit int KV`, `8: 8-bit int KV`, `16: FP4 KV (MXFP4 on SM90; NVFP4 reserved on Blackwell)`.

---

## 2) KV Cache Layout & Allocation (TARGET: 100%)

**Goal:** FP4 KV cache uses **packed FP4 payload + separate scale pool**, with strict block-id invariants.

### 2.1 Data pool layout (FP4 payload)

2.1.1 [x] **FP4 data pool sizing**
- For FP4Mx:
  - `q_bits = 4`
  - `t_bits = 0` (no inline KVp param region)
  - bytes per token per head = `head_dim / 2` (2 FP4 values per byte)

2.1.2 [x] **Non-FP4 modes unchanged**
- int4/int8/unquantized layouts must remain byte-identical.

### 2.2 Scale pool layout (FP4 scales)

2.2.1 [x] **Scale pool sizing for FP4Mx**
- Constraints: `head_dim % 16 == 0`
- `scales_per_head = head_dim / 16`
- bytes per token per head = `2 * scales_per_head` (K + V)
- scale_block_size = `layer_num * kv_head_num * block_seq_len * bytes_per_token`

2.2.2 [x] **Scale pool layout definition (canonical)**
```

scale[layer][kv_head][token][
K_scales(0 .. scales_per_head-1),
V_scales(0 .. scales_per_head-1)
]

```

2.2.3 [x] **SequenceManager dual BlockManagers**
- `block_manager_` (data) + `scale_block_manager_` (scales)
- Same block-id domain, same allocation/eviction history

2.2.4 [x] **Dual-pool invariants enforced**
- Allocation returns identical ids: `scale_ids == data_ids`
- All lock/unlock/evict/free/touch operations mirrored

2.2.5 [x] **Block pointers plumbed end-to-end**
- LlamaBatch keeps and fills:
  - `block_ptrs_` + `scale_block_ptrs_`
- LlamaV2::Forward signature accepts both
- UnifiedAttentionLayer stores `kv_scale_block_ptrs_`
- BlockIteratorFactory offsets both pointers by `cu_block_nums`

2.2.6 [x] **Prefix cache / BlockTrie gating**
- Prefix caching disabled whenever scale pool exists (temporary safety choice)

2.2.7 [x] **SpecPV / partial-KV disabled for FP4**
- Must remain disabled until FP4 flatten is implemented and verified

---

## 3) MXFP4_KV (SM89/SM90) (TARGET: end-to-end)

**Goal:** make MXFP4 usable end-to-end on SM90: prefill writes FP4+scales, decode reads FP4+scales directly.

### 3.1 Prefill (ProcessKV_v2) – MXFP4

3.1.1 [x] **Implement FP4Mx quantization in ProcessKV_v2**
- Requirements:
  - `HeadDim % 16 == 0`
  - `kVecSize == 8`
- Must:
  - compute per-16 block max
  - compute exponent byte `exp = ceil(log2(max/6.0))`, store `exp+127`
  - quantize scaled values onto E2M1 grid
  - pack FP4 nibbles into payload
  - write exponent bytes to scale pool with canonical layout

3.1.2 [x] **Dispatch in invokeProcessKV_v2**
- `kFp4Mx`:
  - requires `scale_blocks != nullptr`
  - dispatch `Tkv = fp4_e2m1_t`

3.1.3 [x] **Do not modify affine int4/int8 paths**
- Keep original warp_stats + ConvertKvCache + StoreQuantParam unchanged

### 3.2 Shared scale offset helper & probe (DEBUG SAFETY NET)

3.2.1 [x] **Shared offset helper exists**
- `get_fp4_mx_scale_base(...)` is the single source of truth for scale addressing

3.2.2 [x] **Finalize FP4 probe kernel (Gate 2)**
- `fp4_kv_probe.cu` implements:
  - `Fp4KvProbeResult { k_scale0, v_scale0, k_scale1, v_scale1, kv_byte0 }`.
  - `Fp4KvProbeKernel` using `get_fp4_mx_scale_base(...)` so scale addressing exactly matches ProcessKV_v2.
  - Layout uses `Fp4ProbeConfig<T, fp4_e2m1_t>` so `block::Layout` is parameterized by runtime `head_dim` / `block_len` (no bogus HeadDim=0).
- Helper `fp4_kv_probe(...)` launches the 1-thread kernel for a single (layer, head, token).
- Helper `fp4_kv_probe_host(...)` allocates a device result buffer, launches the probe, copies back to host, and synchronizes.

3.2.3 [x] **Add host harness for probe**
- Add a minimal test/harness that:
  - runs a tiny prefill
  - invokes the probe for `(layer=0, head=0, token=0)` and `(token=block_len+1)`
  - prints results

3.2.4 [ ] **Add di/16 mapping assertion**
- Probe already reads `k_scale0/k_scale1` and `v_scale0/v_scale1`; host harness must now:
  - check that `scale_idx = di/16` selects `scale[0]` for `di in [0..15]` and `scale[1]` for `di in [16..31]`.
  - log/assert this mapping in a small debug/CI test.

### 3.3 Decode (Gate 3) – MXFP4 direct read + dequant (NO FLATTEN)

> Decode is currently **wired** to `Tkv=fp4_e2m1_t` but is **mathematically wrong** until we:
> - stop loading KVp,
> - read exponent scales from the scale pool,
> - multiply decoded FP4 by `2^exp`.

3.3.1 [x] **Decode Tkv wiring exists**
- `dispatchDecoding` can dispatch `Tkv=fp4_e2m1_t` for `kFp4Mx` (SM89/90 + ENABLE_FP4)

3.3.2 [x] **Introduce explicit FP4Mx branch in StateQK/StatePV**
- Add compile-time flag:
  - `kFp4Mx = std::is_same_v<Tkv, fp4_e2m1_t>`
- In Load:
  - `if constexpr (kQuantKV && !kFp4Mx)` load KVp as before
  - `if constexpr (kFp4Mx)` do NOT load KVp
- In Transform:
  - Base path unchanged
  - Affine int4/int8 unchanged
  - FP4Mx path:
    1) `ConvertKvCache<fp4_e2m1_t, T>` to get `x_scaled`
    2) multiply by `2^(scale_u8-127)` from scale pool per 16 dims
    3) no affine `scale/zero`

3.3.3 [x] **Thread scale_block_ptrs into Impl/Mainloop**
- Provide a scale accessor available to StateQK/StatePV:
  - gets `scale_blocks_seq` (offset by cu_block_nums)
  - provides `scale_base(local_ti)` using `get_fp4_mx_scale_base`

3.3.4 [ ] **Implement FP4Mx scale loads efficiently**
- Load one exponent byte per 16-dim block
- Reuse across lanes; do not load per element
- Cache in registers/shared where sensible

3.3.5 [ ] **Enable decode for FP4Mx**
- Remove the FP4Mx FT_CHECK only after 3.3.2–3.3.4 are correct
- Keep `kFp4Nv` gated

3.3.6 [ ] **Implement SIMT fallback path for FP4Mx**
- Ensure both 81616 and SIMT decode paths can run with FP4Mx

3.3.7 [ ] **Runtime validation on SM90**
- `quant_policy=16`:
  - prefill + decode generates tokens
  - no NaNs, no crash
  - output is sane vs baseline
- Print KV pool footprint difference

---

## 4) FP4 Flatten / SpecPV (TARGET: debug/compat only)

4.1 [x] **Flatten remains gated for FP4**
- `invokeFlattenKV_v2` FT_CHECKs for any FP4 mode

4.2 [ ] **Implement MXFP4 flatten (after decode is correct)**
- Read FP4 payload + exponent bytes from scale pool
- Dequantize to `T` (fp16/bf16) into flat `[B,H,S,D]`
- Maintain RoPE semantics consistent with existing flatten expectations

4.3 [ ] **Keep SpecPV disabled by default**
- Do not enable SpecPV for FP4 until:
  - flatten is correct
  - performance implications are understood
- If enabling later, add a dedicated FP4-compatible SpecPV strategy

---

## 5) NVFP4_KV (Blackwell) (TARGET: true NVFP4 semantics)

**Goal:** implement true NVFP4: FP4 payload + FP8(E4M3) per-16 block scales (+ optional global K/V scales).

5.1 [x] **NVFP4 mode selection exists**
- `GetKvCacheMode` returns `kFp4Nv` on Blackwell SMs

5.2 [x] **NVFP4 currently gated to base KV**
- `invokeProcessKV_v2(kFp4Nv)` falls back to `dispatch(T{})`

5.3 [ ] **Define NVFP4 scale representation**
- Scale pool contains FP8(E4M3) values per 16 dims, not exponent bytes
- Define storage as bytes but interpreted as FP8(E4M3)

5.4 [ ] **Add optional second-level global K/V scales**
- Define where these come from:
  - model weights / config / preprocessing
- Thread these scales into kernels and apply consistently

5.5 [ ] **Implement NVFP4 prefill write**
- FP4 payload written as in FP4Mx (packed)
- FP8(E4M3) block scales written to scale pool
- Apply global K/V scaling strategy (if enabled)

5.6 [ ] **Implement NVFP4 decode**
- Read FP4 payload + FP8 scales (+ global scales)
- Dequantize to FP8 or BF16/FP16 as required by kernel path
- Strict gating by arch and build flags

5.7 [ ] **NVFP4 tests on Blackwell**
- End-to-end correctness and sanity
- Memory footprint and performance measurements
- Keep NVFP4 gated until tests are green

---

## 6) Build Reliability / Tooling

6.1 [x] **EAGLE3 FMHA build guard**
- Heavy FMHA kernels behind compile-time `TM_ENABLE_EAGLE3_FMHA` (default OFF)
- Stub launcher logs and returns, allowing `_turbomind` to build

6.2 [ ] **Ensure no duplicate macro name confusion**
- Env var `TM_ENABLE_EAGLE3_FMHA` vs compile-time macro should be clearly documented
- Consider renaming compile-time macro to `TM_COMPILE_EAGLE3_FMHA` to avoid confusion

6.3 [ ] **Add a focused build command to docs**
- Document the exact command that must pass in CI:
  - `cmake --build build --target _turbomind -j8`

---

## 7) Testing, Validation & Benchmarks

7.1 [ ] **Unit test: FP4Mx encode/decode consistency**
- Compare:
  - `cvt_rn_sat_e2m1_f32` encoding
  - LUT decode path used in decode
- Ensure round-trip error is within expectations

7.2 [ ] **Integration test: MXFP4 end-to-end (SM90)**
- `quant_policy=16`:
  - prefill writes FP4+scales
  - decode reads FP4+scales and produces sane output
- Compare against `quant_policy=0` baseline

7.3 [ ] **Memory footprint test**
- Print KV pool allocation bytes for:
  - `quant_policy=0`
  - `quant_policy=8`
  - `quant_policy=16`
- Confirm ~2× reduction vs fp16/bf16 KV (accounting for scale pool overhead)

7.4 [ ] **Perf smoke**
- Decode throughput sanity (not full benchmark yet)
- Ensure FP4Mx is not accidentally forcing slow paths

7.5 [ ] **Probe tool integration**
- Make `fp4_kv_probe` invocable from a debug flag or unit test target

---

## 8) Docs Sync / Project Hygiene

8.1 [ ] **Update `NVFP4_KV_CACHE.md` to match reality**
- Phase 1 = MXFP4 end-to-end (prefill + decode) on SM90
- Flatten is debug-only follow-up
- NVFP4 remains gated and tracked separately

8.2 [ ] **Keep this TODO file authoritative**
- Every merged PR updates:
  - checkboxes
  - status text
  - any constraints

8.3 [ ] **Repository path hygiene**
- Verify any duplicate/legacy uppercase `LM/LMDEPLOY` folder references are removed (if present)
- Ensure all paths referenced in docs match the repo layout

---

## 9) Quick Status Snapshot (update this after each PR)

9.1 [x] Global config & gating: **DONE**
9.2 [x] Dual pools + allocation + pointer plumbing: **DONE**
9.3 [x] MXFP4 prefill (ProcessKV FP4Mx): **DONE**
9.4 [x] FP4 probe harness: **DONE (basic harness; see 3.2.4 for remaining assertions)**
9.5 [ ] MXFP4 decode (correct scale-pool dequant): **TODO (critical path)**
9.6 [ ] MXFP4 flatten: **TODO (after decode)**
9.7 [ ] NVFP4 true FP8-scale path: **TODO**
9.8 [ ] Tests/benchmarks: **TODO**
