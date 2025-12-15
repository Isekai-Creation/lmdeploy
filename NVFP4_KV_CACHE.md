# FP4 KV Cache Design (NVFP4 + MXFP4) for TurboMind / LMDeploy

This document captures the design, constraints, and implementation plan for adding **FP4‑based KV cache** support into the TurboMind backend used by LMDeploy, with:

- **NVFP4 KV cache** (FP4 E2M1 payload + FP8 E4M3 block scales, Blackwell/SM10x/SM12x), and
- **MXFP4‑style KV cache** as an optional fallback (FP4 payload + exponent‑style per‑block scaling on SM90/Hopper),

starting with **non‑MLA (standard MHA / GQA) models** (e.g. GPT‑OSS‑120B) and adding MLA support in a second phase.

The goal is to:

- Cut KV cache memory by ~2× vs FP8/int8 while
- Maintaining accuracy close to FP8 / BF16
- Preserving existing int4/int8 KV quant modes and paging logic
- Keeping the changes tightly scoped and consistent with NVIDIA’s NVFP4 design in TensorRT‑LLM.

---

## 1. Current LMDeploy / TurboMind KV Cache Architecture (Baseline)

### 1.1 Python side (LMDeploy)

**Key files**

- `LM/lmdeploy/lmdeploy/messages.py`
- `LM/lmdeploy/lmdeploy/turbomind/turbomind.py`
- `LM/lmdeploy/lmdeploy/turbomind/deploy/config.py`
- `LM/lmdeploy/src/turbomind/...` (C++ engine)

**Config surface**

- `TurbomindEngineConfig.quant_policy: int` with current valid values `{0, 4, 8, 16}` for TurboMind:
  - `0`: no KV quantization (KV cache stored in model dtype: FP16/BF16).
  - `4`: 4‑bit integer KV quantization.
  - `8`: 8‑bit integer KV quantization.
  - `16`: FP4 KV cache (FP4 E2M1 payload); maps to MXFP4 on Hopper (SM89/90) and is reserved for NVFP4 on Blackwell (SM10x/SM12x).
- CLI exposes `--quant-policy` with choices `[0, 4, 8, 16]` for the TurboMind engine; PyTorch engine may still restrict this to `{0, 4, 8}`.

**Propagation path**

1. `TurbomindEngineConfig.quant_policy` set from CLI or API.
2. `TurboMind.__init__` builds a YAML config with:
   - `model_config`, `attention_config`, `engine_config`.
3. C++ `LlamaTritonModel` parses `engine_config["quant_policy"]` into:
   - `ModelParam.quant_policy` (int).
4. `LlamaV2` / `UnifiedAttentionLayer` receive `ModelParam` and `EngineParam` and pass `model_param_.quant_policy` into attention and KV cache kernels.

### 1.2 C++ side: model + engine params

**Key structs**

- `ModelParam` in `llama_params.h`:
  - Includes `int quant_policy;`
- `EngineParam` in `llama_params.h`:
  - KV cache parameters: `cache_max_block_count`, `cache_chunk_size`, `session_len`, etc.
  - No NVFP4‑specific fields today.

**Quant policy enum**

- `llama_utils.h` (current):
  - ```cpp
    enum QuantPolicy {
        kNone        = 0x00,
        kReserve1    = 0x01,
        kReserve2    = 0x02,
        kCacheKVInt8 = 0x08,
        kCacheKVInt4 = 0x04,
        // FP4 KV cache (FP4 E2M1 payload + per-16-element scales; MXFP4/NVFP4 family)
        kCacheKVFp4  = 0x10,
    };
    ```
- C++ side interprets `quant_policy` as a **bitmask**:
  - Int modes: `quant_policy & kCacheKVInt8` / `kCacheKVInt4`.
  - FP4 modes: `quant_policy & kCacheKVFp4`, further refined to MXFP4 vs NVFP4 by `KvCacheMode` and SM version (see §3.3).

### 1.3 C++ side: KV block layout and cache managers

**Sequence / block managers**

- `SequenceManager` (`SequenceManager.h`):
  - Tracks sequences and their assigned KV blocks (`BlockIds`).
  - Holds a shared `BlockManager` that physically allocates KV blocks on device.

- `BlockManager` (`BlockManager.h/.cc`):
  - Allocates and manages a pool of `Block{ id, use_count, timestamp, void* data }`.
  - `block_size_` is the number of **bytes** per block, computed once at initialization.
  - Tracks active / cached / free block lists, and provides allocate/lock/unlock/evict APIs.

**Block layout**

- `block::Config<T, Tkv, HeadDim>` and `block::Layout` (`block.h`):
  - Defines how a block’s data is laid out:
    - `q_bits()` = `bitsof<Tkv>` (bits for quantized KV).
    - `t_bits()` = `bitsof<T>` when `T != Tkv` else 0 (bits for “param”/scale data).
    - `token_data_size = q_bits * head_dim / 8`.
    - `token_param_size = t_bits * 2 / 8` (scale + zero per token).
    - `head_data_size = block_len * token_data_size`.
    - `head_param_size = block_len * token_param_size`.
    - `layer_size = head_num * 2 * head_data_size + head_num * 2 * head_param_size`.
    - `block_size = layer_size * layer_num`.

- `get_cache_block_size(dtype, kvtype, layer_num, head_num, head_dim, block_seq_len)` (`kv_cache_utils_v2.cu`):
  - Computes `block_size` based on the layout above.
  - `dtype` is model dtype (fp16/bf16); `kvtype` is storage type (fp16, uint8, uint4).

**KV cache initialization**

- `LlamaBatch::InitializeBufferAndKVCache()` (`LlamaBatch.cc`):
  - Computes `dbits = byte_size(data_type_, 8)` (16 for fp16/bf16).
  - Derives `KvCacheMode` from `model_->param_.quant_policy` and SM version via `GetKvCacheMode`:
    - `kNone`, `kInt4`, `kInt8`, `kFp4Mx`, `kFp4Nv`.
  - Sets `q_bits` / `t_bits` based on the mode (see §3.3):
    - `kNone`: `q_bits = dbits`, `t_bits = 0`.
    - `kInt8`: `q_bits = 8`, `t_bits = dbits`.
    - `kInt4`: `q_bits = 4`, `t_bits = dbits`.
    - `kFp4Mx` / `kFp4Nv`: `q_bits = 4`, `t_bits = 0` (no inline `(scale, zero)`).
  - Constructs `SequenceManager::BlockConfig` with:
    - `head_dim`, `kv_head_num`, `block_seq_len`, and the chosen `q_bits`/`t_bits`.
  - Uses `get_cache_block_size` to compute the **data pool** `block_size`, and a separate `scale_block_size` for FP4 modes (one byte per 16 values for both K and V), which is passed into `SequenceManager` to allocate the FP4 scale pool alongside the data pool.

> **Historical note:** older code used `elem_bits = quant_policy ? quant_policy : dbits` and inferred `q_bits` directly from `quant_policy`. That was only sound when `quant_policy ∈ {0,4,8}` and meant “bit‑width”. In the current design this pattern has been removed in favour of `KvCacheMode` and explicit `q_bits`/`t_bits` selection.

### 1.4 C++ side: KV cache write / read paths

**KV write: prefill**

- `UnifiedAttentionLayer::core_attention` (`unified_attention_layer.cc`):
  - Computes Q/K/V for prefill and decode.
  - For prefill:
    - Calls `invokeProcessKV_v2_(params)`; `params.quant_policy = model_param_.quant_policy;`.

- `invokeProcessKV_v2` (`kv_cache_utils_v2.cu`):
  - Dispatches `ProcessKV_v2` with:
    - `T` = model dtype (`half` or `nv_bfloat16`),
    - `Tkv` chosen from `KvCacheMode`:
      - `uint8_t` if `KvCacheMode::kInt8`,
      - `uint4_t` if `KvCacheMode::kInt4`,
      - `fp4_e2m1_t` if `KvCacheMode::kFp4Mx` (MXFP4 on Hopper),
      - `T` otherwise (no quant, or `kFp4Nv` currently treated as base KV).
  - `ProcessKV_v2`:
    - Loads K/V tiles into `vec_K`, `vec_V`.
    - Optionally adds bias and applies RoPE.
    - For quantized KV (`T != Tkv`):
      - For int4/int8:
        - Uses `warp_stats` to compute per‑tile min/max.
        - Calculates `(scale, zero)` per tile.
        - Uses `ConvertKvCache<T, Tkv>` and `StoreQuantParam<Tkv>` to:
          - Write quantized data (`k_cache`, `v_cache`) into KV blocks.
          - Write per‑token quant params (`k_param`, `v_param`) into the param region of the block.
      - For MXFP4 (`Tkv = fp4_e2m1_t`):
        - Computes per‑16‑dim exponent bytes and writes them into the FP4 **scale pool** following the canonical layout
          `scale[layer][kv_head][token][K_scales(0..D/16-1), V_scales(0..D/16-1)]`.
        - Converts to FP4(E2M1) via `ConvertKvCache<fp4_e2m1_t, T>`’s inverse path and stores packed FP4 payload in the data pool via `block::Head<T, fp4_e2m1_t, BlockLayout>`.
        - Does **not** use the inline `(scale, zero)` param region.
      - For NVFP4 (`KvCacheMode::kFp4Nv`):
        - Remains **gated** and falls back to base KV until true FP8(E4M3) block scale support is implemented.

**KV read: flatten + decode**

- `LlamaV2::flattenPrefixKVForLayer` (`LlamaV2.cc`):
  - Determines `kv_dtype`:
    - `kUint8` if int8 KV,
    - `kUint4` if int4 KV,
    - else `dtype_` (fp16/bf16).
  - Uses `invokeFlattenKV_v2` (`kv_cache_utils_v2.cu`):
    - Reads quantized KV (`Tkv`) and quant params,
    - Dequantizes back to `T` using `ConvertKvCache<Tkv, T>`,
    - Produces flat `[B, H, S, D]` tensors `out_k`, `out_v`.

- Decode path:
  - `UnifiedAttentionLayer` builds `AttentionParams<T>` and sets `params.quant_policy = model_param_.quant_policy;`.
  - `dispatchDecoding<T>` (`decoding.cu`):
    - Sets `is_kv_int8` / `is_kv_int4` flags from `quant_policy`.
    - Dispatches `Decoding` kernels with `Tkv` = `uint8_t`, `uint4_t`, or `T`.

**Important:** Existing 4‑bit KV is an **integer quantization** (uint4_t with per‑token scale/zero), not NVFP4 E2M1. NVFP4 must be treated as a *different* 4‑bit mode, not just “another name” for int4.

---

## 2. External Reference Designs (NVFP4 / FP4 E2M1)

### 2.1 SGLang: FP4 KV cache wrapper for TRT‑LLM MLA

Relevant patterns:

- Adds CLI dtype `--kv-cache-dtype fp4_e2m1`, mapped to `torch.float4_e2m1fn_x2`.
- Provides a pure‑PyTorch `KVFP4QuantizeUtil`:
  - Block size 16; exponent stored as `uint8` per 16‑element block, using `ceil(log2(max|x| / 6.0)) + 127`.
  - FP4 E2M1 values stored as packed nibble codes in `uint8`.
- KV cache layout for MLA:
  - `kv_buffer`: `uint8` tensor storing packed FP4.
  - `kv_scale_buffer`: `uint8` tensor storing FP8‑like exponents per block.
- SGLang does **not** implement NVFP4 attention math; it simply:
  - Chooses the right container types and shapes,
  - Accounts for memory,
  - Passes K/V buffers and scale buffers into TensorRT‑LLM, which implements NVFP4 kernels.

### 2.2 TensorRT‑LLM: NVFP4 KV cache implementation

Key traits we will mirror:

- New KV cache type:
  - `enum class KvCacheDataType { BASE, INT8, FP8, NVFP4 };`.
- KVBlock pools:
  - Use “elements per container” to handle FP4 packing:
    - `getNumEltsPerContainer()` = 2 for FP4 (two FP4 values per uint8), else 1.
  - Pools are sized in units of containers; logical `sizePerHead` gets divided by `numEltsPerContainer`.
- Separate **block scale** pools:
  - Additional KVBlock pools for FP8 scales, with `sizePerHead` = `head_dim / quantBlockSize` (quantBlockSize=16).
  - One scale per 16 values, matching NVFP4 spec.
- QKV preprocessing:
  - `QKVPreprocessingParams` gains:
    - `kv_cache_block_scales_buffer` (KVBlockArray),
    - Combined `qkv_scale_*` fields (Q/K/V scales in a single array).
  - Dispatch chooses `T_cache = __nv_fp4_e2m1` for NVFP4 and uses NVFP4 encode/decode in quantization kernels.
- FMHA integration:
  - FMHA runner receives both KV data pointer and KV scale pointer.
  - Dequantization uses separate Q/K/V scales and per‑block NVFP4 scales.
- Constraints:
  - NVFP4 KV cache only on paged KV, not linear.
  - Head dim must be divisible by 2 and by block size 16.
  - NVFP4 KV cache requires FP8 context FMHA (Q path) and certain SMs (SM100/120/121).
  - MLA currently **does not** support NVFP4 KV (explicitly disabled in TRT‑LLM).

### 2.3 NVIDIA Model‑Optimizer NVFP4 (weights side, used as KV reference)

While TensorRT‑LLM implements the runtime kernels, **Model‑Optimizer** exposes the NVFP4 format and scaling semantics very explicitly in Python. We mirror those semantics for KV cache:

- Core implementation: `modelopt/torch/quantization/qtensor/nvfp4_tensor.py`
  - FP4 payload:
    - Values are quantized to E2M1 using an `e2m1_values` LUT.
    - FP4 codes are packed: `(q_weight[...,1::2] << 4) | q_weight[...,0::2]`.
  - Two‑level scaling:
    - Per‑tensor scale `weights_scaling_factor_2`:
      - `≈ reduce_amax(input) / (6.0 * 448.0)`.
    - Per‑block scale `weights_scaling_factor`:
      - `per_block_amax = reduce_block_amax(input, block_sizes={-1: block_size})`
      - `per_block_scale = per_block_amax / (6.0 * weights_scaling_factor_2)`
      - Stored as `torch.float8_e4m3fn` (viewed as `uint8`).
    - Effective scale at dequant time:
      - `per_block_scale * weights_scaling_factor_2` (both K and V use this pattern).
- ONNX export: `modelopt/onnx/export/nvfp4_exporter.py`
  - `_cast_fp4(array)`:
    - Uses `NVFP4QTensor._cast_fp4` and packs 2×FP4 per byte.
  - `_cast_fp8(array)`:
    - Clamps to ±448 and casts to `float8_e4m3fn` → `uint8`.
  - `compute_scales`:
    - Computes `_sw_f32_per_tensor` and `_sw_f32_per_block` for each weight tensor.
  - `compress_weights` + `_replace_fp4qdq_with_2dq`:
    - Replaces `TRT_FP4QDQ` with:
      1. `sw_f32 = DequantizeLinear(sw_f8_per_block, sw_f32_per_tensor)`
      2. `w32  = DequantizeLinear(w_f4, sw_f32, block_size=16)`
    - This exact “two‑DQ” structure is what we replicate in TurboMind’s NVFP4 KV decode: FP8(E4M3) block scales + FP32 global scales + FP4 payload.

For KV cache we treat each `(layer, kv_head)` like a “weight tensor”:

- FP4 payload layout:
  - Exactly identical to Model‑Optimizer: 2 FP4 codes per byte, E2M1 grid.
- FP8(E4M3) per‑16 block scales:
  - Stored in the **scale pool** as `uint8` bytes, but interpreted as FP8(E4M3).
- Optional global K/V scales:
  - FP32 arrays per `(layer, kv_head)` following the same `amax / (6 * 448)` logic.

TurboMind’s NVFP4 KV semantics are therefore:

- Encode:  
  `x_fp4 = QuantizeFP4_E2M1(x / (scale_fp8 * scale_global))`  
  `scale_fp8 = FP32 → FP8(E4M3)` per 16 dims, `scale_global` is FP32 per (layer, kv_head).
- Decode:  
  `x ≈ DequantFP4_E2M1(payload) * (DequantFP8_E4M3(scale_fp8) * scale_global)`.

This ensures KV cache uses **the same NVFP4 scaling semantics** as weights and ONNX export, just with KV‑specific layout and paging.

---

## 3. Design Goals for NVFP4 in TurboMind / LMDeploy

### 3.1 Scope (Phase 1)

- Target **non‑MLA** models first:
  - Standard MHA/GQA (e.g. GPT‑OSS‑120B).
  - No MLA‑specific KV layout changes in Phase 1.
- Add a **third FP4 KV cache family** on top of existing int4/int8:
  - Existing:
    - `0`: no KV quant.
    - `4`: int4 KV cache (uint4_t + scale/zero).
    - `8`: int8 KV cache (uint8_t + scale/zero).
  - New FP4 bit (`QuantPolicy::kCacheKVFp4` / `quant_policy=16` on TurboMind):
    - On Hopper (SM89/90): **MXFP4_KV** — FP4 E2M1 payload + per‑16‑dim exponent bytes stored in a separate **scale pool**.
    - On Blackwell (SM10x/SM12x): **NVFP4_KV** — FP4 E2M1 payload + per‑16‑dim FP8(E4M3) scales (to be implemented).
- Keep existing int4/int8 behaviour intact and configurable.

### 3.2 Constraints and invariants

We will enforce:

- Device:
  - `device_type == "cuda"` and SM version ≥ required NVFP4 capability (Blackwell).
- Layout:
  - KV cache must be *paged* (TurboMind’s block/paging cache), not any future linear KV path.
- Geometry:
  - `size_per_head` divisible by 2 (for FP4 packing).
  - `size_per_head` divisible by the NVFP4 block size (16 values per exponent).
- Compatibility:
  - When NVFP4 is enabled:
    - SpecPV partial KV path may be disabled initially (safe fallback) if we can’t flatten in NVFP4 easily.
    - Prefix caching and EAGLE behaviour must not regress (KV cache can still be paged/cached; only value representation changes).

### 3.3 KV cache modes and sizing (int4 / int8 / MXFP4 / NVFP4)

To avoid overloading `quant_policy` as both a bitmask and a raw bit‑width, we explicitly classify KV cache into **modes**, and we derive sizing/layout from the mode instead of `elem_bits = quant_policy`.

Define an internal enum (conceptual):

```cpp
enum class KvCacheMode {
    kNone,
    kInt4,
    kInt8,
    kMxFp4,   // FP4 + exponent-style per-block scaling (Hopper fallback)
    kNvFp4,   // FP4 + FP8(E4M3) block scales (Blackwell)
};
```

Mode selection from `ModelParam.quant_policy`:

- Start from the bitmask:
  - `has_int8 = (quant_policy & QuantPolicy::kCacheKVInt8) != 0;`
  - `has_int4 = (quant_policy & QuantPolicy::kCacheKVInt4) != 0;`
  - `has_fp4  = (quant_policy & QuantPolicy::kCacheKVFp4)  != 0;`
- Then:
  - If `has_fp4`:
    - On Blackwell (SM10x/SM12x): `kNvFp4`.
    - On Hopper (SM89/90): `kMxFp4` (MXFP4 exponent‑style format).
  - Else if `has_int8`: `kInt8`.
  - Else if `has_int4`: `kInt4`.
  - Else: `kNone`.

Sizing parameters per mode:

- Let `bits_T = bitsof(T)` for the model dtype (fp16/bf16 → 16).

- **q_bits** (payload bits per KV value) used for `token_data_size`:
  - `kNone`: `q_bits = bits_T` (no quant).
  - `kInt8`: `q_bits = 8`.
  - `kInt4`: `q_bits = 4`.
  - `kMxFp4`: `q_bits = 4` (FP4 payload).
  - `kNvFp4`: `q_bits = 4` (FP4 payload).

- **t_bits** (param bits per KV value) used for `token_param_size`:
  - `kNone`: `t_bits = 0`.
  - `kInt8` / `kInt4`:
    - `t_bits = bits_T`, with the existing layout where each token stores `(scale, zero)` in T for K and V.
  - `kMxFp4` / `kNvFp4`:
    - `t_bits = 0` — FP4 modes **do not** store per‑token `(scale, zero)` inside the data block. All per‑block scale metadata lives in a separate pool (see §5.2), matching TensorRT‑LLM’s NVFP4 design.

When initializing `SequenceManager::BlockConfig` for FP4 modes we will:

- Derive `KvCacheMode` from the bitmask.
- Set `q_bits`/`t_bits` based on the table above.
- Stop using `elem_bits = quant_policy ? quant_policy : dbits` for anything other than the legacy `{0,4,8}` modes.

---

## 4. API & Config Design

### 4.1 Quant policy semantics

We have two realistic options for exposing NVFP4:

#### Option A: Reuse `quant_policy = 4` for NVFP4 (breaking int4 KV)

- Pros:
  - Keeps Python/Turbomind `quant_policy` domain = `{0, 4, 8}`.
  - No CLI change required.
- Cons:
  - Breaks existing semantics where `4` means **integer 4‑bit** KV (uint4_t).
  - All existing tests/configs using 4‑bit KV would silently become NVFP4 KV.
  - Int4 KV (which may be desirable for some models) becomes unavailable or needs a new code path.

Given the risk and lack of strong reason to deprecate int4 KV, this option is **not acceptable** for a clean integration.

#### Option B (chosen): Extend `quant_policy` domain to `{0, 4, 8, 16}` and add a new FP4 bit

- Extend `QuantPolicy` enum (`llama_utils.h`) with:
  - `kCacheKVFp4 = 0x10;`
- Interpret `ModelParam.quant_policy` as a **bitmask**:
  - Lower bits (0x04, 0x08) indicate integer quant bits (int4/int8).
  - New bit 0x10 indicates **FP4 KV cache family**:
    - KV values stored as FP4 E2M1 in the data pool,
    - KV scales stored in a separate scale pool (exponent bytes for MXFP4; FP8(E4M3) for NVFP4).
- Mapping from Python `quant_policy` to C++ `QuantPolicy`:
  - `0`  → `kNone`.
  - `4`  → `kCacheKVInt4`.
  - `8`  → `kCacheKVInt8`.
  - `16` → `kCacheKVFp4`.
- Python side changes (TurboMind):
  - Relax assertions in `TurbomindEngineConfig.__post_init__` to allow `16`.
  - Update CLI helpers (`CLI.utils.ArgumentHelper.quant_policy`) to accept `16` and document it as “FP4 KV cache (MXFP4 on Hopper, NVFP4 reserved on Blackwell)”.
  - For TurboMind engine, treat `quant_policy=16` as “FP4 KV cache” and let `GetKvCacheMode` map to MXFP4 vs NVFP4 based on SM and `ENABLE_FP4`.

This keeps:

- Backwards compatibility for 4‑bit int KV (`quant_policy=4`).
- A clean, explicit code path for NVFP4 (`quant_policy=16`).

### 4.2 FP4‑specific engine options (MXFP4 first, NVFP4 gated)

We will **not** add a separate top‑level flag initially; instead:

- Use `quant_policy = 16` (bit `kCacheKVFp4`) to indicate the FP4 KV cache family.
- Let `GetKvCacheMode(quant_policy, sm_version)` choose:
  - MXFP4 (`kFp4Mx`) on Hopper SMs (89/90) when `ENABLE_FP4` is defined.
  - NVFP4 (`kFp4Nv`) on Blackwell SMs (100/101/120/121) when `ENABLE_FP4` is defined.
- Gate on:
  - CUDA device capability (SM ≥ 100 for **NVFP4**, SM89/90 for **MXFP4**),
  - Build flag `ENABLE_FP4` (mirroring TRT‑LLM),
  - Model dtype and KV shape constraints (e.g., head_dim % 16 == 0).

If needed later, we can add:

- `--kv-cache-format {auto,int,fp4}` as a more descriptive CLI, but this is not required for Phase 1.

---

## 5. C++ Engine Design (TurboMind)

### 5.1 Types and feature flags

#### 5.1.1 New QuantPolicy value

- In `llama_utils.h`:
  - Add `kCacheKVFp4 = 0x10;`.

#### 5.1.2 FP4 type alias

- FP4 E2M1 storage type already exists in TurboMind:
  - `struct fp4_e2m1_t {};` with `bitsof_t<fp4_e2m1_t> = 4` and `SubBytePtr<fp4_e2m1_t>` for packed storage.
- Guard FP4‑specific kernels and dispatch under `#if ENABLE_FP4` where CUDA headers expose FP4 conversion intrinsics.

#### 5.1.3 Capability detection

- Reuse existing SM detection helpers (e.g. `getSMVersion()`) used elsewhere in TurboMind, and gate FP4 modes as follows:
  - **Compile‑time**:
    - Require `ENABLE_FP4` (or equivalent) so FP4 types (`__nv_fp4_e2m1` and related helpers) are available in CUDA headers.
  - **Runtime SM checks**:
    - **NVFP4 path** (true NVFP4, FP8(E4M3) block scales, Blackwell):
      - Allow only on a known whitelist of SMs (e.g. `sm_100`, `sm_101`, `sm_120`, `sm_121`) where NVIDIA documents NVFP4 support.
      - If `quant_policy` requests NVFP4 on an unsupported SM:
        - Either hard‑fail (preferred for clarity), or
        - Explicitly fall back to `kMxFp4` or `kInt8` with a loud warning.
    - **MXFP4 fallback** (FP4 payload + exponent scaling, Hopper/SM90):
      - Allow on SM90/SM89 and above, even when NVFP4 hardware is not present.
      - This gives a “best effort” FP4 KV cache on H100, distinct from NVFP4.
  - If neither FP4 mode is supported at build + runtime:
    - Treat `quant_policy=16` as invalid and require users to choose `0`, `4`, or `8`.

### 5.2 KV block pools and layout

FP4 KV cache (both MXFP4 and NVFP4) uses:

- FP4 E2M1 values stored **packed** in 8‑bit containers (2 FP4 values per byte).
- Separate per‑block scale metadata:
  - MXFP4: 1 exponent byte (UE8M0‑like) per 16 values.
  - NVFP4: 1 FP8(E4M3) value per 16 values, plus optional FP32 K/V tensor‑level scales.

We will mirror the TRT‑LLM structure while fitting TurboMind’s `BlockManager`:

#### 5.2.1 Data pool vs scale pool

We introduce two logically parallel block managers:

1. **Data pool**: existing `BlockManager` that stores KV values.
2. **Scale pool**: new `BlockManager` that stores per‑block FP8 scales.

**Design choice: separate scale pool instead of overloading `t_bits`**

- Existing int4/int8 use `t_bits` and `k_param`/`v_param` to store per‑token `(scale, zero)` of type `T` inside the same block as KV data.
- NVFP4 uses:
  - FP4 values,
  - Coarser per‑16‑element FP8 scales,
  - Optional global K/V scales.
- Overloading `t_bits` and `k_param`/`v_param` to emulate NVFP4 would:
  - Waste space (per‑token instead of per‑block),
  - Make NVFP4 layout incompatible with int4/int8,
  - Complicate block size calculations.

Hence we keep:

- Int4/int8 layout unchanged.
- NVFP4 semantics: **param storage inside blocks is unused**, and all NVFP4 block scales live in a separate pool.

#### 5.2.2 BlockManager integration

In `LlamaBatch::InitializeBufferAndKVCache()`:

- Detect FP4 modes via `KvCacheMode`:
  - `KvCacheMode::kFp4Mx` on Hopper (MXFP4),
  - `KvCacheMode::kFp4Nv` on Blackwell (NVFP4, currently gated to base KV).
- For FP4 modes (`KvCacheMode::kFp4Mx` or `kFp4Nv`):
  - **Data pool layout (packed FP4 in uint8 containers)**:
    - Per token, per head, per K/V:
      - Logical FP4 values per head: `head_dim`.
      - Packed containers per head: `head_dim / 2` bytes (2 FP4 values per byte).
    - Per token across all KV heads:
      - `bytes_per_token_data = num_kv_heads * 2 /*K+V*/ * (head_dim / 2);`
    - Per block (paged KV block length `block_seq_len`) for all layers:
      - `block_size_data = bytes_per_token_data * block_seq_len * layer_num;`
    - We create a `BlockManager` for KV **data** with `block_size_ = block_size_data`.
    - Internally, kernels will view these bytes as FP4 containers and use CUDA FP4 intrinsics to load/store.
  - **Scale pool layout (per‑16‑value scales)**:
    - Per token, per head, per K/V:
      - Blocks along head dim: `blocks_per_head = head_dim / 16;`
      - For MXFP4:
        - 1 exponent byte per block (UE8M0‑style).
      - For NVFP4:
        - 1 FP8(E4M3) value per block, stored as `uint8`.
    - Per token across all KV heads:
      - `bytes_per_token_scales = num_kv_heads * 2 /*K+V*/ * blocks_per_head;`
    - Per block for all layers:
      - `block_size_scales = bytes_per_token_scales * block_seq_len * layer_num;`
    - We create a second `BlockManager` (e.g. `scale_block_manager_`) with `block_size_ = block_size_scales`.
  - **Data+scale pool invariants**:
    - Both pools must:
      - Have the same `cache_max_block_count` and number of blocks.
      - Allocate and evict blocks in lockstep, so **data block id and scale block id are identical**.
      - Be treated as a single logical KV allocation unit by `SequenceManager`.
    - This mirrors TRT‑LLM’s `KVBlockArray` + KV‑scale array sharing the same paging structure.

For non‑NVFP4:

- Behaviour remains unchanged (one `BlockManager` only; no scale pool).

#### 5.2.3 SequenceManager knowledge of scales

- Extend `SequenceManager` to optionally hold a pointer to the scale `BlockManager`:
  - New member: `std::shared_ptr<BlockManager> scale_block_manager_;` (nullable).
  - For NVFP4:
    - `scale_block_manager_` is constructed alongside `block_manager_`.
    - Any allocation/eviction affecting `block_manager_` should also apply to `scale_block_manager_`:
      - Allocation: allocate a block from both managers, use the same `BlockIds` ordering.
      - Eviction / Free: symmetrically free both data and scale blocks.
  - For non‑NVFP4:
    - `scale_block_manager_ == nullptr`, all existing code stays as is.

We can implement the scale pool wiring as a thin layer around `BlockManager`:

- Introduce small helpers in `SequenceManager`:
  - `AllocateDataAndScaleBlocks(int count, BlockIds& data_ids, BlockIds& scale_ids);`
  - `FreeDataAndScaleBlocks(const BlockIds& data_ids, const BlockIds& scale_ids);`

Instead of touching `BlockManager` internals, we keep the dual‑pool logic in `SequenceManager`.

### 5.3 NVFP4 quantization path in `kv_cache_utils_v2`

#### 5.3.1 Dispatch on NVFP4

In `invokeProcessKV_v2` and `invokeFlattenKV_v2` (`kv_cache_utils_v2.cu` / `.h`):

- Detect KV mode via `KvCacheMode` instead of raw `quant_policy` bits.
- Dispatch `Tkv`:
  - `KvCacheMode::kInt8`  → `Tkv = uint8_t` (int8 KV).
  - `KvCacheMode::kInt4`  → `Tkv = uint4_t` (int4 KV).
  - `KvCacheMode::kFp4Mx` → `Tkv = fp4_e2m1_t` (MXFP4 FP4 values).
  - `KvCacheMode::kFp4Nv` → currently treated as base KV (`Tkv = T`) until true NVFP4 is implemented.

#### 5.3.2 NVFP4 quantization (write path)
For FP4 KV cache we distinguish two algorithms, both using FP4(E2M1) payload but **different scale formats**:

1. **MXFP4_KV (fallback, SM90/Hopper)**:
   - FP4 payload packed in bytes.
   - One **exponent byte** per 16 values (UE8M0‑style power‑of‑two scaling).
2. **NVFP4_KV (Blackwell)**:
   - FP4 payload packed in bytes.
   - One **FP8(E4M3) scale value** per 16 values.
   - Optional FP32 tensor‑level K/V scales, as in TRT‑LLM’s QKV preprocessing path.

**MXFP4_KV write sketch (per block, per head, per token):**

1. Group `[head_dim]` into blocks of 16 values in `vec_K` / `vec_V`.
2. For each 16‑value block:
   - `max_val = max(|x_i|)`.
   - Compute exponent `scale_exp = ceil(log2(max_val / 6.0f))` (or similar heuristic).
   - Store `scale_exp` as a `uint8` in the **scale pool**.
3. Normalize & quantize:
   - `scaled = x / 2^scale_exp`.
   - Convert `scaled` to FP4(E2M1) with CUDA intrinsics (`__nv_fp4_e2m1` or equivalent).
   - Pack into bytes (2 FP4s per `uint8`) and store via `block::Head<T, fp4_e2m1_t, BlockLayout>` / `SubBytePtr`.

**NVFP4_KV write sketch (per block, per head, per token):**

1. Group `[head_dim]` into blocks of 16 values in `vec_K` / `vec_V`.
2. For each 16‑value block:
   - Compute per‑block scale as a *true FP value* in FP8(E4M3), e.g. based on min/max or max‑abs, following TRT‑LLM’s NVFP4 QKV preprocessing.
   - Optionally factor out a FP32 K/V tensor‑level scale so the FP8 block scales remain in a well‑behaved numeric range.
   - Store this FP8 scale byte in the **scale pool**.
3. Normalize & quantize:
   - `scaled = x / scale_fp8` (interpreting the FP8 E4M3 bytes as a float scale).
   - Convert `scaled` to FP4(E2M1) with CUDA intrinsics.
   - Pack into bytes and store in the data pool as above.

For both MXFP4_KV and NVFP4_KV:

- Param storage in the data block (`k_param`/`v_param`) remains **unused**; all scale metadata lives in the scale pool (and optional tensor‑level K/V scales).

#### 5.3.3 NVFP4 dequantization (read/flatten path)

In `invokeFlattenKV_v2`:

- For NVFP4:
  - Read packed FP4 values from KV data blocks via `block::Head<T, fp4_e2m1_t, BlockLayout>`.
  - Read per‑block scales from the scale block pool:
    - MXFP4_KV: exponent byte, interpreted as power‑of‑two scale.
    - NVFP4_KV: FP8(E4M3) value, interpreted as a normal floating scale.
  - Dequantize:
    - `x = decode_fp4_e2m1(fp4_value)` → float16/bfloat16.
    - For MXFP4_KV: `x_dequant = x * 2^scale_exp`.
    - For NVFP4_KV: `x_dequant = x * scale_fp8`.
  - Write dequantized `T` into flat `[B, H, S, D]` output.

This is analogous to TRT‑LLM’s `QKVPreprocessingParams` + FMHA setup, but implemented in TurboMind’s flatten path.

#### 5.3.4 Decode path

In `dispatchDecoding<T>` (`decoding.cu`):

- Currently only `INT8` and `INT4` paths exist.
- For FP4 KV modes (MXFP4_KV / NVFP4_KV):
  - The **primary decode path must read FP4 KV + scales directly** and dequantize on‑the‑fly:
    - Extend the decode kernels (or their parameterization) so that:
      - KV payload is loaded as packed FP4 containers,
      - The corresponding per‑block scale is loaded from the scale pool,
      - Dequantization to `T` happens in registers just before Q·Kᵀ and AV operations.
    - This keeps decode bandwidth aligned with compressed KV size and matches the intent of NVFP4 (bandwidth‑bound decode).
  - A **flatten‑to‑T path** (via `invokeFlattenKV_v2`) is still useful, but only as:
    - A debug/diagnostic path, or
    - A compatibility path for features that require dense `[B, H, S, D]` KV (e.g. SpecPV flattening), not the hot decode loop.

Decode work items for implementation:

- Add a new `KvCacheMode` branch in `dispatchDecoding` for FP4 modes.
- Plumb FP4 payload + scale pool pointers into the decode kernels similarly to how TRT‑LLM’s FMHA/XQA code does.
- Ensure the new decode path keeps per‑head/per‑block scaling semantics identical between prefill and decode.

#### 5.3.5 Concrete NVFP4 helpers and kernels (TurboMind entry points)

To make the above semantics implementation‑ready, we standardize the NVFP4‑specific helpers and kernel entry points:

- FP8(E4M3) decode helper (device):

```cpp
// Near other FP4/FP8 helpers (e.g. core/data_type.h or a small fp8_utils.h)
__device__ inline float decode_fp8_e4m3(uint8_t v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    // Example using NV FP8 intrinsics if available
    __nv_fp8_e4m3fn x;
    x.__x = v;
    // Convert to FP32; exact intrinsic depends on CUDA version
    return __half2float(__nv_cvt_fp8_to_fp16(x, __NV_SATFINITE, 0));
#else
    // Fallback: small E4M3 decode using bit manipulation
    return fp8_e4m3_decode_fallback(v);
#endif
}
```

- NVFP4 branch in the KV write dispatcher:

```cpp
// kv_cache_utils_v2.cu
template <typename T>
void invokeProcessKV_v2(/* existing params */,
                        KvCacheMode kv_cache_mode,
                        uint8_t* kv_blocks,
                        uint8_t* scale_blocks,
                        const float* global_k_scales,
                        const float* global_v_scales) {
    switch (kv_cache_mode) {
    case KvCacheMode::kFp4Mx:
        TM_CHECK(scale_blocks != nullptr);
        ProcessKV_v2<T, fp4_e2m1_t, KvCacheMode::kFp4Mx>(
            /* args */, kv_blocks, scale_blocks, nullptr, nullptr);
        break;
    case KvCacheMode::kFp4Nv:
        TM_CHECK(scale_blocks != nullptr);
        ProcessKV_v2<T, fp4_e2m1_t, KvCacheMode::kFp4Nv>(
            /* args */, kv_blocks, scale_blocks,
            global_k_scales, global_v_scales);
        break;
    // existing kNone/kInt4/kInt8 cases...
    }
}
```

- NVFP4 encode inside `ProcessKV_v2`:

```cpp
// kv_cache_utils_v2.cu
template <typename T, typename Tkv, KvCacheMode Mode>
__global__ void ProcessKV_v2(/* geometry, pointers, etc. */,
                             uint8_t* kv_blocks,
                             uint8_t* scale_blocks,
                             const float* global_k_scales,
                             const float* global_v_scales) {
    // ... common setup ...

    if constexpr (Mode == KvCacheMode::kFp4Nv) {
        // For each (layer, kv_head, token) handled by this kernel/block:
        float gk = global_k_scales ? global_k_scales[gk_index] : 1.0f;
        float gv = global_v_scales ? global_v_scales[gv_index] : 1.0f;

        for (int block_idx = 0; block_idx < blocks_per_head; ++block_idx) {
            float k_vals[16];
            float v_vals[16];
            load_kv_block_16(/* args */, k_vals, v_vals);

            float max_k = max_abs_16(k_vals);
            float max_v = max_abs_16(v_vals);

            float s_k = max_k > 0.f ? max_k / (6.0f * gk) : 0.f;
            float s_v = max_v > 0.f ? max_v / (6.0f * gv) : 0.f;

            uint8_t k_scale_fp8 = encode_fp8_e4m3(s_k);
            uint8_t v_scale_fp8 = encode_fp8_e4m3(s_v);
            store_nvfp4_block_scales(scale_blocks, layer, kv_head, token,
                                     block_idx, k_scale_fp8, v_scale_fp8);

            quantize_block_fp4_store(k_vals, s_k * gk, kv_blocks, /* K offset */);
            quantize_block_fp4_store(v_vals, s_v * gv, kv_blocks, /* V offset */);
        }
    }

    // ... other Mode branches ...
}
```

- NVFP4 decode in the attention decode path:

```cpp
// decoding.cu
template <typename T>
void dispatchDecoding(const AttentionParams<T>& params,
                      KvCacheMode kv_cache_mode,
                      const uint8_t* kv_blocks,
                      const uint8_t* scale_blocks,
                      const float* global_k_scales,
                      const float* global_v_scales) {
    switch (kv_cache_mode) {
    case KvCacheMode::kFp4Nv:
        DecodingKernel<T, fp4_e2m1_t, KvCacheMode::kFp4Nv>
            <<<grid, block>>>(params, kv_blocks, scale_blocks,
                              global_k_scales, global_v_scales);
        break;
    // existing cases...
    }
}

template <typename T, typename Tkv, KvCacheMode Mode>
struct StatePV {
    __device__ inline void Transform(/* ... */) {
        if constexpr (Mode == KvCacheMode::kFp4Nv) {
            const uint8_t* scale_base =
                get_fp4_nv_scale_base(scale_blocks_seq, layer_idx, kv_head_idx,
                                      token_idx, scales_per_head);

            for (int block_idx = 0; block_idx < blocks_per_head; ++block_idx) {
                uint8_t k_scale_byte = scale_base[block_idx];
                uint8_t v_scale_byte = scale_base[block_idx + scales_per_head];

                float gk = global_k_scales ? global_k_scales[gk_index] : 1.0f;
                float gv = global_v_scales ? global_v_scales[gv_index] : 1.0f;

                float s_k = decode_fp8_e4m3(k_scale_byte) * gk;
                float s_v = decode_fp8_e4m3(v_scale_byte) * gv;

                T k_vals[16], v_vals[16];
                load_fp4_block_and_convert(kv_blocks_seq, block_idx, k_vals, v_vals);

                for (int i = 0; i < 16; ++i) {
                    k_vals[i] = static_cast<T>(static_cast<float>(k_vals[i]) * s_k);
                    v_vals[i] = static_cast<T>(static_cast<float>(v_vals[i]) * s_v);
                }

                consume_pv_block(k_vals, v_vals, block_idx);
            }
        }

        // ... existing kNone/kInt4/kInt8/kFp4Mx handling ...
    }
};
```

These skeletons are **illustrative** only; the actual implementation must:

- Reuse existing FP4 packing/unpacking helpers for layout correctness.
- Use the same address computation helpers as MXFP4 (`get_fp4_mx_scale_base`) for NVFP4 (`get_fp4_nv_scale_base`).
- Maintain strict gating for NVFP4 (`ENABLE_FP4`, SM version checks) and keep `kFp4Nv` disabled until all TODOs in `NVFP4_KV_TODOS.md` §5.3–5.7 and §7 are satisfied.

---

## 6. Python / LMDeploy Integration (TurboMind)

### 6.1 Engine config & CLI

Changes in `LM/lmdeploy/lmdeploy/messages.py`:

- `TurbomindEngineConfig`:
  - `quant_policy: int` docstring updated to:
    - `0` → no KV quant,
    - `4` → KV int4,
    - `8` → KV int8,
    - `16` → KV NVFP4 (FP4 E2M1 + FP8 scales).
  - `__post_init__`:
    - Relax `assert self.quant_policy in (0, 4, 8)` to allow `16`.

Changes in CLI helper `LM/lmdeploy/lmdeploy/cli/utils.py`:

- `ArgumentHelper.quant_policy`:
  - Extend `choices` from `[0, 4, 8]` to `[0, 4, 8, 16]`.
  - Update help text to mention `16: NVFP4 (FP4 E2M1 KV cache)`.

### 6.2 TurboMind initialization

In `TurboMind.__init__` (`turbomind.py`):

- No structural changes needed:
  - `TurbomindEngineConfig.quant_policy` already passed into YAML and C++.
- Optional improvement:
  - Validate compatibility early:
    - If `quant_policy == 16` and `device_type != "cuda"`:
      - Raise an error: NVFP4 only supported on CUDA.
    - Optionally query device SM via `torch.cuda.get_device_capability` and warn if below required.

### 6.3 TurbomindModelConfig.update_from_engine_config

In `LM/lmdeploy/lmdeploy/turbomind/deploy/config.py`:

- `TurbomindModelConfig.update_from_engine_config` already copies common fields like `session_len` and others.
- Confirm that `quant_policy` is **not** overwritten in a conflicting way; if needed:
  - Ensure `quant_policy` is only sourced from engine config and not from HF overrides.

---

## 7. Phase 1 TODOs (Non‑MLA FP4 KV Cache: MXFP4 + NVFP4)

This section is the implementation checklist. We keep Phase 1 strictly about *non‑MLA* (standard MHA/GQA).

### 7.1 Config & plumbing (Phase 1: MXFP4 first, NVFP4 gated)

- [x] Extend `QuantPolicy` enum with FP4 KV bit (`kCacheKVFp4 = 0x10`) and derive `KvCacheMode` (NONE/INT4/INT8/FP4_MX/FP4_NV) based on `quant_policy` and SM version.
- [x] Allow `quant_policy == 16` in `TurbomindEngineConfig` (TurboMind backend) and wire it through the LMDeploy config / CLI as “FP4 KV cache (MXFP4 on Hopper; NVFP4 reserved on Blackwell)”.
- [ ] Extend PyTorch engine config to accept `quant_policy == 16` (optional; TurboMind‑only for now).
- [ ] In TurboMind Python, validate device type and CUDA capability for NVFP4 (optional but recommended).

### 7.2 KV block pools

- [x] In `LlamaBatch::InitializeBufferAndKVCache()`:
  - [x] Detect FP4 modes from `quant_policy` via `KvCacheMode` (kFp4Mx/kFp4Nv).
  - [x] For **MXFP4** (`KvCacheMode::kFp4Mx`), configure the data pool layout as packed FP4 containers with `q_bits = 4`, `t_bits = 0` (no inline per‑token `(scale, zero)`), and compute a separate `scale_block_size` based on `head_dim / 16`, `num_kv_heads`, `block_seq_len`, and `layer_num` (one byte per 16 values for both K and V).
  - [x] Pass `scale_block_size` into `SequenceManager` so that a dedicated scale `BlockManager` is created only when FP4 KV cache is active.
  - [x] Disable prefix caching (BlockTrie) when a scale pool is present, until BlockTrie is explicitly FP4‑aware.

- [x] Extend `SequenceManager`:
  - [x] Add an optional `scale_block_manager_` constructed alongside the data pool when `scale_block_size > 0`.
  - [x] Mirror `Lock`, `Unlock`, `Free`, `Evict`, and allocate flows across data and scale managers so that **data and scale blocks share the same block IDs and paging behaviour**.
  - [x] Provide `GetScaleBlockPtr(int block_id)` for host code to export per‑block scale pointers.

### 7.3 FP4 quantization kernels (MXFP4_KV + NVFP4_KV)

- [x] Introduce `fp4_e2m1_t` storage type and conversion helpers guarded with `ENABLE_FP4` (see `core/data_type.h` and `quantization.h`).
- [x] Extend `invokeProcessKV_v2`:
  - [x] Branch on FP4 modes via `KvCacheMode` and call `ProcessKV_v2` with `Tkv = fp4_e2m1_t` for MXFP4.
  - [x] Ensure `block::Layout` uses `q_bits = 4` and `t_bits = 0` for FP4 modes (no inline `(scale, zero)`).

- [x] Implement MXFP4_KV quantization inside `ProcessKV_v2`:
  - [x] Per‑block (16 values) exponent computation and storage in the scale pool (uint8 exponent bytes with bias 127).
  - [x] Conversion from `T` to `fp4_e2m1_t` and packed storage in the data pool.

- [ ] Implement NVFP4_KV quantization inside `ProcessKV_v2`:
  - [ ] Per‑block (16 values) FP8(E4M3) scale computation (and optional K/V tensor‑level scales) matching TRT‑LLM’s NVFP4 behaviour.
  - [ ] Conversion from `T` to `fp4_e2m1_t` and packed storage in the data pool.

- [ ] Extend `invokeFlattenKV_v2`:
  - [ ] For FP4 modes, decode FP4 values + block scales (MXFP4 or NVFP4) and output `T` (fp16/bf16) as a **debug/compat** path (not used in the main decode loop).

### 7.4 Decode & flatten integration

- [ ] Flatten path (`LlamaV2::flattenPrefixKVForLayer`):
  - [ ] Set `kv_dtype` / mode appropriately for FP4 KV.
  - [ ] Call `invokeFlattenKV_v2` in FP4 modes as a debug/compatibility path (not the main decode path).
  - [ ] Decide on SpecPV behaviour:
    - [ ] Initially: disable SpecPV when FP4 KV is enabled, unless flatten integration is explicitly implemented.

- [ ] Decode path (`dispatchDecoding`):
  - [ ] Add a new FP4 branch that reads FP4 KV payload + per‑block scales directly and dequantizes in registers before attention.
  - [ ] Ensure no hot decode path depends on flattening the entire KV cache to `T`.

### 7.5 Testing

- [ ] Unit tests at C++ level:
  - [ ] Synthetic test for NVFP4 quantize/dequant round‑trip on small tensors.
  - [ ] Ensure memory consumption (block counts) matches expectations (≈2× savings vs FP8).

- [ ] End‑to‑end tests via LMDeploy:
  - [ ] Run a smaller GPT‑OSS model with `quant_policy=16` and compare:
    - [ ] Accuracy against BF16/FP8 baseline on a curated evaluation set.
    - [ ] Latency and memory (NVFP4 vs FP8).

---

## 8. Phase 2 (Future): MLA FP4 KV Cache (NVFP4 + MXFP4)

> Not in scope for initial implementation; listed here to keep the design coherent.

Once non‑MLA MHA/GQA is stable, MLA can adopt NVFP4 KV using the same **data + scale pool** pattern. This will require:

- MLA‑specific block layouts (non‑rope vs rope components).
- MLA kernels that read NVFP4 KV + scales, similar to TensorRT‑LLM’s MLA kernels but adapted to TurboMind’s MLA implementation.
- Potential alignment with SGLang’s MLA KV layout for NVFP4 (to reuse codepaths in mixed engine setups).

We defer all MLA‑specific details until Phase 1 is complete and validated.

---

## 9. Summary

- Existing LMDeploy/TurboMind KV cache supports:
  - Unquantized, int4, and int8 KV cache in a single `BlockManager` pool.
- FP4 KV cache family (MXFP4_KV + NVFP4_KV) requires:
  - A new FP4 bit in `QuantPolicy` (`16` → `kCacheKVFp4`),
  - Dual block managers (data + scales) for KV cache,
  - FP4 quantization/dequantization kernels in `kv_cache_utils_v2`,
  - A decode path that reads FP4 KV + scales directly in the hot attention loop (no flatten‑before‑decode).
- Phase 1 implementation is staged:
  - First, **MXFP4_KV on Hopper (SM89/90)**: FP4(E2M1) payload + per‑16 exponent scales, end‑to‑end for non‑MLA models (prefill + decode), with SpecPV/flatten kept as debug/compat paths only.
  - Then, **NVFP4_KV on Blackwell (SM10x/SM12x)**: true NVFP4 semantics (FP8(E4M3) block scales + optional global K/V scales) under strict arch gating.
- NVFP4 is kept **gated** until the FP8(E4M3) scale path is fully implemented and tested (see `NVFP4_KV_TODOS.md` for live status and milestones).
