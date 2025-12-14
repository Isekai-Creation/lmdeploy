# LMDeploy / TurboMind Engine Overview

This document captures the current LMDeploy engine architecture (with focus on the TurboMind backend) and contrasts it with other engines in this repo:

- `LM/vllm`
- `LM/sglang`
- `LM/TensorRT-LLM`
- `LM/EasyDeL` (eSurge)

It is written to support work on GPT‑OSS‑120B and EAGLE3, with the explicit goal of matching or exceeding the throughput of vLLM, sglang, TensorRT‑LLM, and EasyDeL under realistic serving loads.

---

## 1. LMDeploy / TurboMind Engine

### 1.1 Python Surface + Backends (LMDeploy)

- High‑level APIs (`lmdeploy.pipeline`, `lmdeploy.serve.api_server`):
  - Pipeline:
    - Builds an engine with a configurable backend (TurboMind, PyTorch).
    - Controls `tp`, `pp`, `max_batch_size`, `session_len`, precision, etc.
  - API server:
    - Wraps the engine and exposes HTTP/WS APIs with streaming.
    - Uses TurboMind via Python bindings for the C++ core engine.

### 1.2 Core Components (C++ / TurboMind)

- **Request / GenerationConfig** (`src/turbomind/engine/request.h`)
  - Encapsulates one logical inference request.
  - `GenerationConfig` holds decoding parameters:
    - `max_new_tokens`, `min_new_tokens`, `eos_ids`, `stop_ids`, `bad_ids`.
    - Sampling: `top_k`, `top_p`, `min_p`, `temperature`, `repetition_penalty`.
    - Outputs: `output_logprobs`, `output_last_hidden_state`, `output_logits`.
    - EAGLE/EAGLE3-specific: `posterior_thresholds`, `posterior_alphas` for draft‑token acceptance gating.
  - `Request` binds:
    - Session info (`SessionParam` – `id`, `step`, `start_flag`, `end_flag`, `kill_flag`).
    - I/O tensors (`inputs`, `outputs`, plus fast-path `output_ids`, `sequence_length`).
    - Callbacks: `forward_cb` (called when first state is available), `end_cb`.
    - `cancel_flag` for cooperative cancellation.
    - `state` (`AtomicRequestState`) for asynchronous status updates.
    - Optional `xgrammar::GrammarMatcher` for constrained decoding.

- **ModelRequest** (`src/turbomind/engine/model_request.{h,cc}`)
  - Host‑side convenience wrapper to submit/cancel/terminate requests.
  - Translates user tensors into internal `Request` objects:
    - Allocates `TensorMap` for inputs/outputs.
    - Computes conservative maximum sizes:
      - `max_seq_len` ≈ `session_len + 1`.
      - `max_out_len` and `max_in_out_len` derived from `max_new_tokens`, `input_len`, `session_len`.
    - Optionally allocates `logits`, `last_hidden_state`, `logprobs` buffers (CPU).
  - On `Forward`:
    - Populates `Request` with session parameters, generation configuration, metrics, grammar matcher, and output tensor references.
    - Enqueues the `Request` to `Gateway::push`.
    - Returns an `OutputParam` wrapper containing:
      - Shared output tensor map.
      - Shared `AtomicRequestState` handle for monitoring.
      - Optional per‑request metrics.

- **RequestQueue** (`src/turbomind/engine/request_queue.{h,cc}`)
  - Per‑rank queue of pending inference and kill requests.
  - Internally:
    - Uses `std::pmr::list<std::shared_ptr<Request>>` backed by `std::pmr::unsynchronized_pool_resource`.
    - Maintains a separate `kill_` vector for kill requests.
    - Uses `flag_` and an `expected_` counter for group synchronization.
  - API:
    - `push(Request)`: enqueue normal request and notify.
    - `kill(Request)`: enqueue kill request and notify.
    - `try_pop(...)`:
      - Non‑blocking, used to steal non‑start (continuation) requests from other queues to improve utilization.
    - `pop(...)`:
      - Blocking or non‑blocking; selects up to `max_infer` requests to run.
      - Updates sync flag; returns whether this rank is first in the DP group to proceed.
    - `assign_unique_ids(...)`: monotonic `unique_id` allocation for each dequeued request.

- **Gateway** (`src/turbomind/engine/gateway.{h,cc}`)
  - Routing and coordination layer between frontend and per‑rank request queues.
  - Responsibilities:
    - **Session‑to‑rank mapping** via `SeqId2Rank`:
      - First request for a session (`start_flag = true`) is assigned in round‑robin across ranks (atomic `next_`).
      - Subsequent requests with `start_flag = false` are routed to the bound rank (stateful sessions).
    - **Request distribution**:
      - `push(Request)`:
        - If continuation: look up rank by `session.id`.
        - If new session: assign via round‑robin.
        - On failure to route: mark request `kInvalid`.
      - `pop(...)`:
        - For a given rank, first attempts to steal continuation requests from other ranks via `try_pop`.
        - Then performs blocking `pop` on local `RequestQueue`, with optional wakeup of sibling ranks in the same group.
        - Assigns `unique_id`s and updates `SeqId2Rank` bindings:
          - Binds new active sessions (`start_flag && !end_flag`).
          - Unbinds killed sessions (`kill_reqs`).
    - **Cancellation**:
      - `cancel(Request)`:
        - Atomically sets `cancel_flag` from `0` to `-1` if the request is still queued.
        - If cancelled before pickup: emits `kCancel` via `notify`.
      - `kill(Request)`:
        - Locates bound rank; enqueues kill request; on failure emits `kInvalid`.
    - **Signal handling**:
      - Uses `SignalBuffer` and a dedicated signal thread to invoke deferred user callbacks (`UpdateState` / `end_cb`) off the hot path.

- **Engine loop / DynamicDecode** (`src/turbomind/layers/DynamicDecodeLayer.cc`)
  - Coordinates decode steps over the active batch:
    - Applies sampling (`SamplingLayer`) and uses generation configuration, including EAGLE posterior gating.
    - Manages speculative‑decoding‑specific flows (EAGLE/EAGLE3 / DeepSeek MTP) when enabled, including:
      - Tail commit (accept/reject of draft tokens).
      - Optional EAGLE performance mode (`LMDEPLOY_EAGLE_PERF_MODE`).
    - Integrates NVTX ranges for profiling and EAGLE debug instrumentation.
  - Works with model‑specific layers (e.g. `LlamaDecoderLayer` / `eagle3_attention_layer`) and KV‑cache abstractions below.

- **KV Cache & Layout**
  - Core utilities:
    - `src/turbomind/kernels/attention/kv_cache_utils_v2.{h,cu}`:
      - KV cache allocation, indexing, and copy helpers.
      - Supports compressed / paged layouts and integration with FlashMLA / FP4 KV.
    - `src/turbomind/models/llama/specpv_kv_cache.{h,cc}`:
      - Partial KV cache (SpecPV) for EAGLE3 target‑tree decode and partial verification.
    - `src/turbomind/kernels/attention/fp4_kv_utils.h`:
      - Utilities for NVFP4 KV quantization (kernel‑side).
  - KV cache characteristics:
    - Page/block‑structured KV with support for:
      - Long context (hierarchical layout).
      - Partial KV reuse for speculative decoding (SpecPV).
      - NVFP4 compression for KV to reduce memory bandwidth and footprint.

- **Speculative decoding kernels (EAGLE / EAGLE3 / DeepSeek MTP)**
  - `src/turbomind/kernels/speculative_decoding` (wired via `eagle_kernels` in `src/turbomind/kernels/CMakeLists.txt`):
    - `eagle_kernels.cu`, `optimized_kernels.cu`, `leaf_mask_kernels.cu`, `packed_mask_kernels.cu`.
    - `tree_accept_kernels.cu`, `target_tree_decode.cu`, `specpv_kv_kernels.cu`, `kv_rewind_helper.cu`.
  - These implement:
    - Draft tree acceptance logic (verify longest valid prefix).
    - Packed / leaf masks for efficient tree traversal on GPU.
    - Partial KV rewind/commit (SpecPV) when draft tokens are rejected.

### 1.2 Execution Model

- **Stateful sessions**
  - Each request belongs to a `SessionParam` with `id` and `step`.
  - `start_flag` / `end_flag` / `kill_flag` define lifecycle:
    - Start: new session; Gateway binds `id → rank`.
    - Continuation: routed to previously bound rank.
    - Kill: explicit termination; Gateway unbinds and cancels.

- **Batching and scheduling**
  - Batching is implemented cooperatively between:
    - `Gateway::pop`: constructs per‑rank batches of `Request`s.
    - Backend decode loop: consumes these requests and runs model forward + sampling, possibly iterating multiple decode steps.
  - Load distribution:
    - Stealing of continuation requests from other ranks (`try_pop`) to improve utilization without breaking session affinity.
    - DP group synchronization via `flag_` in `RequestQueue` to coordinate data‑parallel replicas.

- **Cancel / error propagation**
  - `Request::cancel_flag`:
    - `0` → queued; `1` → picked up by engine; `-1` → cancelled pre‑pickup.
  - `UpdateState` encapsulates status transitions:
    - Allocates a new `RequestState` with status and observed sequence length.
    - Swaps into `AtomicRequestState`; triggers `forward_cb` on first update.
  - Status codes:
    - `kOk`, `kInvalid`, `kConflict`, `kBusy`, `kInactive`, `kFail`, `kTooLong`, `kFinish`, `kCancel`, `kInconsistency`.

### 1.3 Performance‑Critical Paths

- **Attention kernels**
  - Optimized CUDA attention kernels, including FlashMLA integration for DeepSeek style MLA attention and NVFP4 KV.
  - Optional EAGLE3‑specific FMHA path:
    - Controlled via env vars (`TM_ENABLE_EAGLE3_FMHA`, `TM_EAGLE3_FMHA_*`).
    - Supports tile statistics and A/B correctness checks.

- **GEMM / MLP kernels**
  - `src/turbomind/kernels/gemm`:
    - `scheduler.cuh`, `scheduler_sm70.cuh`, and `gemm.cu` implement architecture‑tuned GEMM scheduling.
    - Can enable EAGLE performance mode via environment (`LMDEPLOY_EAGLE_PERF_MODE`).

- **Sampling layer**
  - `src/turbomind/layers/sampling_layers/SamplingLayer.cc`:
    - GPU sampling with environment‑gated EAGLE debug hooks.
    - Implements top‑k/top‑p/min‑p, repetition penalty, and EAGLE‑aware posterior logic.

- **Metrics and instrumentation**
  - `RequestMetrics` (`src/turbomind/utils/metrics.h`):
    - Tracks enqueue time, schedule time, and per‑request timing.
  - NVTX markers in dynamic decode and some kernels for profiling.

---

## 2. vLLM Engine (LM/vllm)

This section summarizes the vLLM engine as implemented in `LM/vllm`, focusing on features relevant to performance and architecture.

### 2.1 Core Engine

- **LLMEngine / EngineCore** (`vllm/v1/engine/llm_engine.py`)
  - High‑level orchestrator:
    - Accepts requests (`add_request`) with prompts and `SamplingParams`.
    - Delegates to `InputProcessor` for tokenization and request shaping.
    - Sends `EngineCoreRequest`s to `EngineCoreClient` (the actual worker / executor).
    - Consumes `EngineCoreOutputs` via `step()` and converts them to `RequestOutput`s using `OutputProcessor`.
  - Multiprocess support:
    - Can run the core engine in a separate process.
    - Handles DP group initialization and coordinated shutdown.

- **SchedulerConfig** (`vllm/config/scheduler.py`)
  - Encodes scheduler behavior:
    - `max_num_batched_tokens`: global per‑step token budget.
    - `max_num_seqs`: maximum sequences per step.
    - `enable_chunked_prefill`: enables chunked prefill of long prompts.
    - `max_num_partial_prefills`, `max_long_partial_prefills`, `long_prefill_token_threshold`:
      - Allow multiple concurrent partial prefill requests with preference for shorter prompts.
    - `policy`: `fcfs` or `priority`.
    - `scheduler_cls`: dynamic scheduler implementation (default or async scheduler).
    - Hybrid KV cache manager controls (enables mixing of different KV layouts / backends).

### 2.2 KV Cache and Attention

- **PagedAttention**
  - vLLM’s original paper and `docs/design/paged_attention.md` describe:
    - KV cache split into fixed‑size **blocks** (pages).
    - KV blocks arranged as `[num_blocks, num_kv_heads, ...]` and reused across prefixes.
    - Single‑query attention kernel (for decode) tuned to:
      - Maximize coalesced memory loads (`q_vecs`, `k_vecs`, `v_vecs`).
      - Let each warp process entire blocks of KV for a query.
  - Today’s implementation:
    - Abstracted into attention backends (e.g. `vllm/v1/attention/backends/flashinfer.py`).
    - Uses FlashInfer’s `BatchPrefillWithPagedKVCacheWrapper` and `BatchDecodeWithPagedKVCacheWrapper` for paged KV and decode.

- **KV Cache Interface** (`vllm/v1/kv_cache_interface.py`)
  - Defines `KVCacheSpec`, `AttentionSpec`, `KVCacheConfig`, etc.
  - Supports:
    - Different layouts (e.g. `NHD`, `HND`).
    - Grouped cache specs and uniform‑type checks for compilation.
    - Hybrid KV cache manager that can mix paged and other layouts where needed.

### 2.3 Scheduling and Execution

- **Continuous batching**
  - Requests are continuously admitted and scheduled every `LLMEngine.step()`:
    - Prefill vs decode handled by the scheduler.
    - Chunked prefill allows mixing long and short prompts without starving shorter ones.
  - Async scheduler option for lower latency and higher utilization.

- **GPU‑centric sampling**
  - Sampling and logits post‑processing are implemented as GPU operations, keeping CPU involvement minimal in the hot path.

### 2.4 Key Takeaways vs TurboMind

- Strong points:
  - Mature **PagedAttention** with prefix sharing and robust page allocator.
  - Aggressive **chunked prefill** and **continuous batching** scheduler.
  - Clean KV cache abstraction that supports multiple backends and hybrid layouts.
  - GPU‑centric pipeline (tokenization aside) with low per‑token overhead.

- Gaps / differences relative to current TurboMind:
  - vLLM’s prefill/decode scheduler is more explicit and configurable (chunked prefill, partial prefill concurrency).
  - PagedAttention’s block allocator is a first‑class component; TurboMind has powerful KV utilities but a less explicit paged allocator abstraction at the engine level.

---

## 3. sglang Engine (LM/sglang)

### 3.1 Scheduler

- **Scheduler** (`python/sglang/srt/managers/scheduler.py`)
  - Central runtime component for a tensor‑parallel worker:
    - Handles request admission, batching, multi‑tenant scheduling, function calling, and disaggregation.
  - Uses multiple mixins:
    - `SchedulerOutputProcessorMixin`, `SchedulerMetricsMixin`, `SchedulerProfilerMixin`.
    - Disaggregation mixins for prefill/decode offload.
    - `SchedulerPPMixin`, `SchedulerDPAttnMixin` for parallelism modes.
    - `SchedulerMultiplexMixin` for multiplexed execution.
  - Integrates:
    - Model configuration, MoE config, FP8 GEMM config, multi‑modal registries.
    - ZMQ‑based communication and gRPC scheduler control plane.

### 3.2 Radix Prefix Cache

- **RadixCache** (`python/sglang/srt/mem_cache/radix_cache.py`)
  - Tree‑structured prefix cache over token sequences with pluggable eviction:
    - Keys: `RadixKey(token_ids, extra_key)` with optional namespace (`extra_key`) to separate tenants / adapters.
    - Values: device indices into KV cache blocks.
  - Features:
    - Supports page size (`page_size`) > 1, enabling paged KV cache semantics:
      - Uses `_key_match_paged` for page‑aligned matching.
    - Eviction policies: LRU, LFU, FIFO, MRU, FILO, priority.
    - EAGLE mode:
      - `is_eagle` enables bigram key conversion to align with EAGLE drafting.
    - Simulated mode (`create_simulated`) for testing / scheduler integration without real KV pools.
  - API:
    - `match_prefix(RadixKey) -> MatchResult`:
      - Returns longest cached prefix indices and terminal node.
    - `insert(RadixKey, value, chunked, priority)`:
      - Inserts new KV blocks for a prefix with optional chunking and priority.

### 3.3 Scheduling & KV Reuse

- Scheduler integrates RadixCache to:
  - Share KV across requests that share prefixes or program fragments (“LLM programs” and branchy prompts).
  - Implement tree caching for SSMs / MLA (HiRadixCache, SWARadixCache, MambaRadixCache).
  - Combine continuous batching with heavy prefix reuse.

### 3.4 Key Takeaways vs TurboMind

- Strong points:
  - Rich **prefix tree cache** with explicit eviction strategies and EAGLE‑aware bigram mode.
  - Scheduler tightly integrated with KV cache shape (tree of prompts / program fragments).
  - Designed for multi‑stage “programs” rather than just single prompts.

- Differences versus TurboMind:
  - TurboMind has powerful KV utilities and SpecPV but no Radix‑style tree cache at the engine level yet.
  - sglang’s scheduler is more deeply integrated with KV tree topology for complex workflows.

---

## 4. TensorRT‑LLM Engine (LM/TensorRT-LLM)

### 4.1 ModelEngine (Python executor)

- **ModelEngine** (`tensorrt_llm/_torch/pyexecutor/model_engine.py`)
  - Abstract base + concrete implementation around TensorRT‑LLM compiled models.
  - Responsibilities:
    - Initialize model from checkpoints via `ModelLoader`.
    - Configure KV cache (`KVCacheParams`) and attention backend.
    - Configure speculative decoding (EAGLE3 and MTP) via `SpecMetadata`, `Eagle3SpecMetadata`, `SpecDecodingTensor`.
    - Integrate CUDA Graphs and `torch.compile`:
      - Batch‑size‑specific CUDA graphs for decode and speculative decode.
      - Piecewise CUDA graph capture for prefill steps.
    - Manage per‑batch buffers for draft tokens, gather indices, accepted tokens, etc.
  - Warmup:
    - `_capture_cuda_graphs`: captures generation‑only CUDA graphs for different batch sizes and draft lengths.
    - `_capture_piecewise_cuda_graphs`: captures prefill graphs for various token counts.

### 4.2 Scheduler

- **Scheduler** (`tensorrt_llm/_torch/pyexecutor/scheduler.py`)
  - `ScheduledRequests`:
    - Holds `context_requests`, `generation_requests`, `paused_requests`.
    - Indicates whether a batch is generation‑only and whether CUDA graphs can be used.
  - `RequestScheduler` & `CapacityScheduler`:
    - Interface for scheduling active requests and enforcing capacity constraints.
    - `BindCapacityScheduler` delegates to C++ impl (`tb_internal.algorithms.CapacityScheduler`):
      - Enforces KV cache capacity and ensures requests can complete given current resources.
  - `MicroBatchScheduler`:
    - Schedules microbatches to respect `max_batch_size` and `max_num_tokens`.
    - `BindMicroBatchScheduler` delegates to C++ via `tb_internal.algorithms.MicroBatchScheduler`.

### 4.3 Key Takeaways vs TurboMind

- Strong points:
  - Tight integration with TensorRT for highly fused kernels.
  - CUDA Graphs and `torch.compile` used heavily to reduce runtime overhead.
  - Scheduler aware of KV cache capacity through a dedicated KVCacheManager.

- Differences versus TurboMind:
  - TensorRT‑LLM favors ahead‑of‑time compilation and fixed shapes for maximum throughput.
  - TurboMind is more flexible but currently makes less use of compiled CUDA graphs and static scheduling.

---

## 5. EasyDeL eSurge Engine (LM/EasyDeL)

### 5.1 Engine Architecture

- **eSurge** (`easydel/inference/esurge/esurge_engine.py`)
  - High‑level JAX engine for inference:
    - Continuous batching with a background scheduler thread.
    - Paged attention with configurable page size.
    - Prefix caching and context management.
    - Streaming generation (`delta_text`, `accumulated_text`, TTFT, tokens/sec).
  - Threading model:
    - Main thread:
      - Accepts API calls (`generate`, `stream`) and submits `EngineRequest`s.
    - Scheduler thread:
      - Runs a loop:
        - Pulls ready requests from queues.
        - Invokes `Scheduler.schedule()` to build a batch.
        - Executes model asynchronously via `eSurgeRunner.execute_model_async`.
        - Drains futures and updates `RequestOutput`s.

### 5.2 Scheduling and Context Management

- Configuration:
  - `max_model_len`, `max_num_seqs`, `max_num_batched_tokens`.
  - `page_size` for KV cache pages (recommended ≥ 256 for GPUs).
  - `reserve_tokens` to keep headroom in context window to avoid OOM / capacity errors.
  - `auto_truncate_prompt`, `auto_cap_new_tokens`, `strict_context`, `truncate_mode` (left/right/middle).
  - `enable_prefix_caching`, `destroy_pages_on_pause`, `overlap_execution`.
- Behavior:
  - Continuous batching with optional overlap between scheduler work and model execution.
  - Context‑aware adjustments (truncation / capping) to maintain safety and utilization.

### 5.3 Key Takeaways vs TurboMind

- Strong points:
  - Modern scheduling with continuous batching and context‑aware prompt management.
  - JAX/XLA backend with optional AOT compilation (`use_aot_forward`, `compile_runner`).
  - Good monitoring and metrics built in.

- Differences versus TurboMind:
  - eSurge couples scheduler and context manager tightly with the JAX runner.
  - TurboMind currently relies more on external orchestration (Gateway + backend loop) and less on explicit context policies.

---

## 6. Summary: What These Engines Do That TurboMind Must Match or Exceed

The engines above share several themes that are critical for GPT‑OSS‑120B performance and speculative decoding:

- **Paged / block KV cache with prefix reuse**
  - vLLM: PagedAttention with a dedicated KV cache manager and FlashInfer integration.
  - sglang: RadixCache and variants (HiRadix, SWARadix, Mamba) with explicit eviction policies and EAGLE bigram mode.
  - TensorRT‑LLM: KVCacheManager integrated with TensorRT kernels and attention backends.
  - EasyDeL eSurge: JAX paged KV with configurable page size and prefix caching.
  - TurboMind:
    - Has KV cache utilities, NVFP4 KV compression, SpecPV partial KV, and FlashMLA support.
    - Needs a clearer, first‑class KV allocator / prefix cache abstraction at the engine level to rival vLLM/sglang, especially for large models with heavy sharing.

- **Continuous batching + smart prefill/decode scheduling**
  - vLLM:
    - Chunked prefill, partial prefill concurrency, hybrid scheduler (FCFS/priority).
  - sglang:
    - Scheduler deeply integrated with Radix cache and disaggregation (prefill/decode offload).
  - TensorRT‑LLM:
    - Capacity scheduler and micro‑batch scheduler aware of KV capacity and CUDA graph suitability.
  - EasyDeL:
    - Background scheduler with configurable batch size / token budgets and optional overlap execution.
  - TurboMind:
    - Gateway + RequestQueue model provides solid basic scheduling.
    - To reach vLLM/sglang/TRT‑LLM performance for GPT‑OSS‑120B, the scheduler must:
      - Explicitly track prefill vs decode phases and chunked prefill.
      - Maintain per‑step token budgets and use continuous admission.
      - Become KV‑capacity‑aware for safe high‑load operation.

- **GPU‑centric pipelines and kernel fusion**
  - vLLM: FlashAttention/FlashInfer kernels and GPU‑only sampling path.
  - TensorRT‑LLM: TensorRT fused kernels, CUDA Graphs, and `torch.compile`.
  - EasyDeL: JAX/XLA fused graph execution with optional AOT.
  - TurboMind:
    - Already has specialized kernels (FlashMLA, NVFP4, speculative decoding). Further gains will come from:
      - More aggressive kernel fusion on attention/MLP/sampling paths.
      - Optional CUDA Graphs around decode loops.

- **Speculative decoding (EAGLE / EAGLE3 / MTP)**
  - vLLM and sglang: support EAGLE‑style or MTP‑style speculative decoding via external draft models or built‑in mechanisms.
  - TensorRT‑LLM:
    - Native EAGLE3 support via `Eagle3ResourceManager` and specialized kernels.
  - TurboMind:
    - Implements EAGLE/EAGLE3 kernels and SpecPV partial KV in C++.
    - Engine‑level interfaces (generation config, KV rewind/commit, scheduler integration) are present but can be tightened and better documented.

This ENGINE.md captures how LMDeploy/TurboMind is structured today and how peer engines in this repo approach KV management, scheduling, and speculative decoding. It is intended as a living reference while we optimize GPT‑OSS‑120B and EAGLE3 paths to surpass vLLM, sglang, TensorRT‑LLM, and EasyDeL on real workloads.
