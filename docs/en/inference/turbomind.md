# Architecture of TurboMind

TurboMind is an inference engine that supports high throughput inference for conversational LLMs. It's based on NVIDIA's [FasterTransformer](https://github.com/NVIDIA/FasterTransformer). Major features of TurboMind include an efficient LLaMa implementation, the persistent batch inference model and an extendable KV cache manager.

## High level overview of TurboMind

```
  +--------------------+
  |        API         |
  +--------------------+
          |    ^
  request |    | stream callback
          v    |
  +--------------------+   fetch   +-------------------+
  |  Persistent Batch  | <-------> |  KV Cache Manager |
  +--------------------+   update  +-------------------+
             ^
             |
             v
+------------------------+
|  LLaMA implementation  |
+------------------------+
| FT kernels & utilities |
+------------------------+
```

## Persistent Batch

You may recognize this feature as "continuous batching" in other repos. But during the concurrent development of the feature, we modeled the inference of a conversational LLM as a persistently running batch whose lifetime spans the entire serving process, hence the name "persistent batch". To put it simply

- The persistent batch as N pre-configured batch slots.
- Requests join the batch when there are free slots available. A batch slot is released and can be reused once the generation of the requested tokens is finished.
- __On cache-hits (see below), history tokens don't need to be decoded in every round of a conversation; generation of response tokens will start instantly.__
- The batch grows or shrinks automatically to minimize unnecessary computations.

## KV Cache Manager

The [KV cache manager](https://github.com/InternLM/lmdeploy/blob/main/src/turbomind/models/llama/SequenceManager.h) of TurboMind is a memory-pool-liked object that also implements LRU policy so that it can be viewed as a form of __cache of KV caches__. It works in the following way

- All device memory required for KV cache is allocated by the manager. A fixed number of slots is pre-configured to match the memory size of the system. Each slot corresponds to the memory required by the KV cache of a single sequence. Allocation chunk-size can be configure to implement pre-allocate/on-demand style allocation policy (or something in-between).
- When space for the KV cache of a new sequence is requested but no free slots left in the pool, the least recently used sequence is evicted from the cache and its device memory is directly reused by the new sequence. However, this is not the end of the story.
- Fetching sequence currently resides in one of the slots resembles a _cache-hit_, the history KV cache is returned directly and no context decoding is needed.
- Victim (evicted) sequences are not erased entirely but converted to its most compact form, i.e. token IDs. When the same sequence id is fetched later (_cache-miss_) the token IDs will be decoded by FMHA backed context decoder and converted back to KV cache.
- The eviction and conversion are handled automatically inside TurboMind and thus transparent to the users. __From the user's aspect, system that use TurboMind has access to infinite device memory.__

## LLaMa implementation

Our implementation of the LLaMa family models is modified from Gpt-NeoX model in FasterTransformer. In addition to basic refactoring and modifications to support the LLaMa family, we made some improvements to enable high performance inference of conversational models, most importantly:

- To support fast context decoding in multi-round conversations. We replaced the attention implementation in context decoder with a [cutlass](https://github.com/NVIDIA/cutlass)-based FMHA implementation that supports mismatched Q/K lengths.
- We introduced indirect buffer pointers in both context FMHA and generation FMHA to support the discontinuity in KV cache within the batch.
- To support concurrent inference with persistent batch, new synchronization mechanism was designed to orchestrate the worker threads running in tensor parallel mode.
- To maximize the throughput, we implement INT8 KV cache support to increase the max batch size. It's effective because in real-world serving scenarios, KV cache costs more memory and consumes more memory bandwidth than weights or other activations.
- We resolved an NCCL hang issue when running multiple model instances in TP mode within a single process, NCCL APIs are now guarded by host-side synchronization barriers.

## API

TurboMind supports a Python API that enables streaming output and tensor parallel mode.

## DriftEngine usage (TurboMind advanced scheduler)

The TurboMind backend can also be driven by a new orchestration called **DriftEngine**, which adds an explicit engine‑level scheduler, KV cache manager, prefix cache, and KV‑aware capacity scheduler on top of the existing kernels. DriftEngine is exposed from Python via a dedicated config and `backend="drift"` selection.

### 1. Using DriftEngine with `pipeline`

The `lmdeploy.pipeline` API accepts a `backend_config` argument. When this is a `DriftEngineConfig` instance, the engine will run with the Drift backend:

```python
from lmdeploy import pipeline  # high-level API
from lmdeploy.messages import DriftEngineConfig, TurboMindSchedulerConfig, TurboMindKVConfig

scheduler_cfg = TurboMindSchedulerConfig(
    max_num_batched_tokens=4096,
    max_num_seqs=128,
    enable_chunked_prefill=True,
    max_num_partial_prefills=2,
    long_prefill_token_threshold=2048,
    prefer_decode_over_prefill=True,
)

kv_cfg = TurboMindKVConfig(
    kv_page_size=128,
    kv_capacity_bytes=None,          # derive from GPU memory and model layout
    prefix_cache_enabled=True,
    prefix_cache_eviction_policy='lru',
)

backend_cfg = DriftEngineConfig(
    model_path="your-turbomind-model-or-gpt-oss-120b",
    tp=1,
    pp=1,
    session_len=8192,
    max_batch_size=256,
    dtype="fp16",
    scheduler=scheduler_cfg,
    kv=kv_cfg,
    prefer_high_throughput=True,
    target_latency_ms_p50=50,
    target_latency_ms_p95=200,
)

with pipeline(model_path="your-turbomind-model-or-gpt-oss-120b",
              backend_config=backend_cfg) as pipe:
    responses = pipe(["hello", "world"])
    print(responses)
```

Key points:

- **Backend selection**  
  - `pipeline` inspects `backend_config`; when it is a `DriftEngineConfig`, the underlying engine uses `backend="drift"` internally instead of the legacy TurboMind loop.
- **Scheduler knobs**  
  - `TurboMindSchedulerConfig` drives the C++ `SchedulerConfig` and `EngineScheduler`: per‑step token budget, max sequences, chunked prefill, and decode‑vs‑prefill bias.
- **KV / prefix knobs**  
  - `TurboMindKVConfig` drives `KVLayout`, `KVCacheManager`, `PrefixCache`, and `CapacityScheduler`: KV page size, capacity, and whether prefix caching + LRU eviction are enabled.

For the full list of fields and their intended semantics, see `LM/lmdeploy/ENGINE.md` and `LM/lmdeploy/ENGINE_TODOS.md` (Sections 1.1, 1.2, 2.1, 2.2, 4.1, 6.1–6.4).

### 2. Using DriftEngine with `drift_api_server`

For serving, you can either pass a `DriftEngineConfig` into `lmdeploy.serve` or use the convenience wrapper `drift_api_server`:

```python
from lmdeploy.api import drift_api_server
from lmdeploy.messages import DriftEngineConfig

backend_cfg = DriftEngineConfig(
    model_path="your-turbomind-model-or-gpt-oss-120b",
    tp=1,
    pp=1,
    session_len=8192,
)

server = drift_api_server(
    model_path="your-turbomind-model-or-gpt-oss-120b",
    model_name="drift-engine-model",
    backend_config=backend_cfg,
    server_name="0.0.0.0",
    server_port=23333,
    log_level="INFO",
)
```

The wrapper simply calls `serve(..., backend="drift", backend_config=backend_cfg, ...)`. Existing OpenAI‑compatible clients can talk to this server in the same way they do for the TurboMind or PyTorch backends; the difference is entirely in how requests are scheduled and how KV is managed internally.

> **Note**  
> DriftEngine is designed as a **non‑speculative** engine for GPT‑OSS‑120B (and related large models). Speculative decoding (EAGLE/EAGLE3/SpecPV) is configured separately via `SpeculativeConfig` and the tasks in `EAGLE_TODOS.md` / `SPECPV_TODO.md`.

## Difference between FasterTransformer and TurboMind

Apart of the features described above, there are still many minor differences that we don't cover in this document. Notably, many capabilities of FT are dropped in TurboMind because of the difference in objectives (e.g. prefix prompt, beam search, context embedding, sparse GEMM, GPT/T5/other model families, etc)

## FAQ

### Supporting Huggingface models

For historical reasons, TurboMind's weight layout is based on [the original LLaMa implementation](https://github.com/facebookresearch/llama) (differ only by a transpose). The implementation in huggingface transformers uses a [different layout](https://github.com/huggingface/transformers/blob/45025d92f815675e483f32812caa28cce3a960e7/src/transformers/models/llama/convert_llama_weights_to_hf.py#L123C76-L123C76) for `W_q` and `W_k` which is handled in [deploy.py](https://github.com/InternLM/lmdeploy/blob/ff4648a1d09e5aec74cf70efef35bfaeeac552e0/lmdeploy/serve/turbomind/deploy.py#L398).
