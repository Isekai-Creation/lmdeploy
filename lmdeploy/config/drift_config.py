from __future__ import annotations

from typing import Any, Dict

import os
from typing import Any, Dict

import torch

from lmdeploy.messages import DriftEngineConfig, TurboMindKVConfig, TurboMindSchedulerConfig


def to_cpp_drift_engine_config(cfg: DriftEngineConfig) -> Dict[str, Any]:
    """Convert Python DriftEngineConfig to a dict consumed by C++ binding."""
    kv = cfg.kv
    sched = cfg.scheduler

    # Let the C++ DriftEngine derive KV capacity from TM_CACHE_MAX_ENTRY_COUNT
    # and current free device memory after weights are loaded. Python should
    # not guess KV capacity up front based on pre-weight free memory, as this
    # can easily over-allocate on large models (e.g. 120B) and cause OOM.
    kv_capacity = getattr(kv, "kv_capacity_bytes", 0) or 0

    args: Dict[str, Any] = {
        "model_path": cfg.model_path,
        "tp": cfg.tp,
        "pp": cfg.pp,
        "session_len": cfg.session_len,
        "max_batch_size": cfg.max_batch_size,
        "dtype": cfg.dtype,
        "scheduler": {
            "max_num_batched_tokens": sched.max_num_batched_tokens,
            "max_num_seqs": sched.max_num_seqs,
            "enable_chunked_prefill": sched.enable_chunked_prefill,
            "max_num_partial_prefills": sched.max_num_partial_prefills,
            "long_prefill_token_threshold": sched.long_prefill_token_threshold,
            "prefer_decode_over_prefill": sched.prefer_decode_over_prefill,
        },
        "kv": {
            "kv_page_size": kv.kv_page_size,
            "kv_capacity_bytes": kv_capacity,
            "prefix_cache_enabled": getattr(kv, "prefix_cache_enabled", True),
        },
        "prefer_high_throughput": cfg.prefer_high_throughput,
        "target_latency_ms_p50": cfg.target_latency_ms_p50,
        "target_latency_ms_p95": cfg.target_latency_ms_p95,
        "max_queued_requests": cfg.max_queued_requests,
        "abort_on_oom": cfg.abort_on_oom,
        "quant_policy": cfg.quant_policy,
        "cache_max_entry_count": cfg.cache_max_entry_count,
        "log_level": cfg.log_level,
        "enable_prefix_caching": getattr(kv, "prefix_cache_enabled", cfg.enable_prefix_caching),
        "enable_speculative_decoding": cfg.enable_speculative_decoding,
        "enable_cuda_graphs": cfg.enable_cuda_graphs,
    }

    # Optional model layout overrides derived from the TurboMind model
    # config in turbomind._setup_drift_engine. When present, these
    # allow the C++ DriftEngine to derive a KV layout that matches
    # the converted model (e.g., GPT-OSS-20B vs GPT-OSS-120B) instead
    # of always falling back to the static 120B layout.
    num_layers = getattr(cfg, "_tm_num_layers", None)
    num_kv = getattr(cfg, "_tm_num_kv_heads", None)
    head_dim = getattr(cfg, "_tm_head_dim", None)
    page_size = getattr(cfg.kv, "kv_page_size", None)
    if num_layers and num_kv and head_dim:
        ml: Dict[str, Any] = {
            "num_layers": int(num_layers),
            "num_kv_heads": int(num_kv),
            "head_dim": int(head_dim),
        }
        if page_size and page_size > 0:
            ml["page_size"] = int(page_size)
        args["model_layout"] = ml

    return args


__all__ = ["DriftEngineConfig", "TurboMindKVConfig", "TurboMindSchedulerConfig", "to_cpp_drift_engine_config"]
