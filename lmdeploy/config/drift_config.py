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

    schedule_policy = getattr(sched, "schedule_policy", None)
    if schedule_policy is None and hasattr(sched, "scheduler_policy"):
        schedule_policy = getattr(sched, "scheduler_policy")
    if schedule_policy is not None and hasattr(schedule_policy, "name"):
        schedule_policy = schedule_policy.name
    schedule_policy = (str(schedule_policy).lower() if schedule_policy else "fcfs")
    if schedule_policy == "priority":
        schedule_policy = "small_first"
    if schedule_policy not in ("fcfs", "small_first"):
        schedule_policy = "fcfs"

    scheduler_spec_enable = bool(getattr(sched, "enable_speculative_decoding", False))
    scheduler_spec_method = getattr(sched, "spec_method", None)
    spec_method_value = scheduler_spec_method or cfg.spec_method or "none"
    spec_method = str(spec_method_value).lower()

    scheduler_max_draft = getattr(sched, "max_draft_tokens_per_seq", None)
    max_draft_tokens = cfg.spec_max_draft_tokens
    if isinstance(scheduler_max_draft, (int, float)) and scheduler_max_draft >= 0:
        max_draft_tokens = int(scheduler_max_draft)

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
            "max_long_partial_prefills": getattr(sched, "max_long_partial_prefills", sched.max_num_partial_prefills),
            "long_prefill_token_threshold": sched.long_prefill_token_threshold,
            "prefer_decode_over_prefill": sched.prefer_decode_over_prefill,
            "schedule_policy": schedule_policy,
            "enable_speculative_decoding": bool(cfg.enable_speculative_decoding) or scheduler_spec_enable,
            "spec_method": spec_method,
            "max_draft_tokens_per_seq": max_draft_tokens,
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
        "spec_method": spec_method,
        "spec_max_draft_tokens": max_draft_tokens,
        "enable_specpv": cfg.enable_specpv,
        "specpv_block_size": cfg.specpv_block_size,
        "specpv_n_sink_blocks": cfg.specpv_n_sink_blocks,
        "specpv_n_retrieval_blocks": cfg.specpv_n_retrieval_blocks,
        "specpv_n_window_blocks": cfg.specpv_n_window_blocks,
        "specpv_n_spec_tokens_buf": cfg.specpv_n_spec_tokens_buf,
        "specpv_partial_threshold": cfg.specpv_partial_threshold,
        "specpv_full_refresh_steps": cfg.specpv_full_refresh_steps,
        "enable_suffix_decoding": cfg.enable_suffix_decoding,
        "suffix_cache_max_depth": cfg.suffix_cache_max_depth,
        "suffix_cache_max_requests": cfg.suffix_cache_max_requests,
        "suffix_max_spec_factor": cfg.suffix_max_spec_factor,
        "suffix_max_spec_offset": cfg.suffix_max_spec_offset,
        "suffix_min_token_prob": cfg.suffix_min_token_prob,
    }

    # Optional model layout overrides derived from the TurboMind model
    # config in turbomind._setup_drift_engine. When present, these
    # allow the C++ DriftEngine to derive a KV layout that matches
    # the converted model (e.g., GPT-OSS-20B vs GPT-OSS-120B) instead
    # of always falling back to the static 120B layout.
    num_layers = getattr(cfg, "_tm_num_layers", None)
    num_kv = getattr(cfg, "_tm_num_kv_heads", None)
    head_dim = getattr(cfg, "_tm_head_dim", None)
    # Prefer an explicit TM-derived page_size override (cache_block_seq_len)
    # when available so that DriftEngine KV pages always map 1:1 to the
    # TurboMind attention block geometry, even if cfg.kv.kv_page_size is
    # later mutated by other helpers.
    page_size = getattr(cfg, "_tm_page_size", None)
    if page_size is None:
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
