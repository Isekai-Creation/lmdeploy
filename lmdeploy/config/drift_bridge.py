# Python bridge to construct DriftEngine config dict for _turbomind binding
# Maps lmdeploy.messages.DriftEngineConfig → flat dict for create_drift_engine

from __future__ import annotations
import math
from typing import Any, Dict

from lmdeploy.messages import DriftEngineConfig as PyDriftEngineConfig


def drift_config_to_cpp_dict(cfg: PyDriftEngineConfig) -> Dict[str, Any]:
    """Convert Python DriftEngineConfig to flat dict for C++ binding."""
    if not isinstance(cfg, PyDriftEngineConfig):
        raise TypeError(f"Expected DriftEngineConfig, got {type(cfg)}")

    # Core fields
    args: Dict[str, Any] = {
        "model_path": cfg.model_path,
        "tp": cfg.tp or 1,
        "pp": cfg.pp or 1,
        "session_len": cfg.session_len,
        "max_batch_size": cfg.max_batch_size,
        "dtype": cfg.dtype or "auto",
    }

    schedule_policy = "fcfs"
    scheduler_spec_enable = False
    spec_method = (getattr(cfg, "spec_method", "none") or "none").lower()
    max_draft_tokens = getattr(cfg, "spec_max_draft_tokens", 0) or 0

    # Scheduler config (flatten nested fields)
    sched_defaults = dict(
        max_num_batched_tokens=2048,
        max_num_seqs=128,
        enable_chunked_prefill=True,
        max_num_partial_prefills=1,
        max_long_partial_prefills=1,
        long_prefill_token_threshold=0,
        prefer_decode_over_prefill=True,
    )
    if hasattr(cfg, "scheduler") and cfg.scheduler:
        s = cfg.scheduler
        policy_value = getattr(s, "schedule_policy", None) or getattr(s, "scheduler_policy", None)
        if policy_value is not None and hasattr(policy_value, "name"):
            policy_value = policy_value.name
        if policy_value:
            schedule_policy = str(policy_value).lower()
        if schedule_policy == "priority":
            schedule_policy = "small_first"
        if schedule_policy not in ("fcfs", "small_first"):
            schedule_policy = "fcfs"

        scheduler_spec_enable = bool(getattr(s, "enable_speculative_decoding", False))
        spec_method_value = getattr(s, "spec_method", None)
        if spec_method_value:
            spec_method = str(spec_method_value).lower()
        sched_draft = getattr(s, "max_draft_tokens_per_seq", None)
        if isinstance(sched_draft, (int, float)) and sched_draft >= 0:
            max_draft_tokens = int(sched_draft)

        args.update({
            "max_num_batched_tokens": getattr(s, "max_num_batched_tokens", sched_defaults["max_num_batched_tokens"]),
            "max_num_seqs": getattr(s, "max_num_seqs", sched_defaults["max_num_seqs"]),
            "enable_chunked_prefill": getattr(s, "enable_chunked_prefill", sched_defaults["enable_chunked_prefill"]),
            "max_num_partial_prefills": getattr(s, "max_num_partial_prefills", sched_defaults["max_num_partial_prefills"]),
            "max_long_partial_prefills": getattr(s, "max_long_partial_prefills", sched_defaults["max_long_partial_prefills"]),
            "long_prefill_token_threshold": getattr(s, "long_prefill_token_threshold", sched_defaults["long_prefill_token_threshold"]),
            "prefer_decode_over_prefill": getattr(s, "prefer_decode_over_prefill", sched_defaults["prefer_decode_over_prefill"]),
        })
    else:
        args.update(sched_defaults)

    args["schedule_policy"] = schedule_policy
    args["scheduler_enable_speculative_decoding"] = bool(getattr(cfg, "enable_speculative_decoding", False)) or scheduler_spec_enable
    args["spec_method"] = spec_method
    args["spec_max_draft_tokens"] = max_draft_tokens
    args["max_draft_tokens_per_seq"] = max_draft_tokens

    # KV config
    kv_page_size = 256
    kv_capacity_bytes = None
    enable_prefix_caching = True
    if hasattr(cfg, "kv") and cfg.kv:
        kv = cfg.kv
        kv_page_size = getattr(kv, "kv_page_size", kv_page_size)
        kv_capacity_bytes = getattr(kv, "kv_capacity_bytes", kv_capacity_bytes)
        enable_prefix_caching = getattr(kv, "prefix_cache_enabled", enable_prefix_caching)

    if kv_capacity_bytes is None:
        kv_gb = getattr(cfg, "kv_cache_memory_gb", None)
        if kv_gb:
            kv_capacity_bytes = int(kv_gb * 1024 * 1024 * 1024)
        else:
            kv_capacity_bytes = 2 * 1024 * 1024 * 1024
    else:
        kv_capacity_bytes = int(kv_capacity_bytes)

    args.update({
        "kv_page_size": kv_page_size,
        "kv_capacity_bytes": kv_capacity_bytes,
        "enable_prefix_caching": enable_prefix_caching,
    })

    # Drift-specific knobs
    sched_obj = getattr(cfg, "scheduler", None)
    latency_p50 = getattr(cfg, "target_latency_ms_p50", None)
    latency_p95 = getattr(cfg, "target_latency_ms_p95", None)
    if latency_p50 is None and sched_obj is not None:
        latency_p50 = getattr(sched_obj, "target_latency_ms_p50", 50)
    if latency_p95 is None and sched_obj is not None:
        latency_p95 = getattr(sched_obj, "target_latency_ms_p95", 200)
    latency_p50 = latency_p50 if latency_p50 is not None else 50
    latency_p95 = latency_p95 if latency_p95 is not None else 200

    args.update({
        "prefer_high_throughput": getattr(cfg, "prefer_high_throughput", True),
        "target_latency_ms_p50": latency_p50,
        "target_latency_ms_p95": latency_p95,
        "max_queued_requests": getattr(cfg, "max_queued_requests", 4096),
        "abort_on_oom": getattr(cfg, "abort_on_oom", True),
        "log_level": getattr(cfg, "log_level", "INFO").lower(),
        "enable_specpv": getattr(cfg, "enable_specpv", False),
        "specpv_block_size": getattr(cfg, "specpv_block_size", 16),
        "specpv_n_sink_blocks": getattr(cfg, "specpv_n_sink_blocks", 2),
        "specpv_n_retrieval_blocks": getattr(cfg, "specpv_n_retrieval_blocks", 256),
        "specpv_n_window_blocks": getattr(cfg, "specpv_n_window_blocks", 8),
        "specpv_n_spec_tokens_buf": getattr(cfg, "specpv_n_spec_tokens_buf", 128),
        "specpv_partial_threshold": getattr(cfg, "specpv_partial_threshold", 4096),
        "specpv_full_refresh_steps": getattr(cfg, "specpv_full_refresh_steps", 32),
        "enable_suffix_decoding": getattr(cfg, "enable_suffix_decoding", False),
        "suffix_cache_max_depth": getattr(cfg, "suffix_cache_max_depth", 64),
        "suffix_cache_max_requests": getattr(cfg, "suffix_cache_max_requests", -1),
        "suffix_max_spec_factor": getattr(cfg, "suffix_max_spec_factor", 1.0),
        "suffix_max_spec_offset": getattr(cfg, "suffix_max_spec_offset", 0.0),
        "suffix_min_token_prob": getattr(cfg, "suffix_min_token_prob", 0.1),
    })

    args["enable_speculative_decoding"] = getattr(cfg, "enable_speculative_decoding", False)

    # Sanity checks
    if args["session_len"] <= 0:
        raise ValueError("session_len must be positive")
    if args["max_batch_size"] <= 0:
        raise ValueError("max_batch_size must be positive")
    if kv_capacity_bytes <= 0:
        raise ValueError("KV capacity must be positive")

    return args


def conservative_baseline_to_cpp_dict(model_path: str = "") -> Dict[str, Any]:
    """Convenience wrapper for DriftEngineConfig.conservative_baseline → C++ dict."""
    cfg = PyDriftEngineConfig.conservative_baseline(model_path=model_path)
    return drift_config_to_cpp_dict(cfg)


def optimized_to_cpp_dict(model_path: str = "") -> Dict[str, Any]:
    """Convenience wrapper for DriftEngineConfig.optimized → C++ dict."""
    cfg = PyDriftEngineConfig.optimized(model_path=model_path)
    return drift_config_to_cpp_dict(cfg)