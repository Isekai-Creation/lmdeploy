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

    # Scheduler config (flatten nested fields)
    sched_defaults = dict(
        max_num_batched_tokens=2048,
        max_num_seqs=128,
        enable_chunked_prefill=True,
        max_num_partial_prefills=1,
        long_prefill_token_threshold=0,
        prefer_decode_over_prefill=True,
    )
    if hasattr(cfg, "scheduler") and cfg.scheduler:
        s = cfg.scheduler
        args.update({
            "max_num_batched_tokens": getattr(s, "max_num_batched_tokens", sched_defaults["max_num_batched_tokens"]),
            "max_num_seqs": getattr(s, "max_num_seqs", sched_defaults["max_num_seqs"]),
            "enable_chunked_prefill": getattr(s, "enable_chunked_prefill", sched_defaults["enable_chunked_prefill"]),
            "max_num_partial_prefills": getattr(s, "max_num_partial_prefills", sched_defaults["max_num_partial_prefills"]),
            "long_prefill_token_threshold": getattr(s, "long_prefill_token_threshold", sched_defaults["long_prefill_token_threshold"]),
            "prefer_decode_over_prefill": getattr(s, "prefer_decode_over_prefill", sched_defaults["prefer_decode_over_prefill"]),
        })
    else:
        args.update(sched_defaults)

    # KV config
    kv_page_size = 128
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
    })

    # MVP safety: enforce non-speculative
    args["enable_speculative_decoding"] = False  # not a binding arg; just documentation

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