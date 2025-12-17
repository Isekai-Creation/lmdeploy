from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DriftMetricsView:
    ema_tokens_per_second: float
    ema_p50_latency_ms: float
    ema_p95_latency_ms: float
    step_prefill_tokens: int
    step_decode_tokens: int
    queued_prefill: int
    queued_decode: int
    active_requests: int
    kv_total_pages: int
    kv_used_pages: int
    kv_free_pages: int
    kv_blocked: int
    kv_rejected: int
    prefix_hits: int
    prefix_misses: int
    prefix_evictions: int
    prefix_bytes_evicted: int

    @classmethod
    def from_cpp(cls, metrics: Any) -> "DriftMetricsView":
        return cls(
            ema_tokens_per_second=float(getattr(metrics, "ema_tokens_per_second", 0.0)),
            ema_p50_latency_ms=float(getattr(metrics, "ema_p50_latency_ms", 0.0)),
            ema_p95_latency_ms=float(getattr(metrics, "ema_p95_latency_ms", 0.0)),
            step_prefill_tokens=int(getattr(metrics, "step_prefill_tokens", 0)),
            step_decode_tokens=int(getattr(metrics, "step_decode_tokens", 0)),
            queued_prefill=int(getattr(metrics, "queued_prefill", 0)),
            queued_decode=int(getattr(metrics, "queued_decode", 0)),
            active_requests=int(getattr(metrics, "active_requests", 0)),
            kv_total_pages=int(getattr(metrics, "kv_total_pages", 0)),
            kv_used_pages=int(getattr(metrics, "kv_used_pages", 0)),
            kv_free_pages=int(getattr(metrics, "kv_free_pages", 0)),
            kv_blocked=int(getattr(metrics, "kv_blocked", 0)),
            kv_rejected=int(getattr(metrics, "kv_rejected", 0)),
            prefix_hits=int(getattr(metrics, "prefix_hits", 0)),
            prefix_misses=int(getattr(metrics, "prefix_misses", 0)),
            prefix_evictions=int(getattr(metrics, "prefix_evictions", 0)),
            prefix_bytes_evicted=int(getattr(metrics, "prefix_bytes_evicted", 0)),
        )

    def asdict(self) -> dict[str, Any]:
        return asdict(self)
