"""
Unit tests for TurboMind EAGLE metrics plumbing.

These tests focus on the Python-side handling of C++ RequestMetrics
objects and ensure that EAGLE-specific fields are surfaced via
req_metrics.spec_info without touching LlamaBatch or LlamaV2_eagle.
"""

from types import SimpleNamespace

from lmdeploy.messages import EngineOutput, RequestMetrics, ResponseType
from lmdeploy.turbomind.turbomind import _get_metrics


def test_eagle_spec_info_populated_when_steps_positive():
    """EAGLE metrics should populate spec_info when eagle_steps > 0."""
    metrics = SimpleNamespace(
        enque_time=0,
        scheduled_time=0,
        eagle_total_draft_tokens=10,
        eagle_total_accepted_tokens=6,
        eagle_steps=3,
        eagle_total_rewound_tokens=4,
        eagle_rewind_steps=2,
    )

    hook = _get_metrics(metrics)

    out = EngineOutput(status=ResponseType.SUCCESS, token_ids=[1, 2, 3])
    hook(out, step=1)

    assert isinstance(out.req_metrics, RequestMetrics)
    spec_info = out.req_metrics.spec_info
    assert spec_info is not None
    assert spec_info["num_draft_tokens"] == 10
    assert spec_info["num_accepted_tokens"] == 6
    # Average accepted per step = 6 / 3 = 2.0
    assert abs(spec_info["avg_accepted_per_step"] - 2.0) < 1e-6
    # KV rewind metrics should be surfaced when non-zero.
    assert spec_info["num_rewound_tokens"] == 4
    assert spec_info["rewind_steps"] == 2


def test_eagle_spec_info_absent_when_no_steps():
    """When eagle_steps == 0, spec_info should remain None."""
    metrics = SimpleNamespace(
        enque_time=0,
        scheduled_time=0,
        eagle_total_draft_tokens=0,
        eagle_total_accepted_tokens=0,
        eagle_steps=0,
    )

    hook = _get_metrics(metrics)

    out = EngineOutput(status=ResponseType.SUCCESS, token_ids=[1])
    hook(out, step=1)

    assert isinstance(out.req_metrics, RequestMetrics)
    assert out.req_metrics.spec_info is None
