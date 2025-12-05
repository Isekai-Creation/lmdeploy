"""
Unit tests for SpeculativeDecodingStats aggregation.

These tests ensure that TurboMind EAGLE metrics plumbed via
RequestMetrics.spec_info are correctly consumed by the generic
SpeculativeDecodingStats helper used on the Python side.
"""

from lmdeploy.messages import EngineOutput, RequestMetrics, ResponseType
from lmdeploy.metrics.stats import SpeculativeDecodingStats


def test_speculative_decoding_stats_update_from_output_with_spec_info():
    """update_from_output should consume spec_info and update counters."""
    req_metrics = RequestMetrics(
        token_timestamp=0.0,
        engine_events=[],
        spec_info={
            "num_draft_tokens": 10,
            "num_accepted_tokens": 6,
        },
    )
    outputs = EngineOutput(
        status=ResponseType.SUCCESS,
        token_ids=[1, 2, 3],
        req_metrics=req_metrics,
    )

    stats = SpeculativeDecodingStats(num_spec_tokens=8)
    stats.update_from_output(outputs)

    assert stats.num_drafts == 1
    assert stats.num_draft_tokens == 10
    assert stats.num_accepted_tokens == 6

    # First 6 positions should be incremented.
    assert all(stats.num_accepted_tokens_per_pos[:6] == 1)
    # Remaining positions stay at zero.
    assert all(stats.num_accepted_tokens_per_pos[6:] == 0)


def test_speculative_decoding_stats_update_from_output_without_spec_info():
    """When spec_info is absent, counters must remain unchanged."""
    # Case 1: req_metrics present but spec_info is None.
    req_metrics = RequestMetrics(token_timestamp=0.0, engine_events=[], spec_info=None)
    outputs = EngineOutput(
        status=ResponseType.SUCCESS,
        token_ids=[1],
        req_metrics=req_metrics,
    )

    stats = SpeculativeDecodingStats(num_spec_tokens=4)
    stats.update_from_output(outputs)

    assert stats.num_drafts == 0
    assert stats.num_draft_tokens == 0
    assert stats.num_accepted_tokens == 0
    assert all(stats.num_accepted_tokens_per_pos == 0)

    # Case 2: req_metrics is completely absent.
    outputs_no_metrics = EngineOutput(
        status=ResponseType.SUCCESS,
        token_ids=[1],
        req_metrics=None,
    )
    stats.update_from_output(outputs_no_metrics)

    assert stats.num_drafts == 0
    assert stats.num_draft_tokens == 0
    assert stats.num_accepted_tokens == 0
    assert all(stats.num_accepted_tokens_per_pos == 0)

