"""
Unit tests for EagleMetricsSummary.

These tests ensure that EagleMetricsSummary correctly summarizes
SpeculativeDecodingStats, including acceptance rates and mean
acceptance length.
"""

from lmdeploy.metrics.stats import EagleMetricsSummary, SpeculativeDecodingStats
from lmdeploy.messages import EngineOutput, RequestMetrics, ResponseType


def test_eagle_metrics_summary_from_stats():
    """from_stats should reflect SpeculativeDecodingStats aggregates."""
    stats = SpeculativeDecodingStats(num_spec_tokens=4)

    # Simulate two drafts: first accepts 2 tokens, second accepts 1.
    # Total: 3 accepted out of 6 draft tokens.
    for num_draft, num_accept in [(3, 2), (3, 1)]:
        stats.update_per_draft(num_draft_tokens=num_draft, num_accepted_tokens=num_accept)

    summary = EagleMetricsSummary.from_stats(stats)

    assert summary.num_drafts == 2
    assert summary.num_draft_tokens == 6
    assert summary.num_accepted_tokens == 3

    # Draft acceptance rate = 3 / 6 = 0.5
    assert abs(summary.draft_acceptance_rate - 0.5) < 1e-6

    # Mean acceptance length (including bonus token) = 1 + 3/2 = 2.5
    assert abs(summary.mean_acceptance_length - 2.5) < 1e-6


def test_eagle_metrics_summary_handles_no_drafts():
    """Summary should return NaN rates when there are no drafts."""
    stats = SpeculativeDecodingStats(num_spec_tokens=4)
    summary = EagleMetricsSummary.from_stats(stats)

    assert summary.num_drafts == 0
    assert summary.num_draft_tokens == 0
    assert summary.num_accepted_tokens == 0
    assert summary.draft_acceptance_rate != summary.draft_acceptance_rate  # NaN
    assert summary.mean_acceptance_length != summary.mean_acceptance_length  # NaN

