"""
Future-facing test skeletons for multi-token EAGLE support (Engineer A scope).

These tests document where A-scope correctness checks for multi-token EAGLE
will live once Engineer B's multi-token loop in LlamaBatch/LlamaV2_eagle is
integrated. They intentionally avoid touching LlamaBatch/LlamaV2_eagle and do
not fake EngineOutput or speculative metrics.

For now they are marked as skipped so they act as executable documentation
without failing CI; real assertions can be filled in once the multi-token
path is wired end-to-end.
"""

import pytest


@pytest.mark.skip(
    reason="Multi-token EAGLE loop not yet integrated in this build; "
    "this test is a placeholder for future A-scope KV rewind checks."
)
def test_multi_token_kv_rewind_lengths_match_accepted_tokens():
    """
    Planned: verify KV rewind lengths vs accepted token counts for multi-token steps.

    Scenario:
        - Construct synthetic per-sequence draft_lengths / accepted_lengths that
          correspond to multi-token speculative steps (e.g., 3 drafts, 2 accepted,
          then another step with 4 drafts, 1 accepted).
        - Use EagleKVRewindConfig + computeAndInvokeKVCacheRewind to compute
          rewind_lengths and apply KV rewind on a synthetic block table.
        - Assert:
            rewind_lengths[slot] == draft_lengths[slot] - accepted_lengths[slot]
          for all active slots, and that the tail blocks in block_tables have
          been cleared consistently with these lengths.
    """
    raise pytest.SkipTest("Multi-token EAGLE KV rewind test not implemented yet")


@pytest.mark.skip(
    reason="Multi-token EAGLE loop not yet integrated in this build; "
    "this test is a placeholder for future A-scope metrics checks."
)
def test_multi_token_eagle_metrics_aggregate_across_steps():
    """
    Planned: verify that EAGLE metrics aggregate correctly for multi-token steps.

    Scenario:
        - Create a synthetic RequestMetrics-like object with fields:
            eagle_total_draft_tokens, eagle_total_accepted_tokens,
            eagle_steps, eagle_total_rewound_tokens, eagle_rewind_steps
          representing multiple speculative steps with varying draft/accept
          counts (multi-token per step).
        - Pass it through the _get_metrics hook from lmdeploy.turbomind.turbomind
          and feed the resulting EngineOutput objects into SpeculativeDecodingStats.
        - Assert:
            - num_draft_tokens / num_accepted_tokens counters on the stats
              object match the underlying RequestMetrics aggregate.
            - num_accepted_tokens <= num_draft_tokens always holds.
            - KV rewind counters are reflected in spec_info when non-zero.
    """
    raise pytest.SkipTest("Multi-token EAGLE metrics aggregation test not implemented yet")

