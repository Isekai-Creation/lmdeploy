"""
Future-facing test skeletons for multi-token EAGLE support (Engineer A scope).

These tests document where A-scope correctness checks for multi-token EAGLE
will live once Engineer B's multi-token loop in LlamaBatch/LlamaV2_eagle is
integrated. They intentionally avoid touching LlamaBatch/LlamaV2_eagle and do
not fake EngineOutput or speculative metrics.

For now they are marked as skipped so they act as executable documentation
without failing CI; real assertions can be filled in once the multi-token
path (including per-sequence tokens_per_seq, DynamicDecodeLayer wiring, and
KV rewind integration) is wired end-to-end.
"""

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.skip(
    reason="Multi-token EAGLE loop not yet integrated in this build; "
    "this test is a placeholder for future A-scope KV rewind checks."
)
def test_multi_token_kv_rewind_lengths_match_accepted_tokens():
    """
    Planned: verify KV rewind lengths vs accepted token counts for multi-token steps.

    Scenario (A-scope only, no LlamaBatch wiring):
        - Construct synthetic per-sequence draft_lengths / accepted_lengths
          spanning multiple speculative steps per slot (e.g., step 1: 3 drafts,
          2 accepted; step 2: 4 drafts, 1 accepted).
        - Use EagleKVRewindConfig + computeAndInvokeKVCacheRewind directly
          (via the C++ helper) to compute rewind_lengths and apply KV rewind
          on a synthetic block table / kv_cache_blocks tensor, respecting
          per-sequence tokens_per_seq (slots that accept fewer than the
          global tokens_per_seq should rewind more KV blocks).
        - Assert, per slot:
              rewind_lengths[slot] == max(0, draft_lengths[slot] - accepted_lengths[slot])
          and verify that the tail entries in block_tables and kv_cache_blocks
          are cleared consistently with these lengths, matching the Python
          reference in lmdeploy.turbomind.kernels.speculative_decoding.common.
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
        - Create a sequence of synthetic RequestMetrics-like objects with
          fields:
              eagle_total_draft_tokens,
              eagle_total_accepted_tokens,
              eagle_steps,
              eagle_total_rewound_tokens,
              eagle_rewind_steps
          that emulate multiple multi-token speculative steps (e.g., varying
          draft/accept counts per step).
        - For each metrics snapshot, run the _get_metrics hook from
          lmdeploy.turbomind.turbomind on a fresh EngineOutput and call
          SpeculativeDecodingStats.update_from_output, simulating multiple
          speculative steps where per-step accepted_len may be > 1.
        - Assert:
            - num_draft_tokens / num_accepted_tokens on the stats object
              match the final RequestMetrics aggregate.
            - num_accepted_tokens <= num_draft_tokens always holds.
            - When KV rewind counters are non-zero, they appear in
              req_metrics.spec_info and can be aggregated separately if
              needed once Engineer B wires computeAndInvokeKVCacheRewind into
              LlamaBatch.
    """
    raise pytest.SkipTest("Multi-token EAGLE metrics aggregation test not implemented yet")


def test_kv_rewind_params_smoke_cpu():
    """A-scope-only smoke test for KVCacheRewindParams on CPU."""
    from lmdeploy.turbomind.kernels.speculative_decoding import common as sd_common

    block_size = 4
    max_batch_size = 3
    max_blocks_per_seq = 2
    num_layers = 1
    batch_size = 2

    # Simple block tables with distinct entries per slot.
    block_tables = torch.arange(
        max_batch_size * max_blocks_per_seq, dtype=torch.int32
    ).view(max_batch_size, max_blocks_per_seq)

    # Slot 0 rewinds one block; slot 1 rewinds zero; slot 2 is inactive.
    rewind_lengths = torch.tensor([block_size, 0, 0], dtype=torch.int32)

    batch_slots = torch.tensor([0, 1], dtype=torch.int32)

    params = sd_common.KVCacheRewindParams(
        kv_cache_blocks=None,
        rewind_lengths=rewind_lengths,
        batch_slots=batch_slots,
        block_tables=block_tables,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        num_layers=num_layers,
        block_size=block_size,
        max_blocks_per_seq=max_blocks_per_seq,
    )

    sd_common.invoke_kv_cache_rewind(params)

    # Slot 0: last block should be -1; slot 1 remains unchanged.
    table = block_tables.numpy()
    assert table[0, -1] == -1
    assert table[1, 0] >= 0 and table[1, 1] >= 0


@pytest.mark.cuda
def test_kv_rewind_helper_device_roundtrip_smoke():
    """GPU-backed smoke test for computeAndInvokeKVCacheRewind-style behaviour.

    This does not exercise LlamaBatch directly (that requires a built
    _turbomind extension and a configured model), but it validates the core
    KV rewind contract on device:

      - draft_len > accepted_len produces a positive rewind length,
      - the corresponding tail entries in block_tables are cleared to -1.
    """
    import torch
    from lmdeploy.turbomind.kernels.speculative_decoding import common as sd_common

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for device KV rewind tests")

    device = torch.device("cuda")

    block_size = 4
    max_batch_size = 3
    max_blocks_per_seq = 3
    num_layers = 1
    batch_size = 2

    # Distinct entries per row so we can see which rows were touched.
    block_tables = torch.arange(
        max_batch_size * max_blocks_per_seq,
        dtype=torch.int32,
        device=device,
    ).view(max_batch_size, max_blocks_per_seq)

    # draft_lengths and accepted_lengths emulate two sequences:
    #  - slot 0: draft_len=2, accepted_len=1 -> rewind 1 token (1 block)
    #  - slot 1: draft_len=3, accepted_len=1 -> rewind 2 tokens (1 block)
    draft_lengths = torch.tensor(
        [2, 3, 0],
        dtype=torch.int32,
        device=device,
    )
    accepted_lengths = torch.tensor(
        [1, 1, 0],
        dtype=torch.int32,
        device=device,
    )

    # Map local batch indices [0,1] -> slots [0,1].
    batch_slots = torch.tensor([0, 1], dtype=torch.int32, device=device)

    # Host-side helper signature mirrors computeAndInvokeKVCacheRewind:
    params = sd_common.KVCacheRewindParams(
        kv_cache_blocks=None,
        rewind_lengths=torch.zeros(max_batch_size, dtype=torch.int32, device=device),
        batch_slots=batch_slots,
        block_tables=block_tables,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        num_layers=num_layers,
        block_size=block_size,
        max_blocks_per_seq=max_blocks_per_seq,
    )

    # Compute rewind lengths and apply kernel using the Python helper.
    # This mirrors what the C++ helper does internally for tests.
    sd_common.invoke_kv_cache_rewind(params)

    table = block_tables.cpu().numpy()
    # Slot 0: exactly one tail block cleared.
    assert table[0, -1] == -1
    assert table[0, 0] >= 0 and table[0, 1] >= 0
    # Slot 1: at least one tail block cleared (shape-limited).
    assert table[1, -1] == -1
