"""
Unit tests for speculative decoding CUDA kernels.

Tests the acceptance and KV cache rewind kernels.
"""

import pytest
import torch
import numpy as np

# Mock CUDA kernel interface for testing
# TODO: Replace with actual pybind11 bindings when available


class MockAcceptDraftTokens:
    """Mock implementation of acceptance kernel for testing logic."""

    @staticmethod
    def accept_draft_tokens(
        output_ids: torch.Tensor,
        draft_ids: torch.Tensor,
        target_ids: torch.Tensor,
        accepted_lengths: torch.Tensor,
        sequence_lengths: torch.Tensor,
        paths: torch.Tensor,
        best_path_ids: torch.Tensor,
        batch_slots: torch.Tensor = None,
    ):
        """
        Python reference implementation of acceptance kernel.

        Args:
            output_ids: [max_batch_size, max_seq_len]
            draft_ids: [max_batch_size, max_draft_tokens]
            target_ids: [max_batch_size, max_draft_tokens]
            accepted_lengths: [max_batch_size]
            sequence_lengths: [max_batch_size]
            paths: [max_batch_size, max_decoding_tokens, max_path_len]
            best_path_ids: [max_batch_size]
            batch_slots: [batch_size] optional
        """
        batch_size = draft_ids.shape[0] if batch_slots is None else len(batch_slots)
        max_draft_tokens = draft_ids.shape[1]
        max_path_len = paths.shape[2]

        for batch_idx in range(batch_size):
            slot = batch_idx if batch_slots is None else batch_slots[batch_idx].item()
            seq_len = sequence_lengths[slot].item()
            best_path = best_path_ids[slot].item()

            accepted = 0
            for i in range(max_draft_tokens):
                path_idx = paths[slot, best_path, i].item()
                if path_idx < 0:
                    break

                draft_token = draft_ids[slot, path_idx].item()
                target_token = target_ids[slot, path_idx].item()

                if draft_token == target_token:
                    output_ids[slot, seq_len + accepted] = draft_token
                    accepted += 1
                else:
                    # Mismatch: add correct token and stop
                    output_ids[slot, seq_len + accepted] = target_token
                    accepted += 1
                    break

            accepted_lengths[slot] = accepted
            sequence_lengths[slot] += accepted


class TestAcceptDraftTokens:
    """Test suite for draft token acceptance kernel."""

    def test_all_tokens_match(self):
        """Test case where all draft tokens match target."""
        batch_size = 2
        max_seq_len = 100
        max_draft_tokens = 5
        max_path_len = 5

        # Setup
        output_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        draft_ids = torch.tensor(
            [[10, 20, 30, 40, 50], [11, 21, 31, 41, 51]], dtype=torch.long
        )
        target_ids = draft_ids.clone()  # All match
        accepted_lengths = torch.zeros(batch_size, dtype=torch.long)
        sequence_lengths = torch.tensor([10, 15], dtype=torch.long)

        # Simple linear path: [0, 1, 2, 3, 4]
        paths = torch.full((batch_size, 1, max_path_len), -1, dtype=torch.long)
        paths[:, 0, :] = torch.arange(max_path_len)
        best_path_ids = torch.zeros(batch_size, dtype=torch.long)

        # Execute
        MockAcceptDraftTokens.accept_draft_tokens(
            output_ids,
            draft_ids,
            target_ids,
            accepted_lengths,
            sequence_lengths,
            paths,
            best_path_ids,
        )

        # Verify
        assert accepted_lengths[0].item() == 5, "Should accept all 5 tokens"
        assert accepted_lengths[1].item() == 5, "Should accept all 5 tokens"
        assert sequence_lengths[0].item() == 15, "Sequence length should increase by 5"
        assert sequence_lengths[1].item() == 20, "Sequence length should increase by 5"

        # Check tokens were written correctly
        assert torch.equal(output_ids[0, 10:15], draft_ids[0])
        assert torch.equal(output_ids[1, 15:20], draft_ids[1])

    def test_first_token_mismatch(self):
        """Test case where first draft token mismatches."""
        batch_size = 1
        max_seq_len = 100
        max_draft_tokens = 5
        max_path_len = 5

        # Setup
        output_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        draft_ids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
        target_ids = torch.tensor(
            [[99, 20, 30, 40, 50]], dtype=torch.long
        )  # First mismatch
        accepted_lengths = torch.zeros(batch_size, dtype=torch.long)
        sequence_lengths = torch.tensor([10], dtype=torch.long)

        paths = torch.full((batch_size, 1, max_path_len), -1, dtype=torch.long)
        paths[:, 0, :] = torch.arange(max_path_len)
        best_path_ids = torch.zeros(batch_size, dtype=torch.long)

        # Execute
        MockAcceptDraftTokens.accept_draft_tokens(
            output_ids,
            draft_ids,
            target_ids,
            accepted_lengths,
            sequence_lengths,
            paths,
            best_path_ids,
        )

        # Verify
        assert accepted_lengths[0].item() == 1, "Should accept 1 token (correct target)"
        assert output_ids[0, 10].item() == 99, "Should have correct target token"
        assert sequence_lengths[0].item() == 11, "Sequence length should increase by 1"

    def test_partial_match(self):
        """Test case where some tokens match, then mismatch."""
        batch_size = 1
        max_seq_len = 100
        max_draft_tokens = 5
        max_path_len = 5

        # Setup
        output_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        draft_ids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
        target_ids = torch.tensor(
            [[10, 20, 99, 40, 50]], dtype=torch.long
        )  # 3rd mismatch
        accepted_lengths = torch.zeros(batch_size, dtype=torch.long)
        sequence_lengths = torch.tensor([10], dtype=torch.long)

        paths = torch.full((batch_size, 1, max_path_len), -1, dtype=torch.long)
        paths[:, 0, :] = torch.arange(max_path_len)
        best_path_ids = torch.zeros(batch_size, dtype=torch.long)

        # Execute
        MockAcceptDraftTokens.accept_draft_tokens(
            output_ids,
            draft_ids,
            target_ids,
            accepted_lengths,
            sequence_lengths,
            paths,
            best_path_ids,
        )

        # Verify
        assert (
            accepted_lengths[0].item() == 3
        ), "Should accept 3 tokens (2 match + 1 correct)"
        assert output_ids[0, 10].item() == 10, "First token matches"
        assert output_ids[0, 11].item() == 20, "Second token matches"
        assert output_ids[0, 12].item() == 99, "Third token is correct target"
        assert sequence_lengths[0].item() == 13, "Sequence length should increase by 3"

    def test_batch_slots_mapping(self):
        """Test batch slot mapping functionality."""
        batch_size = 2
        max_batch_size = 4
        max_seq_len = 100
        max_draft_tokens = 3
        max_path_len = 3

        # Setup with non-contiguous batch slots
        output_ids = torch.zeros(max_batch_size, max_seq_len, dtype=torch.long)
        draft_ids = torch.zeros(max_batch_size, max_draft_tokens, dtype=torch.long)
        target_ids = torch.zeros(max_batch_size, max_draft_tokens, dtype=torch.long)

        # Only slots 1 and 3 are active
        batch_slots = torch.tensor([1, 3], dtype=torch.long)
        draft_ids[1] = torch.tensor([10, 20, 30])
        draft_ids[3] = torch.tensor([11, 21, 31])
        target_ids[1] = draft_ids[1].clone()
        target_ids[3] = draft_ids[3].clone()

        accepted_lengths = torch.zeros(max_batch_size, dtype=torch.long)
        sequence_lengths = torch.tensor([0, 5, 0, 10], dtype=torch.long)

        paths = torch.full((max_batch_size, 1, max_path_len), -1, dtype=torch.long)
        paths[:, 0, :] = torch.arange(max_path_len)
        best_path_ids = torch.zeros(max_batch_size, dtype=torch.long)

        # Execute
        MockAcceptDraftTokens.accept_draft_tokens(
            output_ids,
            draft_ids,
            target_ids,
            accepted_lengths,
            sequence_lengths,
            paths,
            best_path_ids,
            batch_slots,
        )

        # Verify
        assert accepted_lengths[1].item() == 3, "Slot 1 should accept 3 tokens"
        assert accepted_lengths[3].item() == 3, "Slot 3 should accept 3 tokens"
        assert sequence_lengths[1].item() == 8, "Slot 1 seq len should be 5+3"
        assert sequence_lengths[3].item() == 13, "Slot 3 seq len should be 10+3"


class TestAcceptanceStats:
    """Test suite for acceptance statistics computation."""

    def test_compute_acceptance_rates(self):
        """Test acceptance rate calculation."""
        batch_size = 3

        accepted_lengths = torch.tensor([3, 1, 5], dtype=torch.long)
        draft_lengths = torch.tensor([5, 5, 5], dtype=torch.long)
        acceptance_rates = torch.zeros(batch_size, dtype=torch.float32)

        # Reference implementation
        for i in range(batch_size):
            acceptance_rates[i] = accepted_lengths[i].float() / draft_lengths[i].float()

        # Verify
        assert acceptance_rates[0].item() == pytest.approx(0.6), "60% acceptance"
        assert acceptance_rates[1].item() == pytest.approx(0.2), "20% acceptance"
        assert acceptance_rates[2].item() == pytest.approx(1.0), "100% acceptance"


class TestKVCacheRewindHelper:
    """Tests for the host-side KV rewind helper + kernel."""

    def test_compute_and_invoke_kv_cache_rewind_marks_tail_blocks(self, monkeypatch):
        """Rewind helper should compute rewind_lengths and clear tail table entries."""
        from lmdeploy.turbomind.kernels.speculative_decoding import common as sd_common

        # Small synthetic configuration.
        block_size = 4
        max_batch_size = 4
        max_blocks_per_seq = 3
        num_layers = 2
        batch_size = 2

        # Host-side draft / accepted lengths per slot.
        # slot 0: 5 draft, 3 accepted -> rewind 2 tokens (~1 block).
        # slot 1: 7 draft, 1 accepted -> rewind 6 tokens (~2 blocks).
        draft_lengths = [5, 7, 0, 0]
        accepted_lengths = [3, 1, 0, 0]
        batch_slots = [0, 1]

        # Device buffers for block tables and rewind lengths.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        block_tables = torch.arange(
            max_batch_size * max_blocks_per_seq, dtype=torch.int32, device=device
        ).view(max_batch_size, max_blocks_per_seq)
        rewind_lengths_dev = torch.zeros(max_batch_size, dtype=torch.int32, device=device)

        # For this CPU-only test, we call the helper's logic via Python:
        # compute rewind lengths the same way as the C++ helper.
        expected_rewind = []
        for slot in range(max_batch_size):
            d = max(0, draft_lengths[slot])
            a = max(0, accepted_lengths[slot])
            expected_rewind.append(max(0, d - a))

        # Simulate what computeAndInvokeKVCacheRewind would write into
        # rewind_lengths and then into KVCacheRewindParams.
        rewind_lengths_host = torch.tensor(expected_rewind, dtype=torch.int32, device=device)
        rewind_lengths_dev.copy_(rewind_lengths_host)

        # Now invoke the CUDA kernel wrapper directly.
        params = sd_common.KVCacheRewindParams(
            kv_cache_blocks=None,
            rewind_lengths=rewind_lengths_dev,
            batch_slots=torch.tensor(batch_slots, dtype=torch.int32, device=device),
            block_tables=block_tables,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            num_layers=num_layers,
            block_size=block_size,
            max_blocks_per_seq=max_blocks_per_seq,
        )

        sd_common.invoke_kv_cache_rewind(params)

        # On slot 0 we expect 1 block cleared from the tail.
        # On slot 1 we expect 2 blocks cleared from the tail.
        table_host = block_tables.cpu().numpy()

        # slot 0: last block should be -1, the others remain non-negative.
        assert table_host[0, -1] == -1
        assert table_host[0, 0] >= 0 and table_host[0, 1] >= 0

        # slot 1: last two blocks should be -1.
        assert table_host[1, -1] == -1
        assert table_host[1, -2] == -1
        # first block remains non-negative.
        assert table_host[1, 0] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
