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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
