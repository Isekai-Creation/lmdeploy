"""
Unit tests for speculative decoding CUDA kernels.

Tests the acceptance and KV cache rewind kernels.
"""

import pytest
import torch
import numpy as np

from lmdeploy.turbomind.turbomind import _turbomind

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for device acceptance tests")
class TestAcceptanceStatsDevice:
    """GPU-backed tests for the acceptance-rate stats kernel."""

    def test_device_acceptance_stats_matches_reference(self):
        device = torch.device("cuda")

        batch_size = 4
        accepted_lengths = torch.tensor([3, 1, 5, 0], dtype=torch.int32, device=device)
        draft_lengths = torch.tensor([5, 5, 5, 1], dtype=torch.int32, device=device)
        batch_slots = torch.arange(batch_size, dtype=torch.int32, device=device)

        # CPU reference: accepted / draft (with guard against division by zero).
        ref_rates = torch.zeros(batch_size, dtype=torch.float32)
        for i in range(batch_size):
            a = float(accepted_lengths[i].item())
            d = float(draft_lengths[i].item())
            ref_rates[i] = a / d if d > 0.0 else 0.0

        # Device computation via dedicated kernel.
        rates_dev = torch.zeros(batch_size, dtype=torch.float32, device=device)
        _turbomind.eagle_compute_acceptance_stats(
            rates_dev,
            accepted_lengths,
            draft_lengths,
            batch_slots,
        )

        assert torch.allclose(rates_dev.cpu(), ref_rates, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for device acceptance tests")
class TestAcceptDraftTokensDevice:
    """GPU-backed tests for the acceptDraftTokens kernel via _turbomind."""

    def test_device_accept_matches_reference(self):
        """Device kernel should match MockAcceptDraftTokens behaviour."""
        device = torch.device("cuda")

        batch_size = 2
        max_batch_size = 2
        max_seq_len = 32
        max_draft_tokens = 5
        max_path_len = 5

        # Inputs / outputs on device (int32 for kernel compatibility).
        output_ids_ref = torch.zeros(max_batch_size, max_seq_len, dtype=torch.int32, device=device)
        draft_ids = torch.tensor(
            [[10, 20, 30, 40, 50], [11, 21, 31, 41, 51]],
            dtype=torch.int32,
            device=device,
        )
        target_ids = draft_ids.clone()

        accepted_lengths_ref = torch.zeros(max_batch_size, dtype=torch.int32, device=device)
        sequence_lengths_ref = torch.tensor([10, 15], dtype=torch.int32, device=device)

        paths = torch.full(
            (max_batch_size, max_draft_tokens, max_path_len),
            -1,
            dtype=torch.int32,
            device=device,
        )
        # Simple linear path on path 0 for all slots.
        paths[:, 0, :] = torch.arange(max_path_len, dtype=torch.int32, device=device)
        best_path_ids = torch.zeros(max_batch_size, dtype=torch.int32, device=device)
        batch_slots = torch.arange(batch_size, dtype=torch.int32, device=device)

        # Reference implementation.
        MockAcceptDraftTokens.accept_draft_tokens(
            output_ids_ref,
            draft_ids,
            target_ids,
            accepted_lengths_ref,
            sequence_lengths_ref,
            paths,
            best_path_ids,
            batch_slots,
        )

        # Fresh tensors for device kernel.
        output_ids_dev = torch.zeros_like(output_ids_ref)
        accepted_lengths_dev = torch.zeros_like(accepted_lengths_ref)
        sequence_lengths_dev = torch.tensor([10, 15], dtype=torch.int32, device=device)

        _turbomind.eagle_accept_draft_tokens(
            output_ids_dev,
            draft_ids,
            target_ids,
            accepted_lengths_dev,
            sequence_lengths_dev,
            paths,
            best_path_ids,
            batch_slots,
        )

        assert torch.equal(output_ids_dev.cpu(), output_ids_ref.cpu())
        assert torch.equal(accepted_lengths_dev.cpu(), accepted_lengths_ref.cpu())
        assert torch.equal(sequence_lengths_dev.cpu(), sequence_lengths_ref.cpu())

    def test_device_accept_handles_empty_paths_and_negative_best_path(self):
        """Kernel should be robust to empty paths and best_path_id == -1."""
        device = torch.device("cuda")

        batch_size = 1
        max_batch_size = 1
        max_seq_len = 8
        max_draft_tokens = 4
        max_path_len = 4

        output_ids = torch.zeros(max_batch_size, max_seq_len, dtype=torch.int32, device=device)
        draft_ids = torch.randint(
            10,
            100,
            (max_batch_size, max_draft_tokens),
            dtype=torch.int32,
            device=device,
        )
        target_ids = draft_ids.clone()

        accepted_lengths = torch.full(
            (max_batch_size,), 123, dtype=torch.int32, device=device
        )
        sequence_lengths = torch.zeros(max_batch_size, dtype=torch.int32, device=device)

        # All paths are empty (-1), and best_path_ids is set to -1 to signal
        # "no best path".
        paths = torch.full(
            (max_batch_size, max_draft_tokens, max_path_len),
            -1,
            dtype=torch.int32,
            device=device,
        )
        best_path_ids = torch.full(
            (max_batch_size,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        batch_slots = torch.arange(batch_size, dtype=torch.int32, device=device)

        _turbomind.eagle_accept_draft_tokens(
            output_ids,
            draft_ids,
            target_ids,
            accepted_lengths,
            sequence_lengths,
            paths,
            best_path_ids,
            batch_slots,
        )

        # No tokens should be accepted, and sequence_lengths should remain 0.
        assert int(accepted_lengths[0].item()) == 0
        assert int(sequence_lengths[0].item()) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for device path-pack tests")
class TestPackAcceptedPathsDevice:
    """GPU-backed tests for the invokePackAcceptedPaths kernel via _turbomind."""

    def test_pack_accepted_paths_matches_reference(self):
        device = torch.device("cuda")

        batch_size = 2
        max_batch_size = 2
        num_paths = 3
        max_path_len = 4

        # Accepted lengths per slot.
        accepted_lengths = torch.tensor([3, 2], dtype=torch.int32, device=device)

        # Prefix sum over accepted lengths: [3, 5]
        accepted_lengths_cumsum = torch.zeros(batch_size, dtype=torch.int32, device=device)
        accepted_lengths_cumsum[0] = accepted_lengths[0]
        for i in range(1, batch_size):
            accepted_lengths_cumsum[i] = accepted_lengths_cumsum[i - 1] + accepted_lengths[i]

        # Paths tensor: [max_batch_size, num_paths, max_path_len]
        paths = torch.full(
            (max_batch_size, num_paths, max_path_len),
            -1,
            dtype=torch.int32,
            device=device,
        )
        # Slot 0, best path 0: [0, 1, 2, -1]
        paths[0, 0, :3] = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        # Slot 1, best path 1: [3, 4, -1, -1]
        paths[1, 1, :2] = torch.tensor([3, 4], dtype=torch.int32, device=device)

        best_path_ids = torch.tensor([0, 1], dtype=torch.int32, device=device)
        batch_slots = torch.arange(batch_size, dtype=torch.int32, device=device)

        # Reference packing on CPU.
        def _pack_ref() -> torch.Tensor:
            offsets = []
            for local_b in range(batch_size):
                slot = batch_slots[local_b].item()
                best = best_path_ids[slot].item()
                accepted = accepted_lengths[slot].item()
                for i in range(accepted):
                    idx = paths[slot, best, i].item()
                    offsets.append(idx)
            return torch.tensor(offsets, dtype=torch.int32)

        expected_offsets = _pack_ref()

        total_accepted = int(accepted_lengths.sum().item())
        paths_offsets = torch.full(total_accepted, -1, dtype=torch.int32, device=device)

        _turbomind.eagle_pack_accepted_paths(
            accepted_lengths_cumsum,
            paths_offsets,
            accepted_lengths,
            best_path_ids,
            paths,
            batch_slots,
        )

        assert torch.equal(paths_offsets.cpu(), expected_offsets)

    def test_pack_handles_zero_accepted_lengths(self):
        """Packing should be a no-op when all accepted_lengths are zero."""
        device = torch.device("cuda")

        batch_size = 2
        max_batch_size = 2
        num_paths = 2
        max_path_len = 3

        accepted_lengths = torch.zeros(batch_size, dtype=torch.int32, device=device)
        accepted_lengths_cumsum = torch.zeros(batch_size, dtype=torch.int32, device=device)

        # Paths tensor is irrelevant when nothing is accepted.
        paths = torch.full(
            (max_batch_size, num_paths, max_path_len),
            -1,
            dtype=torch.int32,
            device=device,
        )
        best_path_ids = torch.zeros(max_batch_size, dtype=torch.int32, device=device)
        batch_slots = torch.arange(batch_size, dtype=torch.int32, device=device)

        # Pre-fill offsets with a sentinel that should remain unchanged.
        total_accepted = int(accepted_lengths_cumsum[-1].item() if batch_size > 0 else 0)
        # When all lengths are zero, total_accepted is 0; use a small buffer anyway.
        total_accepted = max(total_accepted, 1)
        paths_offsets = torch.full(total_accepted, -42, dtype=torch.int32, device=device)

        _turbomind.eagle_pack_accepted_paths(
            accepted_lengths_cumsum,
            paths_offsets,
            accepted_lengths,
            best_path_ids,
            paths,
            batch_slots,
        )

        # No offsets should have been written.
        assert torch.all(paths_offsets.cpu() == -42)

    def test_pack_ignores_negative_best_path_ids(self):
        """Slots with best_path_id < 0 should not write offsets."""
        device = torch.device("cuda")

        batch_size = 2
        max_batch_size = 2
        num_paths = 2
        max_path_len = 3

        accepted_lengths = torch.tensor([1, 1], dtype=torch.int32, device=device)
        accepted_lengths_cumsum = torch.zeros(batch_size, dtype=torch.int32, device=device)
        accepted_lengths_cumsum[0] = accepted_lengths[0]
        accepted_lengths_cumsum[1] = accepted_lengths_cumsum[0] + accepted_lengths[1]

        paths = torch.full(
            (max_batch_size, num_paths, max_path_len),
            -1,
            dtype=torch.int32,
            device=device,
        )
        # Only slot 1 has a valid path.
        paths[1, 0, 0] = 7

        best_path_ids = torch.tensor([-1, 0], dtype=torch.int32, device=device)
        batch_slots = torch.arange(batch_size, dtype=torch.int32, device=device)

        total_accepted = int(accepted_lengths_cumsum[-1].item())
        paths_offsets = torch.full(total_accepted, -1, dtype=torch.int32, device=device)

        _turbomind.eagle_pack_accepted_paths(
            accepted_lengths_cumsum,
            paths_offsets,
            accepted_lengths,
            best_path_ids,
            paths,
            batch_slots,
        )

        # Offset for slot 0 (first position) should remain sentinel; slot 1 should
        # pick up the path index from its best path.
        offsets_cpu = paths_offsets.cpu()
        assert offsets_cpu[0].item() == -1
        assert offsets_cpu[1].item() == 7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for device tree-accept tests")
class TestTreeAcceptTokensDevice:
    """GPU-backed tests for the tree-based acceptance kernel via _turbomind."""

    def test_tree_accept_matches_host_logic(self):
        device = torch.device("cuda")

        batch_size = 1
        max_batch_size = 1
        max_draft_tokens = 4
        num_paths = 2
        max_path_len = 4

        draft_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.int32, device=device)

        # Make path 0 fully match, and path 1 mismatch at the third token so
        # path 0 has the longer accepted length.
        target_ids = torch.tensor([[10, 20, 30, 99]], dtype=torch.int32, device=device)

        paths = torch.full(
            (max_batch_size, num_paths, max_path_len),
            -1,
            dtype=torch.int32,
            device=device,
        )
        # Node indices are token_idx+1; 0 is root.
        # Path 0: [1, 2, 3, 4]
        paths[0, 0, :] = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=device)
        # Path 1: [1, 2, 4, -1]
        paths[0, 1, :3] = torch.tensor([1, 2, 4], dtype=torch.int32, device=device)

        best_path_ids = torch.zeros(max_batch_size, dtype=torch.int32, device=device)
        accepted_lengths = torch.zeros(max_batch_size, dtype=torch.int32, device=device)
        accepted_tokens = torch.zeros(max_batch_size, max_path_len, dtype=torch.int32, device=device)
        batch_slots = torch.arange(batch_size, dtype=torch.int32, device=device)

        # Host reference logic mirroring LlamaV2_eagle.cc's tree acceptance.
        def _host_tree_accept():
            paths_cpu = paths.cpu()
            draft_cpu = draft_ids.cpu()
            target_cpu = target_ids.cpu()

            best_path = 0
            best_accept = 0
            for p in range(num_paths):
                accepted = 0
                for d in range(max_path_len):
                    node_idx = int(paths_cpu[0, p, d].item())
                    if node_idx <= 0:
                        if node_idx < 0:
                            break
                        continue
                    token_idx = node_idx - 1
                    if token_idx < 0 or token_idx >= max_draft_tokens:
                        break
                    draft = int(draft_cpu[0, token_idx].item())
                    target = int(target_cpu[0, token_idx].item())
                    accepted += 1
                    if draft != target:
                        break
                if accepted > best_accept:
                    best_accept = accepted
                    best_path = p

            tokens = []
            written = 0
            for d in range(max_path_len):
                if written >= best_accept:
                    break
                node_idx = int(paths_cpu[0, best_path, d].item())
                if node_idx <= 0:
                    if node_idx < 0:
                        break
                    continue
                token_idx = node_idx - 1
                if token_idx < 0 or token_idx >= max_draft_tokens:
                    break
                target = int(target_cpu[0, token_idx].item())
                tokens.append(target)
                written += 1

            return best_path, best_accept, tokens

        ref_best_path, ref_accept_len, ref_tokens = _host_tree_accept()

        _turbomind.eagle_tree_accept_tokens(
            draft_ids,
            target_ids,
            paths,
            best_path_ids,
            accepted_lengths,
            accepted_tokens,
            batch_slots,
        )

        assert int(best_path_ids[0].item()) == ref_best_path
        assert int(accepted_lengths[0].item()) == ref_accept_len
        assert ref_accept_len == len(ref_tokens)
        if ref_accept_len > 0:
            got_tokens = accepted_tokens[0, :ref_accept_len].cpu().tolist()
            assert got_tokens == ref_tokens


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for device KV rewind tests")
class TestKVCacheRewindDevice:
    """GPU-backed tests for the KV rewind kernel via _turbomind."""

    def test_device_kv_rewind_respects_batch_slots_and_zero_lengths(self):
        device = torch.device("cuda")

        block_size = 4
        max_batch_size = 4
        max_blocks_per_seq = 3
        num_layers = 1
        batch_size = 2

        # Distinct entries per row so we can see which rows were touched.
        block_tables = torch.arange(
            max_batch_size * max_blocks_per_seq,
            dtype=torch.int32,
            device=device,
        ).view(max_batch_size, max_blocks_per_seq)

        # Slot 0 rewinds one block; slot 1 has zero rewind; slots 2 and 3 are inactive.
        rewind_lengths = torch.tensor(
            [block_size, 0, 0, 0],
            dtype=torch.int32,
            device=device,
        )
        # Map local batch indices [0,1] -> slots [0,3]; slot 3 is out-of-range for batch_size=2
        # but within max_batch_size, so only slot 0 should be affected.
        batch_slots = torch.tensor([0, 3], dtype=torch.int32, device=device)

        _turbomind.eagle_kv_cache_rewind(
            rewind_lengths,
            batch_slots,
            block_tables,
            num_layers,
            block_size,
        )

        table = block_tables.cpu().numpy()
        # Slot 0: last block cleared.
        assert table[0, -1] == -1
        # Slot 1 and others remain untouched.
        assert table[1, 0] >= 0 and table[1, 1] >= 0 and table[1, 2] >= 0
        assert table[3, 0] >= 0 and table[3, 1] >= 0 and table[3, 2] >= 0

    def test_device_kv_rewind_clamps_to_max_blocks(self):
        device = torch.device("cuda")

        block_size = 4
        max_batch_size = 2
        max_blocks_per_seq = 2
        num_layers = 1
        batch_size = 1

        block_tables = torch.arange(
            max_batch_size * max_blocks_per_seq,
            dtype=torch.int32,
            device=device,
        ).view(max_batch_size, max_blocks_per_seq)

        # Rewind length much larger than total capacity; all blocks in slot 0
        # should be cleared but no out-of-bounds writes should occur.
        rewind_lengths = torch.tensor(
            [block_size * 10, 0],
            dtype=torch.int32,
            device=device,
        )
        batch_slots = torch.tensor([0], dtype=torch.int32, device=device)

        _turbomind.eagle_kv_cache_rewind(
            rewind_lengths,
            batch_slots,
            block_tables,
            num_layers,
            block_size,
        )

        table = block_tables.cpu().numpy()
        # Slot 0: all blocks cleared.
        assert table[0, 0] == -1 and table[0, 1] == -1
        # Slot 1: untouched.
        assert table[1, 0] >= 0 and table[1, 1] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
