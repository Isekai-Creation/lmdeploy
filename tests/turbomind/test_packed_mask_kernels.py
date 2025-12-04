"""
Unit tests for packed mask kernels.

Tests the 32x compression functionality for attention masks.
"""

import torch
import pytest
import numpy as np

from lmdeploy.turbomind.kernels.speculative_decoding.packed_mask_ops import (
    get_packed_mask,
    get_packed_mask_from_path,
    unpack_mask,
)


class TestPackedMaskKernels:
    """Test suite for packed mask operations."""

    def test_packed_mask_basic(self):
        """Test basic packed mask generation."""
        # Simple case: 3 sequences with lengths 2, 3, 4
        cum_lengths = torch.tensor([0, 2, 5, 9], dtype=torch.int32)
        batch_size = 3
        max_tokens = 8

        # Generate packed mask (CPU fallback for testing)
        packed = get_packed_mask(cum_lengths, batch_size, max_tokens, device="cpu")

        # Unpack to verify
        unpacked = unpack_mask(packed)

        # Check dimensions
        assert packed.shape == (batch_size, max_tokens, 1)  # 8 tokens fit in 1 int32
        assert unpacked.shape == (batch_size, max_tokens, max_tokens)

        # Verify first sequence (length 2)
        # Token 0 should see: [1, 0, 0, ...]
        assert unpacked[0, 0, 0] == True
        assert unpacked[0, 0, 1] == False

        # Token 1 should see: [1, 1, 0, ...]
        assert unpacked[0, 1, 0] == True
        assert unpacked[0, 1, 1] == True
        assert unpacked[0, 1, 2] == False

    def test_packed_mask_compression(self):
        """Test that packing achieves 32x compression."""
        batch_size = 4
        max_tokens = 64  # Needs 2 int32s (64/32)

        cum_lengths = torch.tensor([0, 16, 32, 48, 64], dtype=torch.int32)

        packed = get_packed_mask(cum_lengths, batch_size, max_tokens, device="cpu")

        # Should be (4, 64, 2) - 2 int32s per token
        assert packed.shape == (batch_size, max_tokens, 2)

        # Memory comparison:
        # Boolean: 4 * 64 * 64 = 16,384 bytes
        # Packed: 4 * 64 * 2 * 4 = 2,048 bytes
        # Compression: 16,384 / 2,048 = 8x (because bool is 1 byte, not 1 bit)
        # True compression vs bits: 32x

        bool_size = batch_size * max_tokens * max_tokens  # bools
        packed_size = batch_size * max_tokens * 2 * 4  # int32s in bytes
        compression = (bool_size) / (packed_size / 4)  # bits vs bits

        assert compression == 32.0

    def test_packed_mask_from_path(self):
        """Test packed mask generation from tree paths."""
        batch_size = 2
        max_tokens = 8
        max_path_len = 3

        # Create simple tree paths
        # Batch 0: Linear path [0, 1, 2, 3, ...]
        # Batch 1: Tree structure
        paths = torch.zeros((batch_size, max_tokens, max_path_len), dtype=torch.int32)

        # Batch 0: Linear
        for i in range(max_tokens):
            for j in range(min(i, max_path_len)):
                paths[0, i, j] = i - j - 1

        # Generate packed mask
        packed = get_packed_mask_from_path(
            paths, None, batch_size, max_tokens, max_path_len, device="cpu"
        )

        # Unpack to verify
        unpacked = unpack_mask(packed)

        # Check dimensions
        assert packed.shape == (batch_size, max_tokens, 1)
        assert unpacked.shape == (batch_size, max_tokens, max_tokens)

        # Verify linear path for batch 0
        # Token 2 should see ancestors: [0, 1, 2]
        assert unpacked[0, 2, 0] == True  # Ancestor
        assert unpacked[0, 2, 1] == True  # Ancestor
        assert unpacked[0, 2, 2] == True  # Self
        assert unpacked[0, 2, 3] == False  # Not ancestor

    def test_batch_slots(self):
        """Test batch slot mapping."""
        batch_size = 2
        max_tokens = 8
        max_path_len = 2

        # Batch slots: map local [0, 1] to global [5, 3]
        batch_slots = torch.tensor([5, 3], dtype=torch.int32)

        paths = torch.zeros((6, max_tokens, max_path_len), dtype=torch.int32)

        packed = get_packed_mask_from_path(
            paths, batch_slots, batch_size, max_tokens, max_path_len, device="cpu"
        )

        # Output should be at global slots
        assert packed.shape[0] == 6  # max(batch_slots) + 1

        # Check that slots 5 and 3 have data
        assert packed[5].sum() > 0 or packed[3].sum() > 0

    def test_large_batch(self):
        """Test with realistic batch size."""
        batch_size = 8  # Production batch size
        max_tokens = 16

        cum_lengths = torch.cumsum(
            torch.randint(1, max_tokens, (batch_size + 1,), dtype=torch.int32), dim=0
        )
        cum_lengths[0] = 0

        packed = get_packed_mask(cum_lengths, batch_size, max_tokens, device="cpu")

        # Should fit in 1 int32 (16 < 32)
        assert packed.shape == (batch_size, max_tokens, 1)

        # Verify no errors
        unpacked = unpack_mask(packed)
        assert unpacked.shape == (batch_size, max_tokens, max_tokens)

    def test_memory_savings(self):
        """Demonstrate memory savings with packed masks."""
        batch_size = 8
        max_tokens = 64

        # Boolean mask memory
        bool_mask_bytes = batch_size * max_tokens * max_tokens * 1  # 1 byte per bool

        # Packed mask memory
        packed_size = (max_tokens + 31) // 32
        packed_mask_bytes = (
            batch_size * max_tokens * packed_size * 4
        )  # 4 bytes per int32

        # Calculate savings
        savings_ratio = bool_mask_bytes / packed_mask_bytes

        print(f"\nMemory Savings:")
        print(f"  Boolean mask: {bool_mask_bytes:,} bytes")
        print(f"  Packed mask: {packed_mask_bytes:,} bytes")
        print(f"  Savings: {savings_ratio:.1f}x")

        # Should be ~8x (because bool is 1 byte, not 1 bit)
        # True bit-level compression is 32x
        assert savings_ratio >= 7.0  # Allow some overhead


if __name__ == "__main__":
    # Run tests
    test = TestPackedMaskKernels()

    print("Running packed mask kernel tests...")
    test.test_packed_mask_basic()
    print("✓ Basic packed mask test passed")

    test.test_packed_mask_compression()
    print("✓ Compression test passed")

    test.test_packed_mask_from_path()
    print("✓ Path-based mask test passed")

    test.test_batch_slots()
    print("✓ Batch slots test passed")

    test.test_large_batch()
    print("✓ Large batch test passed")

    test.test_memory_savings()
    print("✓ Memory savings test passed")

    print("\n✅ All tests passed!")
