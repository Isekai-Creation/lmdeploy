"""
Python bindings for packed mask CUDA kernels.

Provides 32x memory compression for attention masks in EAGLE3 speculation.
"""

import torch
import numpy as np
from typing import Optional

# Try to import the compiled CUDA extension
try:
    from lmdeploy.turbomind import packed_mask_ops

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: Packed mask CUDA kernels not available. Using CPU fallback.")


def get_packed_mask(
    cum_generation_lengths: torch.Tensor,
    batch_size: int,
    max_decoding_tokens: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Generate packed attention masks from cumulative generation lengths.

    Compresses boolean masks into int32 bit fields, achieving 32x memory reduction.

    Args:
        cum_generation_lengths: Cumulative sum of generation lengths [batch_size + 1]
        batch_size: Number of sequences in batch
        max_decoding_tokens: Maximum number of decoding tokens per sequence
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        Packed masks [batch_size, max_decoding_tokens, ceil(max_decoding_tokens/32)]

    Example:
        >>> cum_lengths = torch.tensor([0, 3, 7, 10], device='cuda')
        >>> masks = get_packed_mask(cum_lengths, batch_size=3, max_decoding_tokens=16)
        >>> # masks.shape = (3, 16, 1)  # 16 tokens packed into 1 int32 per position
    """
    if not CUDA_AVAILABLE or device == "cpu":
        return _get_packed_mask_cpu(
            cum_generation_lengths, batch_size, max_decoding_tokens
        )

    # Calculate packed size
    packed_size = (max_decoding_tokens + 31) // 32

    # Allocate output tensor
    packed_mask = torch.zeros(
        (batch_size, max_decoding_tokens, packed_size), dtype=torch.int32, device=device
    )

    # Call CUDA kernel
    packed_mask_ops.get_packed_mask(
        packed_mask, cum_generation_lengths, batch_size, max_decoding_tokens
    )

    return packed_mask


def get_packed_mask_from_path(
    next_draft_paths: torch.Tensor,
    batch_slots: Optional[torch.Tensor],
    batch_size: int,
    max_decoding_tokens: int,
    max_path_len: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Generate packed attention masks from tree paths.

    Args:
        next_draft_paths: Tree paths [batch_size, max_decoding_tokens, max_path_len]
        batch_slots: Batch slot mapping [batch_size], optional
        batch_size: Number of sequences in batch
        max_decoding_tokens: Maximum number of decoding tokens per sequence
        max_path_len: Maximum path length in tree
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        Packed masks [batch_size, max_decoding_tokens, ceil(max_decoding_tokens/32)]

    Example:
        >>> paths = torch.randint(0, 10, (4, 16, 5), device='cuda')
        >>> masks = get_packed_mask_from_path(paths, None, 4, 16, 5)
        >>> # masks.shape = (4, 16, 1)  # 32x compression!
    """
    if not CUDA_AVAILABLE or device == "cpu":
        return _get_packed_mask_from_path_cpu(
            next_draft_paths, batch_slots, batch_size, max_decoding_tokens, max_path_len
        )

    # Calculate packed size
    packed_size = (max_decoding_tokens + 31) // 32

    # Allocate output tensor
    # Note: Output is at batch slots, so use max batch size if slots provided
    max_batch_size = (
        batch_slots.max().item() + 1 if batch_slots is not None else batch_size
    )
    packed_mask = torch.zeros(
        (max_batch_size, max_decoding_tokens, packed_size),
        dtype=torch.int32,
        device=device,
    )

    # Default batch slots if not provided
    if batch_slots is None:
        batch_slots = torch.arange(batch_size, dtype=torch.int32, device=device)

    # Call CUDA kernel
    packed_mask_ops.get_packed_mask_from_path(
        packed_mask,
        batch_slots,
        next_draft_paths,
        batch_size,
        max_decoding_tokens,
        max_path_len,
    )

    return packed_mask


def unpack_mask(packed_mask: torch.Tensor) -> torch.Tensor:
    """
    Unpack int32 bit fields back to boolean mask.

    Useful for debugging and visualization.

    Args:
        packed_mask: Packed masks [batch_size, max_tokens, packed_size]

    Returns:
        Boolean mask [batch_size, max_tokens, max_tokens]
    """
    batch_size, max_tokens, packed_size = packed_mask.shape

    # Allocate boolean mask
    bool_mask = torch.zeros(
        (batch_size, max_tokens, max_tokens),
        dtype=torch.bool,
        device=packed_mask.device,
    )

    # Unpack each int32
    for b in range(batch_size):
        for i in range(max_tokens):
            for p in range(packed_size):
                packed_val = packed_mask[b, i, p].item()
                # Extract 32 bits
                for bit in range(32):
                    pos = p * 32 + bit
                    if pos < max_tokens:
                        bool_mask[b, i, pos] = (packed_val >> bit) & 1

    return bool_mask


# CPU fallback implementations


def _get_packed_mask_cpu(
    cum_generation_lengths: torch.Tensor, batch_size: int, max_decoding_tokens: int
) -> torch.Tensor:
    """CPU fallback for packed mask generation."""
    packed_size = (max_decoding_tokens + 31) // 32
    packed_mask = torch.zeros(
        (batch_size, max_decoding_tokens, packed_size), dtype=torch.int32
    )

    for b in range(batch_size):
        gen_len = cum_generation_lengths[b + 1] - cum_generation_lengths[b]
        for i in range(max_decoding_tokens):
            for p in range(packed_size):
                packed_val = 0
                for bit in range(32):
                    pos = p * 32 + bit
                    if pos < max_decoding_tokens and pos < gen_len and pos <= i:
                        packed_val |= 1 << bit
                packed_mask[b, i, p] = packed_val

    return packed_mask


def _get_packed_mask_from_path_cpu(
    next_draft_paths: torch.Tensor,
    batch_slots: Optional[torch.Tensor],
    batch_size: int,
    max_decoding_tokens: int,
    max_path_len: int,
) -> torch.Tensor:
    """CPU fallback for path-based packed mask generation."""
    packed_size = (max_decoding_tokens + 31) // 32
    max_batch_size = (
        batch_slots.max().item() + 1 if batch_slots is not None else batch_size
    )
    packed_mask = torch.zeros(
        (max_batch_size, max_decoding_tokens, packed_size), dtype=torch.int32
    )

    if batch_slots is None:
        batch_slots = torch.arange(batch_size, dtype=torch.int32)

    for local_b in range(batch_size):
        global_b = batch_slots[local_b].item()
        for i in range(max_decoding_tokens):
            for p in range(packed_size):
                packed_val = 0
                for bit in range(32):
                    pos = p * 32 + bit
                    if pos < max_decoding_tokens:
                        # Check if pos is ancestor of i in tree
                        is_ancestor = pos == i
                        if not is_ancestor and pos < i:
                            # Check path
                            for level in range(max_path_len):
                                if next_draft_paths[global_b, i, level] == pos:
                                    is_ancestor = True
                                    break
                        if is_ancestor:
                            packed_val |= 1 << bit
                packed_mask[global_b, i, p] = packed_val

    return packed_mask
