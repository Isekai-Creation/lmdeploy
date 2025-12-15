"""
Python bindings for leaf mask kernel.

Enables efficient tree traversal in EAGLE3 speculation.
"""

import torch
import numpy as np
from typing import Optional

# Try to import the compiled CUDA extension
try:
    from lmdeploy.turbomind import leaf_mask_ops

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: Leaf mask CUDA kernel not available. Using CPU fallback.")


def build_leaf_mask(
    next_paths: torch.Tensor,
    batch_size: int,
    max_decoding_tokens: int,
    max_path_len: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Build leaf mask for tree nodes.

    Identifies which nodes are leaves (no children) vs non-leaves (have children).
    Critical for efficient tree traversal in EAGLE3.

    Args:
        next_paths: Tree paths [batch_size, max_decoding_tokens, max_path_len]
        batch_size: Number of sequences in batch
        max_decoding_tokens: Maximum number of tokens per sequence
        max_path_len: Maximum path length in tree
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        Leaf mask [batch_size, max_decoding_tokens]
        1 = leaf node, 0 = non-leaf node

    Example:
        >>> paths = torch.tensor([
        ...     [[0, 1, 2], [0, 1, 3], [0, -1, -1]],  # Batch 0
        ... ], dtype=torch.int32)
        >>> mask = build_leaf_mask(paths, 1, 3, 3)
        >>> # mask[0] = [0, 0, 1, 1]  # Nodes 0,1 are non-leaves, 2,3 are leaves
    """
    if not CUDA_AVAILABLE or device == "cpu":
        return _build_leaf_mask_cpu(
            next_paths, batch_size, max_decoding_tokens, max_path_len
        )

    # Allocate output tensor
    leaf_mask = torch.ones(
        (batch_size, max_decoding_tokens), dtype=torch.int8, device=device
    )

    # Call CUDA kernel
    leaf_mask_ops.build_leaf_mask(
        leaf_mask, next_paths, batch_size, max_decoding_tokens, max_path_len
    )

    return leaf_mask


def get_non_leaf_nodes(
    next_paths: torch.Tensor,
    batch_size: int,
    max_decoding_tokens: int,
    max_path_len: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Get indices of non-leaf nodes.

    Useful for selecting nodes that need further expansion.

    Args:
        next_paths: Tree paths [batch_size, max_decoding_tokens, max_path_len]
        batch_size: Number of sequences in batch
        max_decoding_tokens: Maximum number of tokens per sequence
        max_path_len: Maximum path length in tree
        device: Device to run on

    Returns:
        Non-leaf indices [batch_size, max_non_leaves]
        Padded with -1 for sequences with fewer non-leaves
    """
    # Build leaf mask
    leaf_mask = build_leaf_mask(
        next_paths, batch_size, max_decoding_tokens, max_path_len, device
    )

    # Get non-leaf indices (where mask == 0)
    non_leaf_mask = leaf_mask == 0

    # Convert to indices
    non_leaf_indices = []
    for b in range(batch_size):
        indices = torch.where(non_leaf_mask[b])[0]
        non_leaf_indices.append(indices)

    # Pad to same length
    max_non_leaves = max(len(idx) for idx in non_leaf_indices)
    padded_indices = torch.full(
        (batch_size, max_non_leaves), -1, dtype=torch.int32, device=device
    )

    for b, indices in enumerate(non_leaf_indices):
        if len(indices) > 0:
            padded_indices[b, : len(indices)] = indices

    return padded_indices


# CPU fallback implementation


def _build_leaf_mask_cpu(
    next_paths: torch.Tensor,
    batch_size: int,
    max_decoding_tokens: int,
    max_path_len: int,
) -> torch.Tensor:
    """CPU fallback for leaf mask building."""
    leaf_mask = torch.ones((batch_size, max_decoding_tokens), dtype=torch.int8)

    for b in range(batch_size):
        # Mark non-leaves
        for token_idx in range(max_decoding_tokens):
            for level in range(max_path_len - 1):
                node_at_level = next_paths[b, token_idx, level].item()

                if node_at_level >= 0 and node_at_level < max_decoding_tokens:
                    next_node = next_paths[b, token_idx, level + 1].item()

                    # If next level exists, current node is not a leaf
                    if next_node >= 0 and next_node < max_decoding_tokens:
                        leaf_mask[b, node_at_level] = 0

    return leaf_mask
