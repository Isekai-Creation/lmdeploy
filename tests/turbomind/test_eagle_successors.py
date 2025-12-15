"""
Unit tests for successor metadata construction semantics.

These tests mirror the layout produced by invokeExtractSuccessorsFromPaths:

- num_successors[batch, node] counts outgoing edges per node.
- successor_offsets[batch] is a prefix sum over nodes-with-successors.
- successor_counts[successor_offsets[batch] .. successor_offsets[batch+1]) stores
  the count for each node that has at least one successor, in node order.
"""

from typing import List

import numpy as np


def _compute_num_successors_from_paths(
    paths: np.ndarray, max_decoding_tokens: int, max_path_len: int
) -> np.ndarray:
    """
    Pure Python reference for the GPU computeSuccessorCountsKernel.

    Args:
        paths: [batch, max_decoding_tokens, max_path_len] with node indices.
    Returns:
        num_successors: [batch, max_decoding_tokens] counts per node.
    """
    batch_size = paths.shape[0]
    num_successors = np.zeros((batch_size, max_decoding_tokens), dtype=np.int32)
    for b in range(batch_size):
        adj = np.zeros((max_decoding_tokens, max_decoding_tokens), dtype=np.uint8)
        for path_idx in range(max_decoding_tokens):
            path = paths[b, path_idx]
            for level in range(max_path_len - 1):
                from_idx = int(path[level])
                to_idx = int(path[level + 1])
                if (
                    0 <= from_idx < max_decoding_tokens
                    and 0 <= to_idx < max_decoding_tokens
                ):
                    adj[from_idx, to_idx] = 1
        num_successors[b] = adj.sum(axis=1).astype(np.int32)
    return num_successors


def _compact_successors(
    num_successors: np.ndarray, max_decoding_tokens: int
) -> (np.ndarray, np.ndarray):
    """
    Pure Python reference for compactSuccessorsKernel semantics.

    Returns:
        successor_offsets: [batch+1]
        successor_counts: flat list of counts for nodes with successors.
    """
    batch_size = num_successors.shape[0]
    nodes_with_succ_per_batch: List[int] = []
    total_nodes_with_succ = 0
    offsets = []
    counts: List[int] = []

    for b in range(batch_size):
        row = num_successors[b]
        nodes_with_succ = int(np.count_nonzero(row > 0))
        offsets.append(total_nodes_with_succ)
        for node in range(max_decoding_tokens):
            c = int(row[node])
            if c > 0:
                counts.append(c)
        total_nodes_with_succ += nodes_with_succ
        nodes_with_succ_per_batch.append(nodes_with_succ)

    offsets.append(total_nodes_with_succ)
    successor_offsets = np.array(offsets, dtype=np.int32)
    successor_counts = np.array(counts, dtype=np.int32)

    return successor_offsets, successor_counts


def test_successor_layout_simple_linear():
    """
    Simple tree:
        batch=1, tokens=4
        paths: 0->1->2->3

    Expect:
      num_successors = [1,1,1,0]
      successor_offsets = [0, 3]
      successor_counts = [1,1,1]
    """
    batch_size = 1
    max_decoding_tokens = 4
    max_path_len = 4

    paths = -np.ones((batch_size, max_decoding_tokens, max_path_len), dtype=np.int32)
    # Single path: [0,1,2,3]
    paths[0, 0, :] = np.array([0, 1, 2, 3], dtype=np.int32)

    num_succ = _compute_num_successors_from_paths(
        paths, max_decoding_tokens, max_path_len
    )
    assert num_succ.shape == (1, 4)
    assert num_succ[0].tolist() == [1, 1, 1, 0]

    succ_offsets, succ_counts = _compact_successors(
        num_succ, max_decoding_tokens
    )
    # Three nodes have successors; offsets should be [0,3]
    assert succ_offsets.tolist() == [0, 3]
    assert succ_counts.tolist() == [1, 1, 1]


def test_successor_layout_branching_tree():
    """
    Tree with branching:
        root 0 -> 1,2
        1 -> 3
        2 has no children

    Flattened paths (max_decoding_tokens>=4):
        path0: 0->1->3
        path1: 0->2
    Expect:
      num_successors = [2,1,0,0]
      nodes_with_succ=2 => successor_offsets=[0,2]
      successor_counts = [2,1]
    """
    batch_size = 1
    max_decoding_tokens = 4
    max_path_len = 3

    paths = -np.ones((batch_size, max_decoding_tokens, max_path_len), dtype=np.int32)
    # path0: 0->1->3
    paths[0, 0, :] = np.array([0, 1, 3], dtype=np.int32)
    # path1: 0->2
    paths[0, 1, :] = np.array([0, 2, -1], dtype=np.int32)

    num_succ = _compute_num_successors_from_paths(
        paths, max_decoding_tokens, max_path_len
    )
    # root has two children (1,2), node 1 has one child (3), others have none.
    assert num_succ[0].tolist() == [2, 1, 0, 0]

    succ_offsets, succ_counts = _compact_successors(
        num_succ, max_decoding_tokens
    )
    # Two nodes with successors => offsets [0,2], counts [2,1]
    assert succ_offsets.tolist() == [0, 2]
    assert succ_counts.tolist() == [2, 1]

