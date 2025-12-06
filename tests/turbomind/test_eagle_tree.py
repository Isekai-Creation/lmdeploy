"""
Unit tests for the EAGLE SpeculationTree host-side implementation.

These tests validate basic tree construction, path extraction, and
choices-table behaviour without touching the C++ EagleModule or
LlamaBatch integration owned by Engineer B.
"""

from typing import List
import random

import pytest

from lmdeploy.turbomind import eagle_tree


def _build_linear_tokens(n: int) -> List[int]:
    """Helper to build a simple linear token sequence."""
    return list(range(n))


class TestSpeculationTreeBasic:
    """Basic behaviours of SpeculationTree."""

    def test_linear_build_respects_depth_and_paths(self):
        """Linear buildTree should create a chain within max_depth/max_paths."""
        max_depth = 5
        max_paths = 8
        num_tokens = 4

        tree = eagle_tree.SpeculationTree(max_depth=max_depth, max_paths=max_paths)
        tokens = _build_linear_tokens(num_tokens)

        tree.buildTree(tokens, num_tokens)
        tree.extractPaths()

        # For a simple linear chain, we expect a single path from root to leaf
        num_paths = tree.getNumPaths()
        assert num_paths == 1

        paths_flat = list(tree.getPathsFlat())
        # paths_flat is [path_len padded to max_depth] with node indices
        # First entry should be the root (index 0).
        assert paths_flat[0] == 0

    def test_build_with_choices_uses_table(self):
        """buildTreeWithChoices should respect an explicit choices table."""
        max_depth = 4
        max_paths = 4
        num_tokens = 3

        # Root (0) has two children; first child has one child.
        choices = [
            [1, 2],  # children of root
            [3],     # children of node 1
            [],      # children of node 2
            [],      # children of node 3
        ]

        tree = eagle_tree.SpeculationTree(max_depth=max_depth, max_paths=max_paths)
        tokens = _build_linear_tokens(num_tokens)

        tree.buildTreeWithChoices(tokens, num_tokens, choices)
        tree.extractPaths()

        num_paths = tree.getNumPaths()
        assert num_paths >= 1

        paths_flat = list(tree.getPathsFlat())
        # Ensure root node is present and at least one child exists.
        assert paths_flat[0] == 0

    def test_reset_clears_state(self):
        """reset should clear nodes and paths for the next iteration."""
        tree = eagle_tree.SpeculationTree(max_depth=3, max_paths=2)
        tokens = _build_linear_tokens(2)

        tree.buildTree(tokens, len(tokens))
        tree.extractPaths()
        assert tree.getNumPaths() > 0

        tree.reset()
        tree.extractPaths()
        assert tree.getNumPaths() == 0


class TestSpeculationTreeAdvanced:
    """More detailed behaviours of SpeculationTree."""

    def test_build_tree_with_no_tokens_has_no_paths(self):
        """buildTree should handle an empty draft token sequence."""
        tree = eagle_tree.SpeculationTree(max_depth=4, max_paths=4)
        tokens: List[int] = []

        tree.buildTree(tokens, 0)
        tree.extractPaths()

        assert tree.getNumPaths() == 0
        assert list(tree.getPathsFlat()) == []

    def test_build_with_choices_respects_max_depth_and_max_paths(self):
        """buildTreeWithChoices should obey max_depth * max_paths clipping."""
        max_depth = 2
        max_paths = 2
        num_tokens = 16

        tree = eagle_tree.SpeculationTree(max_depth=max_depth, max_paths=max_paths)
        tokens = _build_linear_tokens(num_tokens)

        # Root wants many children, but max_nodes = max_depth * max_paths = 4
        # so we can only create at most 3 non-root nodes.
        choices: List[List[int]] = [
            list(range(8)),  # many requested children from root
        ]

        tree.buildTreeWithChoices(tokens, num_tokens, choices)
        tree.extractPaths()

        num_paths = tree.getNumPaths()
        # One root node + at most 3 children => at most 3 leaf paths.
        assert num_paths == max_depth * max_paths - 1

        # Flattened paths should always have num_paths * max_depth entries.
        flat = list(tree.getPathsFlat())
        assert len(flat) == num_paths * tree.getMaxDepth()

    def test_build_with_choices_honours_max_depth_stop(self):
        """Children beyond max_depth should not be expanded."""
        max_depth = 2
        max_paths = 4
        num_tokens = 4

        tree = eagle_tree.SpeculationTree(max_depth=max_depth, max_paths=max_paths)
        tokens = _build_linear_tokens(num_tokens)

        # Root has two children; the first would like to spawn one more child,
        # but depth limit prevents that extra level.
        choices: List[List[int]] = [
            [1, 2],  # children of root
            [3],     # would be child of node 1, but blocked by max_depth
            [],      # node 2
            [],      # node 3 (unused)
        ]

        tree.buildTreeWithChoices(tokens, num_tokens, choices)
        tree.extractPaths()

        # Level-0: root should be the only non-leaf node.
        non_leaf_level0 = tree.getNonLeafNodes(0)
        assert non_leaf_level0 == [0]

        # Level-1 nodes should remain leaves because we stopped expanding.
        non_leaf_level1 = tree.getNonLeafNodes(1)
        assert non_leaf_level1 == []

    def test_get_non_leaf_nodes_and_find_best_path(self):
        """getNonLeafNodes and findBestPath should reflect tree structure."""
        max_depth = 4
        max_paths = 4
        num_tokens = 3

        tree = eagle_tree.SpeculationTree(max_depth=max_depth, max_paths=max_paths)
        tokens = _build_linear_tokens(num_tokens)

        # Tree shape (node indices):
        #   0 (root)
        #   ├─ 1
        #   │   └─ 3
        #   └─ 2
        choices: List[List[int]] = [
            [1, 2],  # children of root
            [3],     # child of node 1
            [],      # node 2
            [],      # node 3
        ]

        tree.buildTreeWithChoices(tokens, num_tokens, choices)
        tree.extractPaths()

        # Root is non-leaf, and node 1 is also non-leaf.
        assert tree.getNonLeafNodes(0) == [0]
        assert tree.getNonLeafNodes(1) == [1]

        num_paths = tree.getNumPaths()
        assert num_paths == 2

        # When both paths are accepted, the longer path (index 0) should win.
        accepted = [True] * num_paths
        best_idx = tree.findBestPath(accepted)
        assert best_idx == 0

        # When only the second path is accepted, it should be chosen.
        accepted = [False] * num_paths
        accepted[1] = True
        best_idx = tree.findBestPath(accepted)
        assert best_idx == 1

    def test_random_choices_preserve_invariants(self):
        """Random choices tables should never violate basic invariants.

        This fuzz-style test builds a variety of random choices tables and
        ensures that SpeculationTree construction and path extraction do
        not violate the max_depth * max_paths node bound or produce
        inconsistent flattened path sizes.
        """
        max_depth = 4
        max_paths = 4
        max_nodes = max_depth * max_paths

        for seed in range(5):
            random.seed(seed)
            num_tokens = random.randint(1, 16)
            tokens = _build_linear_tokens(num_tokens)

            tree = eagle_tree.SpeculationTree(max_depth=max_depth, max_paths=max_paths)

            # Build a random choices table where each node may have children
            # pointing to later nodes, but we never exceed max_nodes entries.
            num_nodes = random.randint(1, max_nodes)
            choices: List[List[int]] = []
            for node_idx in range(num_nodes):
                # Each node can have 0-3 children, but child indices must stay in range.
                num_children = random.randint(0, 3)
                child_indices = [
                    random.randint(0, max(0, num_nodes - 1)) for _ in range(num_children)
                ]
                choices.append(child_indices)

            tree.buildTreeWithChoices(tokens, num_tokens, choices)
            tree.extractPaths()

            # Invariants: node count should never exceed max_nodes (asserted
            # in C++), and flattened paths must have num_paths * max_depth entries.
            num_paths = tree.getNumPaths()
            if num_paths == 0:
                assert list(tree.getPathsFlat()) == []
            else:
                flat = list(tree.getPathsFlat())
                assert len(flat) == num_paths * tree.getMaxDepth()

    def test_large_random_choices_preserve_invariants(self):
        """Larger random trees should still respect basic invariants."""
        max_depth = 6
        max_paths = 6
        max_nodes = max_depth * max_paths

        for seed in range(3):
            random.seed(1000 + seed)
            num_tokens = random.randint(16, 48)
            tokens = _build_linear_tokens(num_tokens)

            tree = eagle_tree.SpeculationTree(max_depth=max_depth, max_paths=max_paths)

            num_nodes = random.randint(max_paths, max_nodes)
            choices: List[List[int]] = []
            for node_idx in range(num_nodes):
                num_children = random.randint(0, 4)
                child_indices = [
                    random.randint(0, max(0, num_tokens - 1)) for _ in range(num_children)
                ]
                choices.append(child_indices)

            tree.buildTreeWithChoices(tokens, num_tokens, choices)
            tree.extractPaths()

            num_paths = tree.getNumPaths()
            if num_paths == 0:
                assert list(tree.getPathsFlat()) == []
            else:
                flat = list(tree.getPathsFlat())
                assert len(flat) == num_paths * tree.getMaxDepth()
