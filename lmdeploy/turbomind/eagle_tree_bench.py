"""
Microbenchmarks for SpeculationTree build/extract operations.

These helpers exercise the C++ SpeculationTree binding via the Python
`lmdeploy.turbomind.eagle_tree` module and report simple timing stats.
They are CPU-only and do not depend on the TurboMind engine.
"""

from __future__ import annotations

import random
import time
from typing import Dict

from lmdeploy.turbomind import eagle_tree


def benchmark_speculation_tree_build(
    max_depth: int = 4,
    max_paths: int = 8,
    num_tokens: int = 32,
    iters: int = 50,
    seed: int | None = 123,
) -> Dict[str, float]:
    """Microbenchmark SpeculationTree.buildTreeWithChoices + extractPaths.

    Args:
        max_depth: Maximum depth passed to SpeculationTree.
        max_paths: Maximum number of paths passed to SpeculationTree.
        num_tokens: Number of draft tokens to simulate.
        iters: Number of benchmark iterations.
        seed: Optional RNG seed for reproducible choices tables.

    Returns:
        A dict with basic timing statistics (mean ms per build).
    """
    if iters <= 0:
        raise ValueError("iters must be positive")

    rng = random.Random(seed)

    tree = eagle_tree.SpeculationTree(max_depth=max_depth, max_paths=max_paths)

    tokens = list(range(num_tokens))

    def _make_choices():
        choices = []
        for node_idx in range(max_depth * max_paths):
            # Each node randomly branches to up to 3 children with indices
            # below num_tokens; this is intentionally loose but bounded.
            num_children = rng.randint(0, 3)
            child_indices = [rng.randint(0, max(0, num_tokens - 1)) for _ in range(num_children)]
            choices.append(child_indices)
        return choices

    # Warmup.
    choices = _make_choices()
    tree.buildTreeWithChoices(tokens, num_tokens, choices)
    tree.extractPaths()

    start = time.perf_counter()
    for _ in range(iters):
        choices = _make_choices()
        tree.buildTreeWithChoices(tokens, num_tokens, choices)
        tree.extractPaths()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed * 1000.0) / float(iters)

    return {
        "max_depth": float(max_depth),
        "max_paths": float(max_paths),
        "num_tokens": float(num_tokens),
        "iters": float(iters),
        "avg_ms_per_build": avg_ms,
    }

