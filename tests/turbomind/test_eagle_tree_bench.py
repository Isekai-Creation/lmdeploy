"""
Microbenchmark tests for SpeculationTree build/extract.

These tests ensure that the eagle_tree_bench helpers run without error
and return sane timing statistics on CPU-only environments.
"""

import pytest


def test_benchmark_speculation_tree_build_reports_sane_stats():
    pytest.importorskip("lmdeploy.turbomind.eagle_tree")

    from lmdeploy.turbomind.eagle_tree_bench import (  # type: ignore
        benchmark_speculation_tree_build,
    )

    result = benchmark_speculation_tree_build(
        max_depth=4,
        max_paths=8,
        num_tokens=16,
        iters=3,
        seed=42,
    )

    assert result["iters"] == pytest.approx(3.0)
    assert result["max_depth"] == pytest.approx(4.0)
    assert result["avg_ms_per_build"] >= 0.0

