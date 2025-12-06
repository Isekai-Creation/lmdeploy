"""
Microbenchmark-style tests for speculative decoding helpers.

These tests exercise the Python-level KV rewind benchmark to ensure it
produces sane timing statistics on both CPU and CUDA devices (when
available), without depending on the full TurboMind engine.
"""

import pytest


def test_benchmark_kv_cache_rewind_reports_sane_stats():
    """
    The KV cache rewind benchmark should run without error and return
    non-negative timing numbers.
    """
    pytest.importorskip("torch")
    import torch

    from lmdeploy.turbomind.kernels.speculative_decoding import common as sd_common

    # Run a tiny benchmark to keep CI lightweight.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = sd_common.benchmark_kv_cache_rewind(
        batch_size=2,
        max_batch_size=4,
        max_blocks_per_seq=8,
        num_layers=2,
        block_size=4,
        iters=3,
        device=device,
    )

    assert result["iters"] == pytest.approx(3.0)
    assert result["batch_size"] == pytest.approx(2.0)
    assert result["avg_ms_per_call"] >= 0.0
    assert result["rewinds_per_second"] >= 0.0


def test_benchmark_accept_draft_tokens_reports_sane_stats():
    """Acceptance-kernel benchmark should return sane timing statistics."""
    pytest.importorskip("torch")
    import torch

    from lmdeploy.turbomind.kernels.speculative_decoding import common as sd_common

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = sd_common.benchmark_accept_draft_tokens(
        batch_size=2,
        max_seq_len=64,
        max_draft_tokens=4,
        max_path_len=4,
        iters=3,
        device=device,
    )

    assert result["iters"] == pytest.approx(3.0)
    assert result["batch_size"] == pytest.approx(2.0)
    assert result["avg_ms_per_call"] >= 0.0
    assert result["accept_ops_per_second"] >= 0.0


def test_benchmark_pack_accepted_paths_reports_sane_stats():
    """Pack-kernel benchmark should return sane timing statistics."""
    pytest.importorskip("torch")
    import torch

    from lmdeploy.turbomind.kernels.speculative_decoding import common as sd_common

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = sd_common.benchmark_pack_accepted_paths(
        batch_size=2,
        num_paths=4,
        max_path_len=4,
        iters=3,
        device=device,
    )

    assert result["iters"] == pytest.approx(3.0)
    assert result["batch_size"] == pytest.approx(2.0)
    assert result["avg_ms_per_call"] >= 0.0
    assert result["pack_ops_per_second"] >= 0.0
