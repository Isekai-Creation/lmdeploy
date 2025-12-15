"""
Unit tests for EagleModule forward pass on synthetic inputs.

These tests exercise the shallow attention+FC+LM-head path with
minimal shape checks and allocation behaviour, without touching
Engineer B's LlamaBatch/LlamaV2_eagle integration.
"""

import numpy as np
import pytest


def _write_half(path, shape):
    """Write a FP16 tensor with the given shape to path."""
    arr = np.random.randn(*shape).astype(np.float16)
    arr.tofile(path)


def _make_synthetic_eagle_draft(tmp_path):
    """Create a minimal synthetic Eagle draft model directory."""
    model_dir = tmp_path / "eagle_draft"
    model_dir.mkdir()

    hidden_units = 16
    vocab_size = 32
    inter_size = 16

    # Minimal config.yaml used by EagleModule::load.
    (model_dir / "config.yaml").write_text(
        "model_config:\n"
        f"  hidden_units: {hidden_units}\n"
        f"  vocab_size: {vocab_size}\n"
        "  head_num: 1\n"
        f"  size_per_head: {hidden_units}\n"
        f"  inter_size: {inter_size}\n"
    )

    # Draft weights – shapes must match EagleModule::load expectations.
    _write_half(model_dir / "tok_embeddings.weight", (vocab_size, hidden_units))
    _write_half(model_dir / "fc.weight", (hidden_units * 2, hidden_units))

    _write_half(model_dir / "layers.0.attention_norm.weight", (hidden_units,))
    _write_half(model_dir / "layers.0.hidden_norm.weight", (hidden_units,))
    _write_half(model_dir / "layers.0.attention.w_qkv.weight",
                (hidden_units, hidden_units * 3))
    _write_half(model_dir / "layers.0.attention.wo.weight",
                (hidden_units, hidden_units))
    _write_half(model_dir / "layers.0.ffn_norm.weight", (hidden_units,))

    _write_half(model_dir / "layers.0.feed_forward.w1.weight",
                (inter_size, hidden_units))
    _write_half(model_dir / "layers.0.feed_forward.w3.weight",
                (inter_size, hidden_units))
    _write_half(model_dir / "layers.0.feed_forward.w2.weight",
                (inter_size, hidden_units))

    _write_half(model_dir / "norm.weight", (hidden_units,))
    _write_half(model_dir / "output.weight", (hidden_units, vocab_size))

    # Mapping file: keep empty so EagleModule does not attempt to copy
    # non-existent data into an uninitialized tensor.
    (model_dir / "draft_id_to_target_id.weight").write_bytes(b"")

    return model_dir, hidden_units, vocab_size


def test_eagle_module_forward_shapes_and_buffer_reuse(tmp_path):
    """Smoke test: forward produces correct shapes and reuses scratch buffers."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for EagleModule forward smoke test")

    from lmdeploy.turbomind.turbomind import _turbomind

    model_dir, hidden_units, vocab_size = _make_synthetic_eagle_draft(tmp_path)

    # Run the C++ helper which constructs EagleModule, runs forward twice,
    # and reports shapes plus whether scratch buffers were reused.
    result = _turbomind.eagle_forward_smoke(str(model_dir), batch_size=2)

    assert result["logits_shape"] == [2, vocab_size]
    assert result["hidden_shape"] == [2, hidden_units]

    # The helper also reports the model-configured dims for additional safety.
    assert result["vocab_size"] == vocab_size
    assert result["hidden_units"] == hidden_units

    # Critically, repeated forwards with the same batch size should reuse
    # the underlying logits and hidden-state buffers instead of reallocating.
    assert result["reuse_logits_buffer"] is True
    assert result["reuse_hidden_buffer"] is True


def test_eagle_module_forward_microbenchmark(tmp_path):
    """Microbenchmark helper should return sane latency statistics."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for EagleModule forward microbenchmark")

    from lmdeploy.turbomind.turbomind import _turbomind

    model_dir, hidden_units, vocab_size = _make_synthetic_eagle_draft(tmp_path)

    # Run a tiny benchmark (few iters) to keep the test lightweight.
    result = _turbomind.eagle_forward_bench(str(model_dir), batch_size=2, iters=3)

    assert result["batch_size"] == 2
    assert result["iters"] == 3
    assert result["hidden_units"] == hidden_units
    assert result["vocab_size"] == vocab_size

    # Average latency should be non-negative, tokens/sec should be finite.
    assert result["avg_ms_per_forward"] >= 0.0
    assert result["tokens_per_second"] >= 0.0
