import os
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

from lmdeploy.turbomind.eagle_draft_converter import prepare_eagle_draft_from_hf


def _make_eagle3_hf_fixture(root: Path) -> tuple[Path, Path]:
    """Create a minimal synthetic Eagle-3 style HF checkpoint and base model."""
    hf_dir = root / "hf_eagle3"
    base_dir = root / "base_model"
    hf_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Small but consistent geometry for testing.
    hidden_size = 4
    vocab_size = 16
    num_heads = 2
    inter_size = 8

    # Eagle-3-style midlayer geometry with 2H QKV layout:
    # q_proj: [q_out, 2 * hidden_size]
    # k/v_proj: [kv_out, 2 * hidden_size]
    # o_proj: [hidden_size, q_out]
    eagle_q_size = 8
    eagle_kv_size = 4
    eagle_qkv_in_dim = 2 * hidden_size

    # HF-style config.json for the draft model.
    cfg = {
        "model_type": "gpt_oss",
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "num_attention_heads": num_heads,
        "intermediate_size": inter_size,
    }
    # Write a minimal HF-style config.json without depending on torch internals.
    import json

    (hf_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    # Draft safetensors with Eagle-3 midlayer weights.
    draft_tensors = {
        "norm.weight": torch.ones(hidden_size, dtype=torch.bfloat16),
        "fc.weight": torch.zeros(hidden_size, hidden_size * 3, dtype=torch.bfloat16),
        "midlayer.self_attn.q_proj.weight": torch.zeros(
            eagle_q_size, eagle_qkv_in_dim, dtype=torch.bfloat16
        ),
        "midlayer.self_attn.k_proj.weight": torch.zeros(
            eagle_kv_size, eagle_qkv_in_dim, dtype=torch.bfloat16
        ),
        "midlayer.self_attn.v_proj.weight": torch.zeros(
            eagle_kv_size, eagle_qkv_in_dim, dtype=torch.bfloat16
        ),
        "midlayer.self_attn.o_proj.weight": torch.zeros(
            hidden_size, eagle_q_size, dtype=torch.bfloat16
        ),
        "midlayer.mlp.gate_proj.weight": torch.zeros(inter_size, hidden_size, dtype=torch.bfloat16),
        "midlayer.mlp.up_proj.weight": torch.zeros(inter_size, hidden_size, dtype=torch.bfloat16),
        "midlayer.mlp.down_proj.weight": torch.zeros(hidden_size, inter_size, dtype=torch.bfloat16),
    }
    save_file(draft_tensors, hf_dir / "model.safetensors")

    # Base model config and weights for Option-1 LLaMA-style attention and LM head.
    base_cfg = {
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "num_attention_heads": num_heads,
        "intermediate_size": inter_size,
    }
    (base_dir / "config.json").write_text(json.dumps(base_cfg), encoding="utf-8")

    attn_hidden = hidden_size
    base_tensors = {
        "model.embed_tokens.weight": torch.zeros(vocab_size, hidden_size, dtype=torch.float16),
        "lm_head.weight": torch.zeros(vocab_size, hidden_size, dtype=torch.float16),
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(
            attn_hidden, attn_hidden, dtype=torch.float16
        ),
        "model.layers.0.self_attn.k_proj.weight": torch.zeros(
            attn_hidden, attn_hidden, dtype=torch.float16
        ),
        "model.layers.0.self_attn.v_proj.weight": torch.zeros(
            attn_hidden, attn_hidden, dtype=torch.float16
        ),
        "model.layers.0.self_attn.o_proj.weight": torch.zeros(
            attn_hidden, attn_hidden, dtype=torch.float16
        ),
        "model.layers.0.mlp.gate_proj.weight": torch.zeros(inter_size, hidden_size, dtype=torch.float16),
        "model.layers.0.mlp.up_proj.weight": torch.zeros(inter_size, hidden_size, dtype=torch.float16),
        "model.layers.0.mlp.down_proj.weight": torch.zeros(hidden_size, inter_size, dtype=torch.float16),
    }
    save_file(base_tensors, base_dir / "model.safetensors")

    return hf_dir, base_dir


def test_eagle3_converter_exports_native_qkv_with_expected_shapes(tmp_path):
    """prepare_eagle_draft_from_hf should export Eagle-3 QKV/O with the geometry
    inferred from the draft checkpoint and record it in config.yaml."""
    hf_dir, base_dir = _make_eagle3_hf_fixture(tmp_path)
    out_dir = tmp_path / "tm_eagle3"

    prepare_eagle_draft_from_hf(str(hf_dir), str(out_dir), base_model_dir=str(base_dir))

    # Check config.yaml geometry.
    cfg_path = out_dir / "config.yaml"
    assert cfg_path.is_file()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mc = cfg["model_config"]

    assert mc["eagle_q_size"] == 8
    assert mc["eagle_kv_size"] == 4
    draft_hidden = mc["eagle_draft_hidden"]
    expected_qkv_in_dim = 2 * draft_hidden
    assert mc["eagle_qkv_in_dim"] == expected_qkv_in_dim

    # Each Eagle-3 attention weight file should exist and have the right size
    # in elements (bf16 written as raw int16).
    def _check_weight(path: Path, rows: int, cols: int):
        assert path.is_file()
        expected_elems = rows * cols
        bytes_on_disk = os.path.getsize(path)
        # bf16 written as int16 => 2 bytes per element.
        assert bytes_on_disk == expected_elems * 2

    _check_weight(out_dir / "eagle3.q_proj.weight", 8, expected_qkv_in_dim)
    _check_weight(out_dir / "eagle3.k_proj.weight", 4, expected_qkv_in_dim)
    _check_weight(out_dir / "eagle3.v_proj.weight", 4, expected_qkv_in_dim)
    _check_weight(out_dir / "eagle3.o_proj.weight", draft_hidden, 8)
