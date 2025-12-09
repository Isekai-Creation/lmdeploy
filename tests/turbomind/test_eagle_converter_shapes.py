import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import save_file

# Allow running without `pip install -e .` by adding repo root to sys.path.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lmdeploy.turbomind import eagle_draft_converter as conv


def _build_eagle3_stub(draft_dir: Path,
                       base_dir: Path,
                       *,
                       draft_hidden: int = 6,
                       base_hidden: int = 8,
                       kv_size: int = 2,
                       inter_size: int = 10,
                       vocab_size: int = 32) -> None:
    """Create tiny Eagle3-compatible HF + base checkpoints for shape tests."""
    qkv_in_dim = 2 * draft_hidden
    q_size = base_hidden

    draft_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Minimal HF config for AutoConfig.from_pretrained
    cfg = {
        "model_type": "gpt_oss",
        "hidden_size": draft_hidden,
        "vocab_size": vocab_size,
        "num_attention_heads": 2,
        "intermediate_size": inter_size,
    }
    (draft_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    # Deterministic weights in BF16
    def arange(shape):
        return torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape).to(torch.bfloat16)

    tensors = {
        "norm.weight": arange((draft_hidden,)),
        "midlayer.input_layernorm.weight": arange((draft_hidden,)),
        "midlayer.hidden_norm.weight": arange((draft_hidden,)),
        "midlayer.post_attention_layernorm.weight": arange((draft_hidden,)),
        "midlayer.mlp.gate_proj.weight": arange((inter_size * 2, draft_hidden)),
        "midlayer.mlp.up_proj.weight": arange((inter_size * 2, draft_hidden)),
        # Shape that gets transposed to [inter_size, draft_hidden]
        "midlayer.mlp.down_proj.weight": arange((draft_hidden, inter_size)),
        "fc.weight": arange((draft_hidden, draft_hidden * 3)),
        "midlayer.self_attn.q_proj.weight": arange((q_size, qkv_in_dim)),
        "midlayer.self_attn.k_proj.weight": arange((kv_size, qkv_in_dim)),
        "midlayer.self_attn.v_proj.weight": arange((kv_size, qkv_in_dim)),
        "midlayer.self_attn.o_proj.weight": arange((qkv_in_dim, q_size)),
    }
    save_file(tensors, draft_dir / "model.safetensors")

    base_tensors = {
        "model.layers.0.self_attn.o_proj.weight": arange((base_hidden, base_hidden)),
        # LM head in HF layout [vocab, hidden]; converter should transpose to [hidden, vocab].
        "lm_head.weight": arange((vocab_size, base_hidden)),
    }
    save_file(base_tensors, base_dir / "model.safetensors")


def test_eagle3_converter_exports_expected_geometry(tmp_path):
    draft_dir = tmp_path / "hf"
    base_dir = tmp_path / "base"
    out_dir = tmp_path / "out"

    draft_hidden = 6
    base_hidden = 8
    kv_size = 2
    inter_size = 10
    vocab_size = 32
    qkv_in_dim = 2 * draft_hidden
    qkv_out_dim = base_hidden + 2 * kv_size

    _build_eagle3_stub(draft_dir, base_dir,
                       draft_hidden=draft_hidden,
                       base_hidden=base_hidden,
                       kv_size=kv_size,
                       inter_size=inter_size,
                       vocab_size=vocab_size)

    conv.prepare_eagle_draft_from_hf(str(draft_dir), str(out_dir), base_model_dir=str(base_dir))

    cfg_path = out_dir / "config.yaml"
    assert cfg_path.exists()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["model_config"]

    assert cfg["eagle_mode"] == "eagle3"
    assert cfg["eagle_draft_hidden"] == draft_hidden
    assert cfg["eagle_base_hidden"] == base_hidden
    assert cfg["eagle_qkv_in_dim"] == qkv_in_dim
    assert cfg["eagle_qkv_in_factor"] == 2
    assert cfg["eagle_fc_in_dim"] == draft_hidden * 3
    assert cfg["eagle_fc_in_factor"] == 3
    assert cfg["eagle_capture_layers"] == [1, 17, 32]
    assert cfg["eagle_num_capture_layers"] == 3
    assert cfg["eagle_weight_dtype"] == "bf16"

    def expect_size(path: Path, num_elems: int):
        arr = np.fromfile(path, dtype=np.int16)
        assert arr.size == num_elems
        # Deterministic checksum guards silent layout drift.
        return int(arr.sum())

    # Fused QKV and Wo must mirror Eagle3 geometry.
    qkv_path = out_dir / "layers.0.attention.w_qkv.weight"
    wo_path = out_dir / "layers.0.attention.wo.weight"
    assert qkv_path.exists()
    assert wo_path.exists()

    qkv_checksum = expect_size(qkv_path, qkv_in_dim * qkv_out_dim)
    wo_checksum = expect_size(wo_path, base_hidden * base_hidden)
    assert qkv_checksum != 0
    assert wo_checksum != 0

    # LM head should match config hidden_units x vocab_size.
    lm_head_path = out_dir / "output.weight"
    assert lm_head_path.exists()
    expect_size(lm_head_path, cfg["hidden_units"] * vocab_size)

    # Native Eagle3 projections.
    for name, shape in [
        ("eagle3.q_proj.weight", (base_hidden, qkv_in_dim)),
        ("eagle3.k_proj.weight", (kv_size, qkv_in_dim)),
        ("eagle3.v_proj.weight", (kv_size, qkv_in_dim)),
        ("eagle3.o_proj.weight", (qkv_in_dim, base_hidden)),
    ]:
        path = out_dir / name
        assert path.exists()
        expect_size(path, np.prod(shape))

    # Norms live in draft-hidden space.
    for name in ["layers.0.attention_norm.weight", "layers.0.hidden_norm.weight", "layers.0.ffn_norm.weight"]:
        path = out_dir / name
        assert path.exists()
        expect_size(path, draft_hidden)
