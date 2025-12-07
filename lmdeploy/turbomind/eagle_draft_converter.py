"""
Converters for EAGLE draft models into the minimal EagleNet layout
expected by TurboMind's :class:`EagleModule`.

This module currently provides a focused converter for NVIDIA's
GPT-OSS Eagle3 draft checkpoint (``nvidia/gpt-oss-120b-Eagle3``),
whose architecture is ``LlamaForCausalLMEagle3``.  It turns a local
HF directory containing ``config.json`` and ``model.safetensors`` into
an ``EagleModule``-compatible directory with:

- ``config.yaml``::

      model_config:
        hidden_units: <hidden_size>
        vocab_size:   <vocab_size>
        head_num:     <num_attention_heads>
        size_per_head: <head_dim>
        inter_size:   <intermediate_size>

- Weight files (FP16 raw binaries):

  - Always:

    - ``norm.weight``         ← ``model.norm.weight``
    - ``output.weight``       ← ``lm_head.weight.T`` (shape [hidden, vocab])

  - Optional (used when present, otherwise :class:`EagleModule` falls
    back to a simple RMSNorm+LM head path):

    - ``tok_embeddings.weight``              ← ``model.embed_tokens.weight``
    - ``layers.0.attention_norm.weight``     ←
      ``model.layers.0.input_layernorm.weight``
    - ``layers.0.ffn_norm.weight``           ←
      ``model.layers.0.post_attention_layernorm.weight``
    - ``layers.0.hidden_norm.weight``        ← same as ``ffn_norm``
    - ``layers.0.attention.w_qkv.weight``    ← concatenated
      ``q_proj``, ``k_proj``, ``v_proj`` on output dim
    - ``layers.0.attention.wo.weight``       ←
      ``model.layers.0.self_attn.o_proj.weight``
    - ``layers.0.feed_forward.w1.weight``    ←
      ``model.layers.0.mlp.gate_proj.weight``
    - ``layers.0.feed_forward.w3.weight``    ←
      ``model.layers.0.mlp.up_proj.weight``
    - ``layers.0.feed_forward.w2.weight``    ←
      ``model.layers.0.mlp.down_proj.weight``

Typical usage (offline, before constructing a TurboMind pipeline)::

    from lmdeploy.turbomind.eagle_draft_converter import convert_eagle3_draft

    convert_eagle3_draft(
        hf_path=\"/kaggle/input/nvidia-gpt-oss-120b-eagle3/pytorch/default/1\",
        dst_path=\"/dev/shm/models/gpt-oss-120b-eagle3-eaglenet\",
    )

    # Then use dst_path as SpeculativeConfig.model for TurboMind:
    #
    # spec_cfg = SpeculativeConfig(
    #     method=\"eagle3\",
    #     model=\"/dev/shm/models/gpt-oss-120b-eagle3-eaglenet\",
    #     num_speculative_tokens=3,
    # )
"""

from __future__ import annotations

import glob
import os
from typing import Optional

import torch
import yaml
from safetensors import safe_open
from transformers import AutoConfig


def _write_half_tensor(t: torch.Tensor, path: str) -> None:
    """Write a tensor as FP16 raw binary to ``path``."""
    t = t.to(torch.float16)
    arr = t.contiguous().cpu().numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr.tofile(path)


def _load_single_safetensors(path: str) -> str:
    """Find a single ``*.safetensors`` file under ``path``."""
    candidates = glob.glob(os.path.join(path, "*.safetensors"))
    if not candidates:
        raise RuntimeError(f"No *.safetensors files found under {path!r}")
    if len(candidates) > 1:
        # For this dedicated converter we expect a single shard; bail
        # out loudly if we see more.
        raise RuntimeError(
            f"Expected a single safetensors shard under {path!r}, found: {candidates}"
        )
    return candidates[0]


def convert_eagle3_draft(hf_path: str, dst_path: str) -> None:
    """Convert a local Eagle3 HF draft directory into EagleNet layout.

    Args:
        hf_path: Local directory containing the HF draft checkpoint
            (``config.json`` + ``model.safetensors``) for
            ``LlamaForCausalLMEagle3``.
        dst_path: Destination directory for the EagleNet draft
            (will contain ``config.yaml`` and ``*.weight`` files).
    """
    os.makedirs(dst_path, exist_ok=True)

    # 1) Load HF config to drive config.yaml.
    cfg = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
    hidden_size = int(getattr(cfg, "hidden_size"))
    vocab_size = int(getattr(cfg, "vocab_size"))
    num_heads = int(getattr(cfg, "num_attention_heads"))
    head_dim: Optional[int] = getattr(cfg, "head_dim", None)
    if head_dim is None:
        head_dim = hidden_size // num_heads
    inter_size = int(getattr(cfg, "intermediate_size"))

    tm_cfg = {
        "model_config": {
            "hidden_units": hidden_size,
            "vocab_size": vocab_size,
            "head_num": num_heads,
            "size_per_head": head_dim,
            "inter_size": inter_size,
        }
    }
    cfg_yaml = os.path.join(dst_path, "config.yaml")
    with open(cfg_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(tm_cfg, f)

    # 2) Load HF weights from model.safetensors (local file).
    st_path = _load_single_safetensors(hf_path)

    with safe_open(st_path, framework="pt") as sf:
        keys = set(sf.keys())

        def get(name: str) -> Optional[torch.Tensor]:
            if name not in keys:
                return None
            return sf.get_tensor(name)

        # Optional token embeddings.
        emb = get("model.embed_tokens.weight")
        if emb is not None:
            _write_half_tensor(emb, os.path.join(dst_path, "tok_embeddings.weight"))

        # Required: final RMSNorm and LM head.
        norm = get("model.norm.weight")
        if norm is None:
            raise RuntimeError("model.norm.weight not found in draft safetensors")
        _write_half_tensor(norm, os.path.join(dst_path, "norm.weight"))

        lm_head = get("lm_head.weight")
        if lm_head is None:
            raise RuntimeError("lm_head.weight not found in draft safetensors")
        # HF lm_head is [vocab, hidden]; EagleModule expects [hidden, vocab].
        lm_head_t = lm_head.transpose(0, 1)
        _write_half_tensor(lm_head_t, os.path.join(dst_path, "output.weight"))

        # Optional layer-0 norms and attention/MLP weights. Naming follows
        # standard LlamaForCausalLM conventions; if the Eagle3 checkpoint
        # deviates, these will simply be skipped and EagleModule will fall
        # back to RMSNorm+LM head.

        # Layer 0 norms.
        attn_norm = get("model.layers.0.input_layernorm.weight")
        if attn_norm is not None:
            _write_half_tensor(
                attn_norm,
                os.path.join(dst_path, "layers.0.attention_norm.weight"),
            )

        ffn_norm = get("model.layers.0.post_attention_layernorm.weight")
        if ffn_norm is not None:
            _write_half_tensor(
                ffn_norm,
                os.path.join(dst_path, "layers.0.ffn_norm.weight"),
            )
            # Also treat this as "hidden_norm" if present.
            _write_half_tensor(
                ffn_norm,
                os.path.join(dst_path, "layers.0.hidden_norm.weight"),
            )

        # Attention Q/K/V/O.
        q_w = get("model.layers.0.self_attn.q_proj.weight")
        k_w = get("model.layers.0.self_attn.k_proj.weight")
        v_w = get("model.layers.0.self_attn.v_proj.weight")
        o_w = get("model.layers.0.self_attn.o_proj.weight")
        if q_w is not None and k_w is not None and v_w is not None:
            # Concatenate on the output-dim side to form [hidden, 3 * hidden].
            qkv = torch.cat([q_w, k_w, v_w], dim=0)
            _write_half_tensor(
                qkv, os.path.join(dst_path, "layers.0.attention.w_qkv.weight")
            )
        if o_w is not None:
            _write_half_tensor(
                o_w, os.path.join(dst_path, "layers.0.attention.wo.weight")
            )

        # MLP gate/up/down.
        gate_w = get("model.layers.0.mlp.gate_proj.weight")
        up_w = get("model.layers.0.mlp.up_proj.weight")
        down_w = get("model.layers.0.mlp.down_proj.weight")
        if gate_w is not None:
            _write_half_tensor(
                gate_w, os.path.join(dst_path, "layers.0.feed_forward.w1.weight")
            )
        if up_w is not None:
            _write_half_tensor(
                up_w, os.path.join(dst_path, "layers.0.feed_forward.w3.weight")
            )
        if down_w is not None:
            _write_half_tensor(
                down_w, os.path.join(dst_path, "layers.0.feed_forward.w2.weight")
            )


__all__ = ["convert_eagle3_draft"]

