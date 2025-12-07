# Copyright (c) OpenMMLab. All rights reserved.
"""
Helpers for preparing TurboMind EAGLE draft models from HuggingFace
EAGLE3 checkpoints.

This module focuses on the `nvidia/gpt-oss-120b-Eagle3` style draft
model, whose config looks like:

    {
      "architectures": ["LlamaForCausalLMEagle3"],
      "hidden_size": 2880,
      "intermediate_size": 17280,
      "num_attention_heads": 64,
      "vocab_size": 201088,
      ...
    }

The goal is **not** to perfectly reproduce the original PyTorch EAGLE3
network, but to build a *minimal, shape-correct EagleNet-style draft*
directory that TurboMind's `EagleModule::load` can consume:

    draft_dir/
      config.yaml
      tok_embeddings.weight          (optional but recommended)
      fc.weight                      (stubbed identity-style FC)
      layers.0.attention_norm.weight (optional)
      layers.0.hidden_norm.weight    (optional)
      layers.0.attention.w_qkv.weight(stubbed identity-style QKV)
      layers.0.attention.wo.weight   (stubbed identity)
      layers.0.ffn_norm.weight       (optional)
      layers.0.feed_forward.w1.weight(optional, MLP gate)
      layers.0.feed_forward.w3.weight(optional, MLP up)
      layers.0.feed_forward.w2.weight(optional, MLP down)
      norm.weight                    (required)
      output.weight                  (required, LM head)

Required semantics:

  - `config.yaml` exposes the core `model_config` fields that
    `EagleModule::load` needs: `hidden_units`, `vocab_size`,
    `head_num`, `size_per_head`, `inter_size`.
  - `norm.weight` and `output.weight` must be present; when they are
    the draft module will at least run the minimal RMSNorm + LM head
    path over the *target* model's last hidden states.

Nice-to-have semantics:

  - When available, we reuse the draft model's token embeddings and
    first-layer norms/MLP weights.
  - QKV and FC weights are currently **stubbed** as identity-style
    projections so that the shallow EagleNet block is exercised
    without trying to reverse-engineer NVIDIA's exact layout.
    This is sufficient for functional EAGLE3 experiments and keeping
    the speculative decode hot path allocation-free.
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import yaml
from safetensors import safe_open
from transformers import AutoConfig

Logger = logging.Logger


@dataclass
class EagleDraftMeta:
    """Minimal metadata for an EagleNet draft model."""

    hidden_units: int
    vocab_size: int
    head_num: int
    size_per_head: int
    inter_size: int


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _find_safetensor_shards(model_dir: str) -> Iterable[str]:
    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shards:
        raise RuntimeError(f"No .safetensors shards found under {model_dir!r}")
    return shards


def _load_tensor_from_shards(
    shards: Iterable[str],
    key: str,
    *,
    optional: bool = False,
    logger: Optional[Logger] = None,
) -> Optional[torch.Tensor]:
    """Load a single tensor key from a list of safetensors shards."""
    for path in shards:
        with safe_open(path, framework="pt", device="cpu") as f:
            if key in f.keys():
                t = f.get_tensor(key)
                if logger:
                    logger.debug("Loaded %s from %s with shape %s", key, os.path.basename(path), tuple(t.shape))
                return t
    if optional:
        if logger:
            logger.warning("Optional tensor %s not found in shards", key)
        return None
    raise RuntimeError(f"Tensor {key!r} not found in any safetensors shard")


def _save_half_tensor(t: torch.Tensor, path: str) -> None:
    """Save a tensor as raw FP16 (.weight) in row-major order."""
    _ensure_dir(os.path.dirname(path))
    t = t.detach().contiguous().to(torch.float16).cpu()
    t.numpy().tofile(path)


def _build_meta_from_config(cfg: "AutoConfig") -> EagleDraftMeta:
    hidden = int(getattr(cfg, "hidden_size"))
    vocab = int(getattr(cfg, "vocab_size"))
    heads = int(getattr(cfg, "num_attention_heads"))
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        # Fall back to a simple split; EagleModule only needs these for
        # logging / buffer sizing, not strict geometry checks.
        head_dim = hidden // heads
    inter = int(getattr(cfg, "intermediate_size"))
    return EagleDraftMeta(
        hidden_units=hidden,
        vocab_size=vocab,
        head_num=heads,
        size_per_head=int(head_dim),
        inter_size=inter,
    )


def _write_config_yaml(meta: EagleDraftMeta, out_dir: str) -> None:
    cfg: Dict[str, Dict[str, int]] = {
        "model_config": {
            "hidden_units": meta.hidden_units,
            "vocab_size": meta.vocab_size,
            "head_num": meta.head_num,
            "size_per_head": meta.size_per_head,
            "inter_size": meta.inter_size,
        }
    }
    _ensure_dir(out_dir)
    cfg_path = os.path.join(out_dir, "config.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)


def _maybe_write_identity_attention(
    meta: EagleDraftMeta,
    out_dir: str,
    logger: Optional[Logger] = None,
) -> None:
    """Write identity-style QKV / Wo / FC weights.

    These are *not* derived from the HF draft checkpoint; they are
    intentionally simple so that:

      - The shallow EagleNet block has valid shapes.
      - The speculative decode hot path can exercise the kernels.
      - We avoid making fragile assumptions about EAGLE3 internals.
    """

    h = meta.hidden_units
    # QKV: [hidden, 3 * hidden]; each of Q, K, V is identity.
    attn_qkv = torch.zeros(h, 3 * h, dtype=torch.float32)
    eye = torch.eye(h, dtype=torch.float32)
    attn_qkv[:, :h] = eye
    attn_qkv[:, h : 2 * h] = eye
    attn_qkv[:, 2 * h :] = eye
    _save_half_tensor(attn_qkv, os.path.join(out_dir, "layers.0.attention.w_qkv.weight"))

    # Wo: [hidden, hidden], identity.
    attn_o = torch.eye(h, dtype=torch.float32)
    _save_half_tensor(attn_o, os.path.join(out_dir, "layers.0.attention.wo.weight"))

    # FC: [2 * hidden, hidden]; top half is identity, bottom half zeros.
    fc = torch.zeros(2 * h, h, dtype=torch.float32)
    fc[:h, :h] = torch.eye(h, dtype=torch.float32)
    _save_half_tensor(fc, os.path.join(out_dir, "fc.weight"))

    if logger:
        logger.info(
            "Wrote identity-style draft attention/FC weights: hidden=%d (attn_qkv, attn_o, fc)",
            h,
        )


def prepare_eagle_draft_from_hf(
    hf_model_dir: str,
    out_dir: str,
    logger: Optional[Logger] = None,
) -> str:
    """Convert a HF Eagle3 draft model into an EagleNet draft directory.

    Parameters
    ----------
    hf_model_dir:
        Local path to a HuggingFace checkpoint directory, such as the
        Kaggle-mounted `/kaggle/input/nvidia-gpt-oss-120b-eagle3/...`.
    out_dir:
        Destination directory where `config.yaml` and `.weight` files
        will be written. If `config.yaml` already exists, the function
        assumes conversion has been done and returns immediately.
    logger:
        Optional logger for progress / warnings.

    Returns
    -------
    str
        The `out_dir` path (for convenience).
    """

    log = logger or logging.getLogger(__name__)
    _ensure_dir(out_dir)

    cfg_path = os.path.join(out_dir, "config.yaml")
    if os.path.exists(cfg_path):
        log.info("EAGLE draft already prepared at %s; reusing", out_dir)
        return out_dir

    log.info("Preparing TurboMind EAGLE draft from HF model at %s", hf_model_dir)

    cfg = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
    archs = getattr(cfg, "architectures", [])
    if not archs or "LlamaForCausalLMEagle3" not in archs:
        log.warning(
            "HF model at %s does not advertise LlamaForCausalLMEagle3 in architectures=%r; "
            "continuing anyway, but conversion is tuned for Eagle3-style drafts.",
            hf_model_dir,
            archs,
        )

    meta = _build_meta_from_config(cfg)
    _write_config_yaml(meta, out_dir)

    shards = list(_find_safetensor_shards(hf_model_dir))

    # --- Required tensors: model.norm.weight, lm_head.weight ---
    norm = _load_tensor_from_shards(shards, "model.norm.weight", logger=log)
    lm_head = _load_tensor_from_shards(shards, "lm_head.weight", logger=log)

    if norm.shape != (meta.hidden_units,):
        log.warning(
            "model.norm.weight shape %s != (hidden_units=%d,); proceeding but EagleModule may warn.",
            tuple(norm.shape),
            meta.hidden_units,
        )

    # HF lm_head.weight is [vocab, hidden]; EagleModule expects [hidden, vocab].
    if lm_head.shape[-1] != meta.hidden_units and lm_head.shape[0] == meta.hidden_units:
        # Some checkpoints might already store [hidden, vocab].
        lm_head_t = lm_head
    else:
        lm_head_t = lm_head.transpose(0, 1)

    _save_half_tensor(norm, os.path.join(out_dir, "norm.weight"))
    _save_half_tensor(lm_head_t, os.path.join(out_dir, "output.weight"))

    # --- Optional: token embeddings (EagleModule treats as optional) ---
    try:
        embed = _load_tensor_from_shards(
            shards, "model.embed_tokens.weight", optional=True, logger=log
        )
        if embed is not None:
            _save_half_tensor(embed, os.path.join(out_dir, "tok_embeddings.weight"))
    except Exception as exc:  # very defensive
        log.warning("Failed to export tok_embeddings.weight: %s", exc)

    # --- Optional: layer norms + MLP weights from the first Eagle layer ---
    try:
        attention_norm = _load_tensor_from_shards(
            shards, "model.layers.0.input_layernorm.weight", optional=True, logger=log
        )
        if attention_norm is not None:
            _save_half_tensor(
                attention_norm,
                os.path.join(out_dir, "layers.0.attention_norm.weight"),
            )
    except Exception as exc:
        log.warning("Failed to export layers.0.attention_norm.weight: %s", exc)

    try:
        hidden_norm = _load_tensor_from_shards(
            shards, "model.layers.0.post_attention_layernorm.weight", optional=True, logger=log
        )
        if hidden_norm is not None:
            _save_half_tensor(
                hidden_norm,
                os.path.join(out_dir, "layers.0.hidden_norm.weight"),
            )
            # Use the same vector for ffn_norm if a dedicated one is not present.
            _save_half_tensor(
                hidden_norm,
                os.path.join(out_dir, "layers.0.ffn_norm.weight"),
            )
    except Exception as exc:
        log.warning("Failed to export layers.0.hidden_norm/ffn_norm.weight: %s", exc)

    # MLP gate/up/down: try to map Eagle3 MLP into EagleNet feed_forward.
    try:
        gate = _load_tensor_from_shards(
            shards, "model.layers.0.mlp.gate_proj.weight", optional=True, logger=log
        )
        up = _load_tensor_from_shards(
            shards, "model.layers.0.mlp.up_proj.weight", optional=True, logger=log
        )
        down = _load_tensor_from_shards(
            shards, "model.layers.0.mlp.down_proj.weight", optional=True, logger=log
        )

        if gate is not None:
            _save_half_tensor(
                gate, os.path.join(out_dir, "layers.0.feed_forward.w1.weight")
            )
        if up is not None:
            _save_half_tensor(
                up, os.path.join(out_dir, "layers.0.feed_forward.w3.weight")
            )

        if down is not None:
            # Expect either [hidden, inter] or [inter, hidden]; we want [inter, hidden].
            if down.shape == (meta.inter_size, meta.hidden_units):
                down_aligned = down
            elif down.shape == (meta.hidden_units, meta.inter_size):
                down_aligned = down.transpose(0, 1)
            else:
                log.warning(
                    "Unexpected down_proj shape %s; expected (%d, %d) or (%d, %d). "
                    "Using as-is; EagleModule will still run but MLP semantics may differ.",
                    tuple(down.shape),
                    meta.inter_size,
                    meta.hidden_units,
                    meta.hidden_units,
                    meta.inter_size,
                )
                down_aligned = down
            _save_half_tensor(
                down_aligned,
                os.path.join(out_dir, "layers.0.feed_forward.w2.weight"),
            )
    except Exception as exc:
        log.warning("Failed to export Eagle3 MLP weights: %s", exc)

    # --- Stubbed attention / FC weights (identity-style) ---
    _maybe_write_identity_attention(meta, out_dir, logger=log)

    log.info("EAGLE draft export complete at %s", out_dir)
    return out_dir


__all__ = ["prepare_eagle_draft_from_hf", "EagleDraftMeta"]

