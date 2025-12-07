"""
Helpers for preparing TurboMind EAGLE draft models from HuggingFace
EAGLE/EAGLE3 checkpoints.

This is intentionally focused on the GPT‑OSS Eagle3 draft model
(``nvidia/gpt-oss-120b-Eagle3``) but is written so that it can also
handle more conventional Llama‑style drafts where the checkpoint
contains ``model.embed_tokens.weight``, ``lm_head.weight``, etc.

For TurboMind, the draft model must be stored on disk in an
``EagleModule``‑compatible layout, i.e. a directory containing:

  - ``config.yaml`` with::

        model_config:
          hidden_units: <hidden_size>
          vocab_size:   <vocab_size>
          head_num:     <num_attention_heads>
          size_per_head: <head_dim>
          inter_size:   <intermediate_size>

  - weight files like:

        tok_embeddings.weight              (optional)
        fc.weight                          (optional)
        layers.0.attention_norm.weight     (optional)
        layers.0.hidden_norm.weight        (optional)
        layers.0.ffn_norm.weight           (optional)
        layers.0.attention.w_qkv.weight    (optional)
        layers.0.attention.wo.weight       (optional)
        layers.0.feed_forward.w1.weight    (optional)
        layers.0.feed_forward.w3.weight    (optional)
        layers.0.feed_forward.w2.weight    (optional)
        norm.weight                        (required)
        output.weight                      (required)

The converter below guarantees that:

  - ``config.yaml`` is always written from the HF config.
  - ``norm.weight`` is always written from the draft checkpoint
    (EAGLE3 provides it as ``norm.weight``).
  - ``output.weight`` is written as:
      * the base model LM head if a base model directory is provided
        and we can find ``lm_head.weight`` there; otherwise
      * a synthetic, small‑variance random matrix of shape
        ``[hidden_units, vocab_size]``.
  - When draft‑specific MLP / norm weights can be located (e.g.
    ``midlayer.*`` in the Eagle3 checkpoint), they are mapped into
    the corresponding EagleNet files so the shallow draft block has
    sensible parameters.

This is sufficient to make TurboMind's EAGLE3 integration *functional*
for GPT‑OSS 120B + Eagle3: EagleModule will always have a usable
RMSNorm + LM head path, and when extra weights are present it will
run the shallow block as well. The draft model is approximate but
shape‑correct and allocation‑free.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, Iterable, Optional, Tuple

import torch
import yaml
from safetensors import safe_open
from transformers import AutoConfig


def _write_tensor(t: torch.Tensor, path: str, dtype: torch.dtype) -> None:
    """Write a tensor as raw binary with the given dtype."""
    t = t.to(dtype)
    arr = t.contiguous().cpu().numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr.tofile(path)


def _find_safetensor_shards(path: str) -> Iterable[str]:
    """Return all *.safetensors shards under ``path`` (sorted)."""
    return sorted(glob.glob(os.path.join(path, "*.safetensors")))


def _load_tensor_from_shards(
    shards: Iterable[str],
    key: str,
    *,
    optional: bool = False,
) -> Optional[torch.Tensor]:
    """Load a tensor by key from one or more safetensors shards."""
    for p in shards:
        with safe_open(p, framework="pt", device="cpu") as f:
            if key in f.keys():
                return f.get_tensor(key)
    if optional:
        return None
    raise RuntimeError(f"Tensor {key!r} not found in any safetensors shard")


def _detect_layout(keys: Iterable[str]) -> str:
    """Heuristic to detect draft checkpoint layout."""
    ks = set(keys)
    if any(k.startswith("midlayer.") for k in ks) or "fc.weight" in ks:
        return "eagle3_midlayer"
    if "lm_head.weight" in ks or any(k.startswith("model.layers.") for k in ks):
        return "llama_like"
    return "unknown"


def _collect_all_keys(shards: Iterable[str]) -> Dict[str, None]:
    keys: Dict[str, None] = {}
    for p in shards:
        with safe_open(p, framework="pt", device="cpu") as f:
            for k in f.keys():
                keys.setdefault(k, None)
    return keys


def _resolve_base_lm_head(
    base_model_dir: Optional[str],
    hidden_size: int,
    vocab_size: int,
    logger=None,
) -> torch.Tensor:
    """Try to obtain LM head weights from a base model; fallback to random."""
    log = logger
    if base_model_dir:
        # Case 1: TurboMind export (triton_models/weights/output.weight).
        tm_weights = os.path.join(
            base_model_dir, "triton_models", "weights", "output.weight"
        )
        if os.path.exists(tm_weights):
            if log:
                log.info("Using TurboMind output.weight from %s", tm_weights)
            # Raw FP16 [hidden, vocab] stored row‑major.
            num_elems = hidden_size * vocab_size
            arr = (
                torch.fromfile(tm_weights, dtype=torch.float16)
                if hasattr(torch, "fromfile")
                else torch.tensor([])
            )
            if arr.numel() == num_elems:
                return arr.view(hidden_size, vocab_size).to(torch.float32)
            if log:
                log.warning(
                    "output.weight at %s has unexpected size %d, "
                    "expected %d; falling back to HF/head or random.",
                    tm_weights,
                    arr.numel(),
                    num_elems,
                )

        # Case 2: HF base model with lm_head.weight.
        shards = _find_safetensor_shards(base_model_dir)
        if shards:
            try:
                lm_head = _load_tensor_from_shards(shards, "lm_head.weight", optional=True)
            except Exception:
                lm_head = None
            if lm_head is not None:
                if lm_head.shape == (vocab_size, hidden_size):
                    return lm_head.to(torch.float32).transpose(0, 1)
                if lm_head.shape == (hidden_size, vocab_size):
                    return lm_head.to(torch.float32)
                if log:
                    log.warning(
                        "Base lm_head.weight shape %s does not match "
                        "(%d, %d) or (%d, %d); using synthetic head.",
                        tuple(lm_head.shape),
                        vocab_size,
                        hidden_size,
                        hidden_size,
                        vocab_size,
                    )

    # Fallback: small‑variance random head.
    if log:
        log.warning(
            "Falling back to synthetic LM head for EAGLE draft "
            "(base model LM head not found)."
        )
    return torch.empty(hidden_size, vocab_size, dtype=torch.float32).normal_(0.0, 0.02)


def _convert_llama_like(
    hf_dir: str,
    out_dir: str,
    hidden_size: int,
    vocab_size: int,
    logger=None,
) -> None:
    """Handle full Llama‑style checkpoints with lm_head + layers[0]."""
    shards = _find_safetensor_shards(hf_dir)
    keys = _collect_all_keys(shards)

    # Use FP16 for standard Llama-like drafts.
    w_dtype = torch.float16

    # Token embeddings (optional).
    emb = _load_tensor_from_shards(
        shards, "model.embed_tokens.weight", optional=True
    )
    if emb is not None:
        _write_tensor(emb, os.path.join(out_dir, "tok_embeddings.weight"), w_dtype)

    # Output norm and LM head (required).
    norm = _load_tensor_from_shards(shards, "model.norm.weight")
    _write_tensor(norm, os.path.join(out_dir, "norm.weight"), w_dtype)

    lm_head = _load_tensor_from_shards(shards, "lm_head.weight")
    # HF lm_head is typically [vocab, hidden]; EagleModule expects [hidden, vocab].
    if lm_head.shape == (vocab_size, hidden_size):
        lm_head_t = lm_head.transpose(0, 1)
    else:
        lm_head_t = lm_head
    _write_tensor(lm_head_t, os.path.join(out_dir, "output.weight"), w_dtype)

    # Layer‑0 norms.
    attn_norm = _load_tensor_from_shards(
        shards, "model.layers.0.input_layernorm.weight", optional=True
    )
    if attn_norm is not None:
        _write_tensor(
            attn_norm,
            os.path.join(out_dir, "layers.0.attention_norm.weight"),
            w_dtype,
        )

    ffn_norm = _load_tensor_from_shards(
        shards, "model.layers.0.post_attention_layernorm.weight", optional=True
    )
    if ffn_norm is not None:
        _write_tensor(
            ffn_norm,
            os.path.join(out_dir, "layers.0.ffn_norm.weight"),
            w_dtype,
        )
        _write_tensor(
            ffn_norm,
            os.path.join(out_dir, "layers.0.hidden_norm.weight"),
            w_dtype,
        )

    # Attention / MLP when present – optional, purely best‑effort.
    q_w = _load_tensor_from_shards(
        shards, "model.layers.0.self_attn.q_proj.weight", optional=True
    )
    k_w = _load_tensor_from_shards(
        shards, "model.layers.0.self_attn.k_proj.weight", optional=True
    )
    v_w = _load_tensor_from_shards(
        shards, "model.layers.0.self_attn.v_proj.weight", optional=True
    )
    o_w = _load_tensor_from_shards(
        shards, "model.layers.0.self_attn.o_proj.weight", optional=True
    )
    if q_w is not None and k_w is not None and v_w is not None:
        qkv = torch.cat([q_w, k_w, v_w], dim=0)
        _write_tensor(
            qkv,
            os.path.join(out_dir, "layers.0.attention.w_qkv.weight"),
            w_dtype,
        )
    if o_w is not None:
        _write_tensor(
            o_w, os.path.join(out_dir, "layers.0.attention.wo.weight"), w_dtype
        )

    gate_w = _load_tensor_from_shards(
        shards, "model.layers.0.mlp.gate_proj.weight", optional=True
    )
    up_w = _load_tensor_from_shards(
        shards, "model.layers.0.mlp.up_proj.weight", optional=True
    )
    down_w = _load_tensor_from_shards(
        shards, "model.layers.0.mlp.down_proj.weight", optional=True
    )
    if gate_w is not None:
        _write_tensor(
            gate_w,
            os.path.join(out_dir, "layers.0.feed_forward.w1.weight"),
            w_dtype,
        )
    if up_w is not None:
        _write_tensor(
            up_w,
            os.path.join(out_dir, "layers.0.feed_forward.w3.weight"),
            w_dtype,
        )
    if down_w is not None:
        _write_tensor(
            down_w, os.path.join(out_dir, "layers.0.feed_forward.w2.weight"), w_dtype
        )


def _convert_eagle3_midlayer(
    hf_dir: str,
    out_dir: str,
    hidden_size: int,
    vocab_size: int,
    inter_size: int,
    base_model_dir: Optional[str],
    logger=None,
) -> None:
    """Handle NVIDIA GPT‑OSS Eagle3 midlayer checkpoint (model.safetensors)."""
    log = logger
    shards = _find_safetensor_shards(hf_dir)
    keys = _collect_all_keys(shards)

    def get(name: str, *, optional: bool = False) -> Optional[torch.Tensor]:
        return _load_tensor_from_shards(shards, name, optional=optional)

    # Use BF16 for Eagle3 midlayer drafts (matches HF checkpoint).
    w_dtype = torch.bfloat16

    # Output norm from draft.
    norm = get("norm.weight")
    if norm is None:
        raise RuntimeError("norm.weight not found in Eagle3 draft checkpoint")
    _write_tensor(norm, os.path.join(out_dir, "norm.weight"), w_dtype)

    # LM head: ideally from base LM; otherwise synthetic.
    lm_head = _resolve_base_lm_head(
        base_model_dir, hidden_size, vocab_size, logger=log
    )
    _write_tensor(lm_head, os.path.join(out_dir, "output.weight"), w_dtype)

    # Midlayer norms.
    attn_norm = get("midlayer.input_layernorm.weight", optional=True)
    if attn_norm is not None:
        _write_tensor(
            attn_norm,
            os.path.join(out_dir, "layers.0.attention_norm.weight"),
            w_dtype,
        )

    hidden_norm = get("midlayer.hidden_norm.weight", optional=True)
    if hidden_norm is not None:
        _write_tensor(
            hidden_norm,
            os.path.join(out_dir, "layers.0.hidden_norm.weight"),
            w_dtype,
        )

    ffn_norm = get("midlayer.post_attention_layernorm.weight", optional=True)
    if ffn_norm is not None:
        _write_tensor(
            ffn_norm,
            os.path.join(out_dir, "layers.0.ffn_norm.weight"),
            w_dtype,
        )

    # Midlayer MLP gate/up/down.
    gate_w = get("midlayer.mlp.gate_proj.weight", optional=True)
    up_w = get("midlayer.mlp.up_proj.weight", optional=True)
    down_w = get("midlayer.mlp.down_proj.weight", optional=True)

    if gate_w is not None:
        _write_tensor(
            gate_w,
            os.path.join(out_dir, "layers.0.feed_forward.w1.weight"),
            w_dtype,
        )
    if up_w is not None:
        _write_tensor(
            up_w,
            os.path.join(out_dir, "layers.0.feed_forward.w3.weight"),
            w_dtype,
        )
    if down_w is not None:
        # Expect [hidden, inter] – transpose to [inter, hidden].
        if down_w.shape == (hidden_size, inter_size):
            down_aligned = down_w.transpose(0, 1)
        else:
            down_aligned = down_w
            if log:
                log.warning(
                    "midlayer.mlp.down_proj.weight shape %s != (%d, %d); "
                    "using as‑is for w2.weight.",
                    tuple(down_w.shape),
                    hidden_size,
                    inter_size,
                )
        _write_tensor(
            down_aligned,
            os.path.join(out_dir, "layers.0.feed_forward.w2.weight"),
            w_dtype,
        )

    # Pre-FC over concatenated hidden states. The Eagle3 checkpoint's
    # `fc.weight` has shape [hidden, 3 * hidden] (out, in). For the
    # simplified EagleNet block, approximate this by reusing the part
    # that would act on the last 2 * hidden features. We then
    # transpose to obtain a [2 * hidden, hidden] matrix matching the
    # EagleNet FC geometry.
    fc_w = get("fc.weight", optional=True)
    if fc_w is not None:
        if fc_w.shape != (hidden_size, hidden_size * 3):
            if log:
                log.warning(
                    "fc.weight shape %s != (%d, %d); "
                    "skipping EagleNet fc mapping.",
                    tuple(fc_w.shape),
                    hidden_size,
                    hidden_size * 3,
                )
        else:
            fc_slice = fc_w[:, hidden_size : 3 * hidden_size]  # [hidden, 2*hidden]
            fc_eaglenet = fc_slice.transpose(0, 1)  # [2*hidden, hidden]
            _write_tensor(
                fc_eaglenet,
                os.path.join(out_dir, "fc.weight"),
                w_dtype,
            )

    # Attention weights (midlayer.self_attn.*) do not map cleanly onto
    # the simplified EagleNet geometry (which assumes [hidden, 3 *
    # hidden] QKV with hidden-dim inputs). Instead of forcing a fragile
    # reshaping, we currently provide a lightweight identity-style
    # attention block so that EagleModule can exercise its shallow
    # attention+FC path while the majority of learned behaviour comes
    # from the Eagle3 norms, FC, MLP, and LM head above.
    #
    # Concretely: Q, K, and V are each initialised as identity
    # projections over the hidden_units, and Wo is identity. This keeps
    # the draft network well-defined and allocation-stable without
    # inventing an arbitrary mapping for the midlayer.self_attn.*
    # matrices.
    eye = torch.eye(hidden_size, dtype=w_dtype)
    qkv_eye = torch.cat([eye, eye, eye], dim=1)  # [hidden, 3*hidden]
    _write_tensor(
        qkv_eye,
        os.path.join(out_dir, "layers.0.attention.w_qkv.weight"),
        w_dtype,
    )
    _write_tensor(
        eye,
        os.path.join(out_dir, "layers.0.attention.wo.weight"),
        w_dtype,
    )


def prepare_eagle_draft_from_hf(
    hf_model_dir: str,
    out_dir: str,
    *,
    base_model_dir: Optional[str] = None,
    logger=None,
) -> str:
    """Prepare an EagleNet draft directory from a HF EAGLE/EAGLE3 model.

    Parameters
    ----------
    hf_model_dir:
        Local directory containing the HF draft checkpoint (e.g.
        ``nvidia/gpt-oss-120b-Eagle3`` on Kaggle).
    out_dir:
        Destination directory where ``config.yaml`` and ``*.weight``
        files will be written.
    base_model_dir:
        Optional local directory of the *target* model (e.g.
        ``openai/gpt-oss-120b``). When provided, the converter will
        try to pull the LM head from the base model instead of
        synthesising it.
    logger:
        Optional logger for diagnostics.

    Returns
    -------
    str
        The resolved ``out_dir`` path.
    """
    log = logger
    os.makedirs(out_dir, exist_ok=True)

    cfg = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
    hidden_size = int(getattr(cfg, "hidden_size"))
    vocab_size = int(getattr(cfg, "vocab_size"))
    num_heads = int(getattr(cfg, "num_attention_heads"))
    head_dim: Optional[int] = getattr(cfg, "head_dim", None)
    if head_dim is None and num_heads > 0:
        head_dim = hidden_size // num_heads
    inter_size = int(getattr(cfg, "intermediate_size"))

    # Peek at keys to decide layout.
    shards = _find_safetensor_shards(hf_model_dir)
    if not shards:
        raise RuntimeError(f"No *.safetensors files found under {hf_model_dir!r}")
    keys = _collect_all_keys(shards).keys()
    layout = _detect_layout(keys)

    # Decide draft weight dtype and record it in config.yaml so
    # EagleModule::load can allocate tensors with the correct dtype.
    if layout == "eagle3_midlayer":
        eagle_dtype = "bf16"
    else:
        eagle_dtype = "fp16"

    tm_cfg = {
        "model_config": {
            "hidden_units": hidden_size,
            "vocab_size": vocab_size,
            "head_num": num_heads,
            "size_per_head": head_dim,
            "inter_size": inter_size,
            "eagle_weight_dtype": eagle_dtype,
        }
    }
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(tm_cfg, f)

    if layout == "eagle3_midlayer":
        if log:
            log.info("Detected Eagle3 midlayer layout for %s", hf_model_dir)
        _convert_eagle3_midlayer(
            hf_model_dir,
            out_dir,
            hidden_size,
            vocab_size,
            inter_size,
            base_model_dir,
            logger=log,
        )
    elif layout == "llama_like":
        if log:
            log.info("Detected Llama‑like layout for %s", hf_model_dir)
        _convert_llama_like(
            hf_model_dir,
            out_dir,
            hidden_size,
            vocab_size,
            logger=log,
        )
    else:
        raise RuntimeError(
            f"Unsupported EAGLE draft layout for {hf_model_dir!r}; "
            f"keys look like: {sorted(list(keys))[:16]} ..."
        )

    return out_dir


__all__ = ["prepare_eagle_draft_from_hf"]
