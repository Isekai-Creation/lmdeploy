"""
Offline helper to compare HF Eagle3 logits and TurboMind Eagle3 draft-head
stages on the same hidden / capture inputs.

This is not a unit test (no pytest harness); it is intended to be run
manually on a GPU box (e.g. Kaggle) to debug Eagle3 draft-head numerics.

Usage example (from repo root):

    python -m lmdeploy.tests.turbomind.eagle3_compare \
        --draft-dir /path/to/eagle3_draft_dir \
        --target-model nvidia/gpt-oss-120b-Eagle3 \
        --prompt "Hello, world"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from lmdeploy.turbomind.turbomind import _tm as _turbomind
import yaml

try:
    # Optional: LMDeploy PyTorchEngine comparison path.
    from lmdeploy import pipeline as lm_pipeline
    from lmdeploy import PytorchEngineConfig, GenerationConfig as LMGenerationConfig
    from lmdeploy.speculative_config import SpeculativeConfig

    _HAVE_LMDEPLOY = True
except Exception:  # pragma: no cover - LMDeploy may not be importable in some envs
    _HAVE_LMDEPLOY = False


def _stage_stats(name: str, tm: torch.Tensor, hf: torch.Tensor | None) -> Tuple[float, float, float] | None:
    """Print simple per-stage alignment metrics between TM and HF tensors. Returns (mean_abs, max_abs, cosine) or None."""
    tm = tm.to(torch.float32)
    if hf is None:
        print(
            f"{name}: only TM tensor available; "
            f"shape={tuple(tm.shape)}, mean={tm.mean().item():.4f}, std={tm.std().item():.4f}"
        )
        return None

    hf = hf.to(torch.float32)
    if tm.shape != hf.shape:
        print(
            f"{name}: shape mismatch TM {tuple(tm.shape)} vs HF {tuple(hf.shape)}; "
            "skipping direct diff/cosine."
        )
        return None

    diff = (tm - hf).abs()
    mean_abs = diff.mean().item()
    max_abs = diff.max().item()

    tm_flat = tm.view(tm.shape[0], -1)
    hf_flat = hf.view(hf.shape[0], -1)
    num = (tm_flat * hf_flat).sum(dim=-1)
    denom = tm_flat.norm(dim=-1) * hf_flat.norm(dim=-1) + 1e-8
    cos = (num / denom).mean().item()

    print(
        f"{name}: mean_abs_diff={mean_abs:.4e}, "
        f"max_abs_diff={max_abs:.4e}, cosine={cos:.4f}"
    )
    return mean_abs, max_abs, cos


def _build_hidden_and_capture_from_ids(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    device: torch.device,
    capture_layers: Tuple[int, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the HF target model on given ids and extract hidden/capture.

    Returns:
        last_hidden: [B, H]
        capture:     [B, H * len(capture_layers)]
    """
    model.eval()
    input_ids = input_ids.to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)

    hidden_states = out.hidden_states
    last_hidden = hidden_states[-1][:, -1, :]  # [B, H]

    if not capture_layers:
        return last_hidden, last_hidden.new_zeros((last_hidden.shape[0], 0))

    layers = []
    L = len(hidden_states)
    for idx in capture_layers:
        orig_idx = idx
        if idx < 0:
            idx = L + idx
        if idx < 0 or idx >= L:
            print(f"[eagle3_compare] Skipping capture layer idx {orig_idx} (resolved {idx}) for L={L}")
            continue
        layers.append(hidden_states[idx][:, -1, :])
    if not layers:
        capture = last_hidden.new_zeros((last_hidden.shape[0], 0))
    else:
        capture = torch.cat(layers, dim=-1)
    return last_hidden, capture


def _cpu_attn_ref(qkv_tm, attn_out_tm, cfg: dict) -> None:
    """Optional tiny CPU ref on QKV when metadata is available."""
    if qkv_tm is None or attn_out_tm is None:
        return
    try:
        q_size = int(cfg.get("eagle_q_size", 0))
        kv_size = int(cfg.get("eagle_kv_size", 0))
        num_q_heads = int(cfg.get("eagle_num_heads", cfg.get("eagle_num_attention_heads", 0)))
        num_kv_heads = int(cfg.get("eagle_num_kv_heads", 0))
        head_dim = int(cfg.get("eagle_head_dim", 0))
    except Exception:
        return
    if not (q_size and kv_size and num_q_heads and num_kv_heads and head_dim):
        return

    qkv_tm_t = torch.from_dlpack(qkv_tm.__dlpack__()).to(torch.float32)
    attn_out_tm_t = torch.from_dlpack(attn_out_tm.__dlpack__()).to(torch.float32)
    if qkv_tm_t.shape[0] != attn_out_tm_t.shape[0]:
        return
    B = qkv_tm_t.shape[0]
    if qkv_tm_t.shape[1] < q_size + 2 * kv_size:
        return
    group = math.ceil(num_q_heads / num_kv_heads)
    q = qkv_tm_t[:, :q_size].view(B, num_q_heads, head_dim)
    k = qkv_tm_t[:, q_size : q_size + kv_size].view(B, num_kv_heads, head_dim)
    v = qkv_tm_t[:, q_size + kv_size : q_size + 2 * kv_size].view(B, num_kv_heads, head_dim)
    ctx = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(head_dim)
    for b in range(B):
        for h in range(num_q_heads):
            kvh = min(num_kv_heads - 1, h // group)
            score = (q[b, h] * k[b, kvh]).sum() * scale
            w = torch.softmax(score.view(1), dim=-1)
            ctx[b, h] = w * v[b, kvh]
    ctx_flat = ctx.view(B, -1)
    _stage_stats("ATTN_OUT_CPU_REF", ctx_flat, attn_out_tm_t)


def run_compare(
    draft_dir: str,
    target_model_name: str,
    prompt: str,
    device: str = "cuda",
    capture_layers: Tuple[int, ...] = (1, 17, 32),
    cosine_tol: float = 0.0,
    max_abs_tol: float = 0.0,
    topk_overlap_tol: float = 0.0,
    dump_debug: bool = False,
    cpu_ref: bool = False,
) -> None:
    """Load HF Eagle3 + TurboMind draft and print basic alignment stats."""
    dev = torch.device(device)

    cfg = AutoConfig.from_pretrained(target_model_name)
    target = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": dev},
    )

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc.input_ids.to(dev)
    except Exception:
        # Fallback: synthetic ids if tokenizer not available.
        vocab_size = int(getattr(cfg, "vocab_size", 0) or 0)
        if vocab_size <= 0:
            raise RuntimeError(
                f"Could not resolve vocab_size from config for {target_model_name!r}"
            )
        seq_len = 16
        batch_size = 1
        torch.manual_seed(0)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)

    # HF target hidden + capture for the last position.
    last_hidden, capture = _build_hidden_and_capture_from_ids(
        target, input_ids, dev, capture_layers=capture_layers
    )
    last_hidden_hf = last_hidden.to(torch.float32)

    # HF Eagle3 logits on the same position.
    with torch.no_grad():
        out = target(input_ids=input_ids)
        logits_hf = out.logits[:, -1, :].to(torch.float32)

    # TurboMind Eagle3 draft debug via eagle3_forward_debug.
    # hidden_states / captured_hidden are expected to be rank-2 tensors.
    last_hidden_tm = last_hidden.to(dtype=torch.bfloat16, device=dev)
    capture_tm = capture.to(dtype=torch.bfloat16, device=dev)

    tm_debug = _turbomind.eagle3_forward_debug(draft_dir, last_hidden_tm, capture_tm)

    logits_tm = torch.from_dlpack(tm_debug["logits"].__dlpack__()).to(torch.float32)
    logits_hf = logits_hf.to(torch.float32)

    fc_tm = tm_debug.get("fc_out")
    attn_out_tm = tm_debug.get("attn_out")
    qkv_tm = tm_debug.get("qkv")
    ffn_out_tm = tm_debug.get("ffn_out")
    pre_tm = tm_debug.get("pre_head_hidden")

    # These reference tensors are placeholders; when HF Eagle3 draft exports
    # become available, wire them here for true stagewise comparisons.
    hf_stage_ref = last_hidden_hf

    fc_stats = None
    if fc_tm is not None:
        fc_tm_t = torch.from_dlpack(fc_tm.__dlpack__())
        fc_stats = _stage_stats("FC_OUT", fc_tm_t, hf_stage_ref)
    else:
        print("FC_OUT: not available from TurboMind debug binding.")

    attn_stats = None
    if attn_out_tm is not None:
        attn_out_tm_t = torch.from_dlpack(attn_out_tm.__dlpack__())
        attn_stats = _stage_stats("ATTN_OUT", attn_out_tm_t, hf_stage_ref)
    else:
        print("ATTN_OUT: not available from TurboMind debug binding.")

    if qkv_tm is not None:
        qkv_tm_t = torch.from_dlpack(qkv_tm.__dlpack__()).to(torch.float32)
        if dump_debug:
            print(f"QKV: shape={tuple(qkv_tm_t.shape)}, mean={qkv_tm_t.mean().item():.4f}, std={qkv_tm_t.std().item():.4f}")
    else:
        print("QKV: not available from TurboMind debug binding.")

    ffn_stats = None
    if ffn_out_tm is not None:
        ffn_out_tm_t = torch.from_dlpack(ffn_out_tm.__dlpack__())
        ffn_stats = _stage_stats("FFN_OUT", ffn_out_tm_t, hf_stage_ref)
    else:
        print("FFN_OUT: not available from TurboMind debug binding.")

    pre_stats = None
    if pre_tm is not None:
        pre_tm_t = torch.from_dlpack(pre_tm.__dlpack__())
        pre_stats = _stage_stats("PRE_HEAD_HIDDEN", pre_tm_t, hf_stage_ref)
    else:
        print("PRE_HEAD_HIDDEN: not available from TurboMind debug binding.")

    # Logits: compare directly vs HF.
    logits_stats = _stage_stats("LOGITS", logits_tm, logits_hf)

    # Argmax and top-k overlap.
    top1_hf = logits_hf.argmax(dim=-1)
    top1_tm = logits_tm.argmax(dim=-1)
    match_rate = (top1_hf == top1_tm).float().mean().item()

    k = 5
    topk_hf = logits_hf.topk(k, dim=-1).indices
    topk_tm = logits_tm.topk(k, dim=-1).indices
    overlap = (topk_hf[..., None] == topk_tm[..., None, :]).any(dim=-1)
    topk_overlap = overlap.float().mean().item()

    if cosine_tol or max_abs_tol or topk_overlap_tol:
        if logits_stats is None:
            raise AssertionError("No logits stats available to apply tolerances.")
        _, max_abs_diff, cos_val = logits_stats
        if cosine_tol and cos_val < cosine_tol:
            raise AssertionError(f"logits cosine {cos_val:.4f} below tol {cosine_tol}")
        if max_abs_tol and max_abs_diff > max_abs_tol:
            raise AssertionError(f"logits max_abs_diff {max_abs_diff:.4e} above tol {max_abs_tol}")
        if topk_overlap_tol and topk_overlap < topk_overlap_tol:
            raise AssertionError(
                f"top{k} overlap {topk_overlap:.3f} below tol {topk_overlap_tol}"
            )
        # Optional stagewise checks using same tolerances when HF refs exist.
        if attn_stats is not None:
            _, attn_max, attn_cos = attn_stats
            if cosine_tol and attn_cos < cosine_tol:
                raise AssertionError(f"attn_out cosine {attn_cos:.4f} below tol {cosine_tol}")
            if max_abs_tol and attn_max > max_abs_tol:
                raise AssertionError(f"attn_out max_abs_diff {attn_max:.4e} above tol {max_abs_tol}")
        if pre_stats is not None:
            _, pre_max, pre_cos = pre_stats
            if cosine_tol and pre_cos < cosine_tol:
                raise AssertionError(f"pre_head_hidden cosine {pre_cos:.4f} below tol {cosine_tol}")
            if max_abs_tol and pre_max > max_abs_tol:
                raise AssertionError(f"pre_head_hidden max_abs_diff {pre_max:.4e} above tol {max_abs_tol}")

    print(f"\nPrompt (unused for synthetic ids): {prompt!r}")
    print(f"argmax_match_rate={match_rate:.3f}")
    print(f"top{k}_overlap={topk_overlap:.3f}")
    if dump_debug:
        print(f"HF top1 IDs: {top1_hf.tolist()}")
        print(f"TM top1 IDs: {top1_tm.tolist()}")

    if cpu_ref:
        cfg_yaml = Path(draft_dir) / "config.yaml"
        cfg_dict = {}
        if cfg_yaml.is_file():
            try:
                cfg_dict = yaml.safe_load(cfg_yaml.read_text())
            except Exception:
                cfg_dict = {}
        _cpu_attn_ref(qkv_tm, attn_out_tm, cfg_dict)


def run_lmdeploy_pytorch_compare(
    target_model_name: str,
    spec_model_name: str | None,
    prompt: str,
) -> None:
    """Optionally compare LMDeploy PyTorchEngine+Eagle3 behaviour to HF.

    Note: some GPT-OSS checkpoints (for example GPT-OSS-120B with MXFP4
    weights) use quantization methods that LMDeploy's PyTorch backend
    does not yet support. In that case we catch the RuntimeError and
    skip this comparison instead of aborting the HF↔TurboMind path.
    """
    if not _HAVE_LMDEPLOY:
        print("LMDeploy not importable; skipping LMDeploy PyTorchEngine comparison.")
        return
    if spec_model_name is None:
        print("No --spec-model provided; skipping LMDeploy PyTorchEngine comparison.")
        return

    print("\n===== LMDeploy PyTorchEngine + Eagle3 comparison =====")

    engine_cfg = PytorchEngineConfig(max_batch_size=1)
    spec_cfg = SpeculativeConfig(
        method="eagle3",
        num_speculative_tokens=3,
        model=spec_model_name,
        eagle_debug=True,
    )

    try:
        pipe = lm_pipeline(
            target_model_name,
            backend_config=engine_cfg,
            speculative_config=spec_cfg,
        )
    except RuntimeError as exc:
        msg = str(exc)
        if "Unsupported quant method" in msg or "mxfp4" in msg.lower():
            print(
                f"LMDeploy PyTorchEngine cannot be used for target={target_model_name!r} "
                f"with draft={spec_model_name!r}: {msg}. "
                "Skipping PyTorchEngine comparison; HF vs TurboMind Eagle3 draft "
                "comparison above is still valid."
            )
            return
        raise

    gen_cfg = LMGenerationConfig(
        max_new_tokens=1,
        temperature=0.0,
        top_k=20,
        top_p=0.8,
    )

    outputs = pipe([prompt], gen_config=[gen_cfg])
    out = outputs[0]
    token_ids = out.token_ids
    print(
        f"LMDeploy PyTorchEngine first new token id: {token_ids[-1] if token_ids else 'N/A'}"
    )

    spec_info = getattr(getattr(out, "req_metrics", None), "spec_info", None)
    if spec_info is not None:
        print("LMDeploy PyTorchEngine EAGLE spec_info:", spec_info)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare HF Eagle3 vs TurboMind Eagle3 draft-head logits on a simple prompt, "
            "and optionally LMDeploy PyTorchEngine+Eagle3 behaviour."
        )
    )
    parser.add_argument(
        "--draft-dir",
        required=True,
        help="Path to the TurboMind Eagle3 draft directory (config.yaml + weights).",
    )
    parser.add_argument(
        "--target-model",
        required=True,
        help="HF model id or path for the Eagle3 target (e.g. nvidia/gpt-oss-120b-Eagle3).",
    )
    parser.add_argument(
        "--prompt",
        default="Hello, Eagle3.",
        help="Prompt to run through the models (used only for LMDeploy path here).",
    )
    parser.add_argument(
        "--spec-model",
        default=None,
        help=(
            "Optional HF Eagle3 draft model id/path for LMDeploy PyTorchEngine "
            "(e.g. nvidia/gpt-oss-120b-Eagle3). When provided, the script will "
            "also run a one-step LMDeploy PyTorchEngine+Eagle3 decode for comparison."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device to run on (default: cuda).",
    )
    parser.add_argument(
        "--capture-layer-ids",
        type=str,
        default="1,17,32",
        help="Comma-separated HF hidden-state indices to capture (default matches TRT Eagle3: 1,17,32).",
    )
    parser.add_argument(
        "--cosine-tol",
        type=float,
        default=0.0,
        help="Optional cosine tolerance on logits; raise if below.",
    )
    parser.add_argument(
        "--max-abs-tol",
        type=float,
        default=0.0,
        help="Optional max-abs-diff tolerance on logits; raise if above.",
    )
    parser.add_argument(
        "--topk-overlap-tol",
        type=float,
        default=0.0,
        help="Optional top-k overlap tolerance on logits; raise if below.",
    )
    parser.add_argument(
        "--dump-debug",
        action="store_true",
        help="Print extra debug stats (QKV/top1 lists) for manual inspection.",
    )
    parser.add_argument(
        "--cpu-ref",
        action="store_true",
        help="Run a tiny CPU attention reference when metadata is available.",
    )
    args = parser.parse_args()

    capture_layers = tuple(int(x) for x in args.capture_layer_ids.split(",") if x != "")

    run_compare(
        draft_dir=args.draft_dir,
        target_model_name=args.target_model,
        prompt=args.prompt,
        device=args.device,
        capture_layers=capture_layers,
        cosine_tol=args.cosine_tol,
        max_abs_tol=args.max_abs_tol,
        topk_overlap_tol=args.topk_overlap_tol,
        dump_debug=args.dump_debug,
        cpu_ref=args.cpu_ref,
    )

    # Optional LMDeploy PyTorchEngine comparison: compares the first new token
    # and spec_info metrics for a one-step Eagle3 decode on the same prompt.
    if args.spec_model is not None:
        run_lmdeploy_pytorch_compare(
            target_model_name=args.target_model,
            spec_model_name=args.spec_model,
            prompt=args.prompt,
        )


if __name__ == "__main__":
    main()
