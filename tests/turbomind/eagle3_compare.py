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
from typing import Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from lmdeploy.turbomind.turbomind import _tm as _turbomind

try:
    # Optional: LMDeploy PyTorchEngine comparison path.
    from lmdeploy import pipeline as lm_pipeline
    from lmdeploy import PytorchEngineConfig, GenerationConfig as LMGenerationConfig
    from lmdeploy.speculative_config import SpeculativeConfig

    _HAVE_LMDEPLOY = True
except Exception:  # pragma: no cover - LMDeploy may not be importable in some envs
    _HAVE_LMDEPLOY = False


def _stage_stats(name: str, tm: torch.Tensor, hf: torch.Tensor | None) -> None:
    """Print simple per-stage alignment metrics between TM and HF tensors."""
    tm = tm.to(torch.float32)
    if hf is None:
        print(
            f"{name}: only TM tensor available; "
            f"shape={tuple(tm.shape)}, mean={tm.mean().item():.4f}, std={tm.std().item():.4f}"
        )
        return

    hf = hf.to(torch.float32)
    if tm.shape != hf.shape:
        print(
            f"{name}: shape mismatch TM {tuple(tm.shape)} vs HF {tuple(hf.shape)}; "
            "skipping direct diff/cosine."
        )
        return

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


def _build_hidden_and_capture_from_ids(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    device: torch.device,
    num_capture_layers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the HF target model on given ids and extract hidden/capture.

    Returns:
        last_hidden: [B, H]
        capture:     [B, H * num_capture_layers]
    """
    model.eval()
    input_ids = input_ids.to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)

    hidden_states = out.hidden_states
    last_hidden = hidden_states[-1][:, -1, :]  # [B, H]

    if num_capture_layers <= 0:
        return last_hidden, last_hidden.new_zeros((last_hidden.shape[0], 0))

    layers = []
    L = len(hidden_states)
    start = max(0, L - num_capture_layers)
    for idx in range(start, L):
        layers.append(hidden_states[idx][:, -1, :])
    capture = torch.cat(layers, dim=-1)
    return last_hidden, capture


def run_compare(
    draft_dir: str,
    target_model_name: str,
    prompt: str,
    device: str = "cuda",
    num_capture_layers: int = 3,
) -> None:
    """Load HF Eagle3 + TurboMind draft and print basic alignment stats."""
    dev = torch.device(device)

    # We only need hidden states / logits; using synthetic ids keeps this helper
    # usable even if tokenizer files are missing or non-standard. If you want
    # exact prompt-based comparison, you can replace this with real tokenization.
    cfg = AutoConfig.from_pretrained(target_model_name)
    vocab_size = int(getattr(cfg, "vocab_size", 0) or 0)
    if vocab_size <= 0:
        raise RuntimeError(
            f"Could not resolve vocab_size from config for {target_model_name!r}"
        )

    target = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": dev},
    )

    # Simple synthetic batch: one sequence of modest length.
    seq_len = 16
    batch_size = 1
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)

    # HF target hidden + capture for the last position.
    last_hidden, capture = _build_hidden_and_capture_from_ids(
        target, input_ids, dev, num_capture_layers=num_capture_layers
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

    fc_tm = tm_debug.get("fc_out")
    attn_out_tm = tm_debug.get("attn_out")
    ffn_out_tm = tm_debug.get("ffn_out")
    pre_tm = tm_debug.get("pre_head_hidden")

    # For now we use HF last_hidden as a crude reference for FC/ATTN/FFN/PRE.
    # Once a proper HF Eagle3 draft head is available, replace these with
    # real stage tensors (hf_fc, hf_attn, hf_ffn, hf_pre).
    hf_stage_ref = last_hidden_hf

    if fc_tm is not None:
        fc_tm_t = torch.from_dlpack(fc_tm.__dlpack__())
        _stage_stats("FC_OUT", fc_tm_t, hf_stage_ref)
    else:
        print("FC_OUT: not available from TurboMind debug binding.")

    if attn_out_tm is not None:
        attn_out_tm_t = torch.from_dlpack(attn_out_tm.__dlpack__())
        _stage_stats("ATTN_OUT", attn_out_tm_t, hf_stage_ref)
    else:
        print("ATTN_OUT: not available from TurboMind debug binding.")

    if ffn_out_tm is not None:
        ffn_out_tm_t = torch.from_dlpack(ffn_out_tm.__dlpack__())
        _stage_stats("FFN_OUT", ffn_out_tm_t, hf_stage_ref)
    else:
        print("FFN_OUT: not available from TurboMind debug binding.")

    if pre_tm is not None:
        pre_tm_t = torch.from_dlpack(pre_tm.__dlpack__())
        _stage_stats("PRE_HEAD_HIDDEN", pre_tm_t, hf_stage_ref)
    else:
        print("PRE_HEAD_HIDDEN: not available from TurboMind debug binding.")

    # Logits: compare directly vs HF.
    _stage_stats("LOGITS", logits_tm, logits_hf)

    # Argmax and top-k overlap.
    top1_hf = logits_hf.argmax(dim=-1)
    top1_tm = logits_tm.argmax(dim=-1)
    match_rate = (top1_hf == top1_tm).float().mean().item()

    k = 5
    topk_hf = logits_hf.topk(k, dim=-1).indices
    topk_tm = logits_tm.topk(k, dim=-1).indices
    # Any overlap in top-k per position.
    overlap = (topk_hf[..., None] == topk_tm[..., None, :]).any(dim=-1)
    topk_overlap = overlap.float().mean().item()

    print(f"\nPrompt (unused for synthetic ids): {prompt!r}")
    print(f"argmax_match_rate={match_rate:.3f}")
    print(f"top{k}_overlap={topk_overlap:.3f}")
    print(f"HF top1 IDs: {top1_hf.tolist()}")
    print(f"TM top1 IDs: {top1_tm.tolist()}")


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
        "--num-capture-layers",
        type=int,
        default=3,
        help="Number of last layers to concatenate into capture (default: 3).",
    )
    args = parser.parse_args()

    run_compare(
        draft_dir=args.draft_dir,
        target_model_name=args.target_model,
        prompt=args.prompt,
        device=args.device,
        num_capture_layers=args.num_capture_layers,
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
