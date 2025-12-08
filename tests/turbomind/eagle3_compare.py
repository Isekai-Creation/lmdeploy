"""
Offline helper to compare HF Eagle3 logits against TurboMind EagleModule
logits on the same hidden / capture inputs.

This is not a unit test (no pytest harness); it is intended to be run
manually on a GPU box (e.g. Kaggle) to debug Eagle3 draft-head numerics.

Usage example (from repo root):

    python -m lmdeploy.tests.turbomind.eagle3_compare \\
        --draft-dir /path/to/eagle3_draft_dir \\
        --target-model nvidia/gpt-oss-120b-Eagle3 \\
        --prompt "Hello, world"
"""

from __future__ import annotations

import argparse
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmdeploy.turbomind.turbomind import _tm as _turbomind


def _build_hidden_and_capture(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    num_capture_layers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the HF target model and extract last-layer + capture features.

    Returns:
        last_hidden: [B, H]
        capture:     [B, H * num_capture_layers]
    """
    model.eval()
    with torch.no_grad():
        toks = tokenizer(
            [prompt],
            return_tensors="pt",
        ).to(device)
        out = model(**toks, output_hidden_states=True)

    # Use the last generated position for comparison.
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

    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    target = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": dev},
    )

    last_hidden, capture = _build_hidden_and_capture(
        target,
        tokenizer,
        prompt,
        dev,
        num_capture_layers=num_capture_layers,
    )

    # HF Eagle3 logits on the same positions.
    with torch.no_grad():
        # Re-run to get logits for the final position only.
        toks = tokenizer([prompt], return_tensors="pt").to(dev)
        out = target(**toks)
        logits_hf = out.logits[:, -1, :].to(torch.float32)

    # TurboMind EagleModule logits via debug binding.
    # hidden_states / captured_hidden are expected to be rank-2 tensors.
    last_hidden_tm = last_hidden.to(dtype=torch.bfloat16, device=dev)
    capture_tm = capture.to(dtype=torch.bfloat16, device=dev)

    logits_tm_tm = _turbomind.eagle_forward_logits_debug(
        draft_dir, last_hidden_tm, capture_tm
    )
    logits_tm = torch.from_dlpack(logits_tm_tm.__dlpack__()).to(torch.float32)

    # Compare argmax and top-k overlap.
    top1_hf = logits_hf.argmax(dim=-1)
    top1_tm = logits_tm.argmax(dim=-1)
    match_rate = (top1_hf == top1_tm).float().mean().item()

    k = 5
    topk_hf = logits_hf.topk(k, dim=-1).indices
    topk_tm = logits_tm.topk(k, dim=-1).indices
    # Any overlap in top-k per position.
    overlap = (topk_hf[..., None] == topk_tm[..., None, :]).any(dim=-1)
    topk_overlap = overlap.float().mean().item()

    print(f"Prompt: {prompt!r}")
    print(f"argmax_match_rate={match_rate:.3f}")
    print(f"top{k}_overlap={topk_overlap:.3f}")
    print(f"HF top1 IDs: {top1_hf.tolist()}")
    print(f"TM top1 IDs: {top1_tm.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare HF Eagle3 vs TurboMind EagleModule logits on a simple prompt."
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
        help="Prompt to run through the models.",
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


if __name__ == "__main__":
    main()

