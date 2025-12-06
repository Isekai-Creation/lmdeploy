"""
Offline helpers for inspecting TurboMind EAGLE runs.

These utilities stay strictly in Engineer-A scope: they build a TurboMind
pipeline with a given :class:`SpeculativeConfig`, run a few prompts, and
summarize EAGLE metrics via :class:`EagleMetricsSummary`. They do not
modify LlamaBatch, LlamaV2_eagle, DynamicDecodeLayer, or any decode
logic owned by Engineer B.
"""

from __future__ import annotations

from typing import List, Optional


def inspect_offline_eagle(
    model_path: str,
    spec_model_path: str,
    num_spec_tokens: int = 4,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 64,
) -> "dict":
    """Run a small TurboMind EAGLE pipeline and print metrics.

    Args:
        model_path: Path to the TurboMind target model.
        spec_model_path: Path to the EAGLE draft model.
        num_spec_tokens: SpeculativeConfig.num_speculative_tokens to use.
        prompts: Optional list of prompts to run. If omitted, a single
            generic prompt is used.
        max_new_tokens: Maximum tokens to generate per request.

    Returns:
        A ``dict`` produced by :meth:`EagleMetricsSummary.to_dict()` with
        aggregate EAGLE metrics for the run.

    Notes:
        - This helper requires a CUDA-enabled environment with a built
          TurboMind backend; it is intended for offline debugging / tuning,
          not for unit tests.
        - The function is side-effect free with respect to decode logic:
          it only consumes existing metrics exposed via req_metrics.spec_info.
    """
    from lmdeploy import pipeline as lm_pipeline
    from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig
    from lmdeploy.metrics.stats import (
        EagleMetricsSummary,
        SpeculativeDecodingStats,
    )
    from lmdeploy.speculative_config import (
        SpeculativeConfig,
        validate_eagle_runtime_config,
    )

    if prompts is None:
        prompts = [
            "Briefly explain how EAGLE speculative decoding accelerates "
            "TurboMind offline generation."
        ]

    spec_cfg = SpeculativeConfig(
        method="eagle3",
        model=spec_model_path,
        num_speculative_tokens=num_spec_tokens,
        eagle_debug=False,
        eagle_metrics_debug=True,
    )

    engine_config = TurbomindEngineConfig(
        tp=1,
        speculative_config=spec_cfg,
        enable_metrics=True,
    )
    validate_eagle_runtime_config(engine_config, spec_cfg)

    pipe = lm_pipeline(
        model_path,
        backend_config=engine_config,
        speculative_config=spec_cfg,
    )

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=20,
        top_p=0.8,
    )
    responses = pipe(prompts, gen_config=[gen_cfg for _ in prompts])

    stats = SpeculativeDecodingStats(num_spec_tokens=num_spec_tokens)
    for r in responses:
        stats.update_from_output(r)

    if stats.num_drafts == 0 or stats.num_draft_tokens == 0:
        print(
            "[EAGLE][inspect] No speculative metrics found; ensure the model "
            "was exported with EAGLE enabled and that the draft model path "
            "is correct."
        )
        return {}

    summary = EagleMetricsSummary.from_stats(stats)
    summary_dict = summary.to_dict()

    print("[EAGLE][inspect] Aggregate metrics:")
    for k, v in summary_dict.items():
        print(f"  {k}: {v}")

    print(
        f"[EAGLE3] num_spec_tokens={spec_cfg.num_speculative_tokens}, "
        f"mean_acceptance_length={summary.mean_acceptance_length:.3f}, "
        f"mean_acceptance_rate={summary.mean_acceptance_rate:.3f}"
    )

    return summary_dict


def eagle3_multitoken_smoke(
    model_path: str,
    spec_model_path: str,
    prompt: str = "Hello, world",
    num_spec_tokens: int = 4,
    max_new_tokens: int = 32,
) -> None:
    """Run a single TurboMind EAGLE3 multi-token pass and print a summary.

    This is a manual sanity helper for single-GPU offline runs. It builds
    a TurboMind pipeline with EAGLE3 multi-token enabled (tp=1), runs one
    prompt, and prints both the generated text and key speculative metrics.
    """
    from lmdeploy import pipeline as lm_pipeline
    from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig
    from lmdeploy.speculative_config import (
        SpeculativeConfig,
        validate_eagle_runtime_config,
    )

    spec_cfg = SpeculativeConfig(
        method="eagle3",
        model=spec_model_path,
        num_speculative_tokens=num_spec_tokens,
        eagle_debug=False,
        eagle_metrics_debug=True,
    )

    # Use a simple overshoot for session_len relative to max_new_tokens.
    engine_cfg = TurbomindEngineConfig(
        tp=1,
        session_len=max_new_tokens * 4,
        speculative_config=spec_cfg,
        enable_metrics=True,
    )

    # Validate config early so misconfigurations fail fast.
    validate_eagle_runtime_config(engine_cfg, spec_cfg)

    pipe = lm_pipeline(
        model_path,
        backend_config=engine_cfg,
        speculative_config=spec_cfg,
    )

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
    )

    outputs = pipe([prompt], gen_config=gen_cfg)
    out = outputs[0]

    print("=== EAGLE3 multi-token smoke ===")
    print(f"method=eagle3, num_spec_tokens={num_spec_tokens}, max_new_tokens={max_new_tokens}")
    print(f"tp={engine_cfg.tp}, session_len={engine_cfg.session_len}")
    print("--- Output ---")
    text = getattr(out, "text", None)
    if text is not None:
        print(text)
    else:
        print("<no text field on output>")

    metrics = getattr(out, "req_metrics", None)
    spec_info = getattr(metrics, "spec_info", None) if metrics is not None else None

    if spec_info is not None:
        print("--- EAGLE metrics ---")
        for k, v in spec_info.items():
            print(f"{k}={v}")

        num_drafts = spec_info.get("num_drafts", 0)
        total_draft = spec_info.get("total_draft_tokens", 0)
        total_accepted = spec_info.get("total_accepted_tokens", 0)
        mean_accept_len = spec_info.get("mean_acceptance_length", 0.0)

        print("--- EAGLE sanity ---")
        print(f"num_drafts>0? {num_drafts > 0}")
        print(f"multi_token_effect? mean_acceptance_length>1 => {mean_accept_len > 1.0}")
    else:
        print("WARNING: spec_info is None; EAGLE did not run or metrics were disabled.")
