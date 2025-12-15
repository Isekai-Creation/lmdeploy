"""
Example usage of speculative decoding with LMDeploy TurboMind.

Demonstrates how to use Draft/Target, EAGLE/EAGLE3, and NGram
speculative decoding via the high-level offline pipeline.
"""

from typing import List

import lmdeploy
from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig
from lmdeploy.metrics.stats import EagleMetricsSummary, SpeculativeDecodingStats
from lmdeploy.speculative_config import SpeculativeConfig


TARGET_MODEL_PATH = "/path/to/target/model"
DRAFT_MODEL_PATH = "/path/to/draft/model"
EAGLE_DRAFT_MODEL_PATH = "/path/to/eagle/draft/model"


def _run_pipeline(
    model_path: str,
    spec_config,
    prompt: str,
) -> List:
    """Helper to build a TurboMind pipeline and run a single prompt."""
    engine_config = TurbomindEngineConfig(tp=1)
    pipe = lmdeploy.pipeline(
        model_path,
        backend_config=engine_config,
        speculative_config=spec_config,
    )
    gen_cfg = GenerationConfig(
        max_new_tokens=64,
        temperature=0.0,
        top_k=20,
        top_p=0.8,
    )
    return pipe([prompt], gen_config=[gen_cfg])


def example_draft_target():
    """Example: Draft/Target speculative decoding."""
    print("=" * 80)
    print("Example 1: Draft/Target Speculative Decoding (TurboMind)")
    print("=" * 80)

    spec_config = SpeculativeConfig(
        method="draft_target",
        model=DRAFT_MODEL_PATH,
        num_speculative_tokens=3,
    )

    prompt = "Explain the concept of artificial intelligence."
    responses = _run_pipeline(TARGET_MODEL_PATH, spec_config, prompt)
    for r in responses:
        print("\n[Draft/Target] Output:\n", r.text)


def example_eagle3():
    """Example: EAGLE3 speculative decoding with metrics."""
    print("\n" + "=" * 80)
    print("Example 2: EAGLE3 Speculative Decoding (TurboMind)")
    print("=" * 80)

    spec_config = SpeculativeConfig(
        method="eagle3",
        model=EAGLE_DRAFT_MODEL_PATH,
        num_speculative_tokens=5,
        # Leave structural fields to SpeculativeConfig defaults unless tuning.
    )

    prompt = "Describe how EAGLE speculative decoding accelerates generation."
    responses = _run_pipeline(TARGET_MODEL_PATH, spec_config, prompt)
    for r in responses:
        print("\n[EAGLE3] Output:\n", r.text)

    # Aggregate EAGLE metrics from the TurboMind offline pipeline.
    stats = SpeculativeDecodingStats(
        num_spec_tokens=spec_config.num_speculative_tokens
    )
    for r in responses:
        stats.update_from_output(r)

    if stats.num_drafts > 0 and stats.num_draft_tokens > 0:
        summary = EagleMetricsSummary.from_stats(stats)
        summary_dict = summary.to_dict()
        print("\n[EAGLE3] Metrics summary:")
        for k, v in summary_dict.items():
            print(f"  {k}: {v}")
    else:
        print(
            "\n[EAGLE3] No speculative metrics were reported. "
            "Ensure the TurboMind engine was built with EAGLE enabled "
            "and that SpeculativeConfig.method is set to 'eagle3'."
        )


def example_ngram():
    """Example: NGram speculative decoding."""
    print("\n" + "=" * 80)
    print("Example 3: NGram Speculative Decoding (TurboMind)")
    print("=" * 80)

    spec_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=3,
        max_matching_ngram_size=4,
        is_public_pool=True,
    )

    prompt = "List three applications of large language models."
    responses = _run_pipeline(TARGET_MODEL_PATH, spec_config, prompt)
    for r in responses:
        print("\n[NGram] Output:\n", r.text)


def example_disabled():
    """Example: TurboMind without speculative decoding."""
    print("\n" + "=" * 80)
    print("Example 4: No Speculative Decoding (Baseline TurboMind)")
    print("=" * 80)

    prompt = "Give a short introduction to LMDeploy."
    responses = _run_pipeline(TARGET_MODEL_PATH, None, prompt)
    for r in responses:
        print("\n[Baseline] Output:\n", r.text)


if __name__ == "__main__":
    print("LMDeploy Speculative Decoding Examples (TurboMind Offline Pipeline)\n")
    print("NOTE: Update TARGET_MODEL_PATH / DRAFT_MODEL_PATH / EAGLE_DRAFT_MODEL_PATH")
    print("      to point to real TurboMind models before running.\n")

    try:
        example_draft_target()
    except Exception as e:
        print(f"Draft/Target example error: {e}")

    try:
        example_eagle3()
    except Exception as e:
        print(f"EAGLE3 example error: {e}")

    try:
        example_ngram()
    except Exception as e:
        print(f"NGram example error: {e}")

    try:
        example_disabled()
    except Exception as e:
        print(f"Disabled example error: {e}")

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)
