import os
import sys
from pathlib import Path

import pytest
import torch

# Add lmdeploy to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lmdeploy import TurbomindEngineConfig, GenerationConfig, pipeline as lm_pipeline
from lmdeploy.pytorch.config import TurboMindKVConfig


def _get_model_path():
    """Helper to get model path from environment or provide a default."""
    model_path = os.environ.get("MODEL_PATH")
    if model_path:
        return model_path
    # Provide a small default model for testing if MODEL_PATH is not set
    # NOTE: Replace with a tiny actual model if available, e.g., a 1B param model
    pytest.skip(
        "MODEL_PATH environment variable not set. "
        "Skipping MXFP4 E2E test. "
        "Set MODEL_PATH to a valid model, e.g., internlm/internlm-chat-7b"
    )
    return "dummy/model/path" # This line should ideally not be reached


@pytest.mark.cuda
def test_mxfp4_equals_baseline_tokens():
    """
    Test MXFP4 generation against baseline (FP16/BF16) on token IDs.
    MXFP4 with quant_policy=16 should produce token IDs very close to
    quant_policy=0 (no KV quant) for greedy sampling.
    """
    model_path = _get_model_path()

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TurboMind tests")

    # Engine config for baseline (quant_policy=0)
    baseline_engine_config = TurbomindEngineConfig(
        tp=1,
        max_batch_size=1,
        kv=TurboMindKVConfig(kv_capacity_bytes=0.5),
        quant_policy=0,  # No KV quantization
        enable_prefix_caching=False,
    )

    # Engine config for MXFP4 (quant_policy=16)
    mxfp4_engine_config = TurbomindEngineConfig(
        tp=1,
        max_batch_size=1,
        kv=TurboMindKVConfig(kv_capacity_bytes=0.5),
        quant_policy=16,  # MXFP4 KV quantization
        enable_prefix_caching=False,
    )

    # Baseline pipeline
    baseline_pipe = lm_pipeline(
        model_path,
        backend_config=baseline_engine_config,
        speculative_config=None, # No speculative config for this test
    )

    # MXFP4 pipeline
    mxfp4_pipe = lm_pipeline(
        model_path,
        backend_config=mxfp4_engine_config,
        speculative_config=None, # No speculative config for this test
    )

    prompt = "Compare MXFP4 and baseline token generation results."
    gen_configs = [
        GenerationConfig(
            max_new_tokens=10,
            temperature=0.0,  # Greedy sampling
            top_k=1,
            random_seed=42,
        )
    ]

    print(f"\nRunning baseline generation for model: {model_path}")
    baseline_outputs = baseline_pipe([prompt], gen_config=gen_configs)
    print(f"Running MXFP4 generation for model: {model_path}")
    mxfp4_outputs = mxfp4_pipe([prompt], gen_config=gen_configs)

    assert len(baseline_outputs) == len(mxfp4_outputs) == 1
    
    baseline_token_ids = baseline_outputs[0].token_ids
    mxfp4_token_ids = mxfp4_outputs[0].token_ids

    print(f"Baseline Token IDs: {baseline_token_ids}")
    print(f"MXFP4 Token IDs:    {mxfp4_token_ids}")

    # For greedy sampling, token IDs should be very close or identical
    # Small differences due to quantization noise might be acceptable,
    # but for initial testing, we aim for exact match if possible or a high degree of similarity.
    # The requirement is "Differences within expected quantization error."
    # For now, let's assert token IDs are identical, then refine if needed.
    assert baseline_token_ids == mxfp4_token_ids, \
        "MXFP4 token IDs differ from baseline token IDs."
    
    # TODO: Capture and compare logits, and check for NaNs/Infs
    # This requires changes to lm_pipeline or GenerationConfig to return logits.
    # For now, we only compare token_ids.
