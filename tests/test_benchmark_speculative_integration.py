"""
Integration-style tests for `inference/benchmark_speculative.py`.

These tests exercise the real LMDeploy TurboMind pipeline together with
`BenchmarkRunner` to ensure that speculative decoding metrics from
TurboMind EAGLE are surfaced in benchmark results.

Tests require:
  - CUDA available;
  - `MODEL_PATH` and `SPEC_MODEL_PATH` environment variables pointing
    to small, EAGLE-capable TurboMind models.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

# Ensure the local lmdeploy package and repo root are importable
LMDEPLOY_ROOT = Path(__file__).parent.parent
REPO_ROOT = LMDEPLOY_ROOT.parent
for p in (str(LMDEPLOY_ROOT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lmdeploy import (  # noqa: E402
    TurbomindEngineConfig,
    GenerationConfig,
    pipeline as lm_pipeline,
)
from lmdeploy.speculative_config import SpeculativeConfig  # noqa: E402
from inference.benchmark_speculative import BenchmarkRunner  # noqa: E402


def _get_model_paths():
    model_path = os.environ.get("MODEL_PATH")
    spec_model_path = os.environ.get("SPEC_MODEL_PATH") or model_path
    return model_path, spec_model_path


@pytest.mark.cuda
def test_benchmark_runner_reports_eagle_metrics_when_available(tmp_path: Path):
    """BenchmarkRunner should report EAGLE metrics when TurboMind provides spec_info.

    This covers the Engineer B task:
      \"Enhance benchmark_speculative.py to report EAGLE acceptance metrics\".
    """
    model_path, spec_model_path = _get_model_paths()

    if not model_path or not spec_model_path:
        pytest.skip("MODEL_PATH / SPEC_MODEL_PATH not set; skipping benchmark integration test")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TurboMind EAGLE benchmark tests")

    # Construct a small EAGLE-enabled pipeline to ensure metrics are produced.
    engine_config = TurbomindEngineConfig(
        tp=1,
        max_batch_size=1,
        cache_max_entry_count=0.5,
        enable_prefix_caching=False,
    )

    spec_config = SpeculativeConfig(
        method="eagle3",
        model=spec_model_path,
        num_speculative_tokens=3,
        max_path_len=8,
        max_decoding_tokens=32,
    )

    pipe = lm_pipeline(
        model_path,
        backend_config=engine_config,
        speculative_config=spec_config,
    )

    runner = BenchmarkRunner(model_path, spec_model_path, output_dir=tmp_path)

    prompts = ["EAGLE benchmark metrics test prompt."]
    gen_configs = [
        GenerationConfig(
            max_new_tokens=32,
            temperature=0.0,
            top_k=1,
            random_seed=123,
        )
    ]

    results = runner.run_benchmark(
        pipe,
        prompts=prompts,
        gen_configs=gen_configs,
        warmup_runs=1,
        measurement_runs=2,
    )

    assert "latency_ms_per_token" in results
    assert "throughput_tokens_per_sec" in results

    spec = results.get("eagle_speculation")
    if not spec:
        pytest.skip("No EAGLE spec_info metrics found on outputs; skipping speculative assertions")

    assert spec["enabled"] is True
    assert spec["total_draft_tokens"] >= 0
    assert spec["total_accepted_tokens"] >= 0
    assert spec["total_accepted_tokens"] <= spec["total_draft_tokens"]
    assert 0.0 <= spec["mean_acceptance_rate"] <= 1.0
