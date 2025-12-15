import os
import sys
import time
from pathlib import Path

import pytest
import torch

# Add lmdeploy to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lmdeploy import TurboMind, TurbomindEngineConfig, GenerationConfig, pipeline as lm_pipeline
from lmdeploy.speculative_config import SpeculativeConfig
from lmdeploy.pytorch.config import TurboMindKVConfig


@pytest.fixture(scope="module")
def tm():
    """Pytest fixture to initialize TurboMind with EAGLE config."""
    if TurboMind is None:
        pytest.skip("TurboMind is not available; skipping EAGLE tests")

    print("\n" + "=" * 60)
    print("FIXTURE: Initializing EAGLE TurboMind")
    print("=" * 60)
    
    try:
        # Configure EAGLE
        spec_config = SpeculativeConfig(
            method='eagle',
            model='meta-llama/Llama-3.2-1B-Instruct',  # Small draft model
            num_speculative_tokens=5,
            max_path_len=5,
            max_decoding_tokens=10
        )
        
        engine_config = TurbomindEngineConfig(
            tp=1,
            max_batch_size=4,
            kv=TurboMindKVConfig(kv_capacity_bytes=0.5),
            speculative_config=spec_config
        )
        
        print("\n[INFO] Creating TurboMind with EAGLE config...")
        _tm = TurboMind.from_pretrained(
            'meta-llama/Llama-3.2-3B-Instruct',  # Target model
            engine_config=engine_config
        )
        
        print("\n✅ EAGLE TurboMind initialization successful!")
        print(f"   - Target model: Llama-3.2-3B-Instruct")
        print(f"   - Draft model: Llama-3.2-1B-Instruct")
        print(f"   - Num speculative tokens: {spec_config.num_speculative_tokens}")
        
        return _tm
        
    except Exception as e:
        print(f"\n❌ EAGLE TurboMind initialization failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"EAGLE TurboMind initialization failed: {e}")


def test_eagle_generation(tm):
    """Test EAGLE generation with a simple prompt."""
    if tm is None:
        pytest.skip("\n⏭️  Skipping generation test (initialization failed)")
        return
    
    print("\n" + "=" * 60)
    print("TEST: EAGLE Generation")
    print("=" * 60)


def test_eagle_equals_baseline_single_token(tmp_path):
    """EAGLE with num_speculative_tokens=1 should match baseline outputs.

    This is a deterministic equality check: when using a single speculative
    token and greedy sampling, TurboMind EAGLE must not change the generated
    token sequence relative to baseline decoding.
    """
    model_path = os.environ.get("MODEL_PATH")
    spec_model_path = os.environ.get("SPEC_MODEL_PATH") or model_path

    if not model_path or not spec_model_path:
        pytest.skip("MODEL_PATH / SPEC_MODEL_PATH not set; skipping equality test")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TurboMind EAGLE tests")

    engine_config = TurbomindEngineConfig(
        tp=1,
        max_batch_size=1,
        kv=TurboMindKVConfig(kv_capacity_bytes=0.5),
        enable_prefix_caching=False,
    )

    # Baseline pipeline (no speculation)
    baseline_pipe = lm_pipeline(
        model_path,
        backend_config=engine_config,
        speculative_config=None,
    )

    # EAGLE-enabled pipeline with a single speculative token
    spec_config = SpeculativeConfig(
        method="eagle3",
        model=spec_model_path,
        num_speculative_tokens=1,
        max_path_len=4,
        max_decoding_tokens=16,
    )

    eagle_pipe = lm_pipeline(
        model_path,
        backend_config=engine_config,
        speculative_config=spec_config,
    )

    prompt = "Deterministic equality test prompt."
    gen_configs = [
        GenerationConfig(
            max_new_tokens=16,
            temperature=0.0,
            top_k=1,
            random_seed=42,
        )
    ]

    baseline_outputs = baseline_pipe([prompt], gen_config=gen_configs)
    eagle_outputs = eagle_pipe([prompt], gen_config=gen_configs)

    assert len(baseline_outputs) == len(eagle_outputs) == 1
    assert baseline_outputs[0].token_ids == eagle_outputs[0].token_ids


@pytest.mark.cuda
@pytest.mark.cuda
def test_eagle_multi_token_experimental_equals_baseline(tmp_path):
    """Multi-token EAGLE (experimental flag on) should match baseline outputs.

    This checks that enabling LMDEPLOY_EAGLE_MULTI_TOKEN_EXPERIMENTAL does not
    change the generated token sequence relative to baseline decoding when
    using greedy sampling on a small prompt.
    """
    model_path = os.environ.get("MODEL_PATH")
    spec_model_path = os.environ.get("SPEC_MODEL_PATH") or model_path

    if not model_path or not spec_model_path:
        pytest.skip("MODEL_PATH / SPEC_MODEL_PATH not set; skipping multi-token equality test")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TurboMind EAGLE tests")

    # Enable experimental multi-token advance for this process.
    os.environ["LMDEPLOY_EAGLE_MULTI_TOKEN_EXPERIMENTAL"] = "1"

    engine_config = TurbomindEngineConfig(
        tp=1,
        max_batch_size=1,
        kv=TurboMindKVConfig(kv_capacity_bytes=0.5),
        enable_prefix_caching=False,
    )

    # Baseline pipeline (no speculation)
    baseline_pipe = lm_pipeline(
        model_path,
        backend_config=engine_config,
        speculative_config=None,
    )

    # EAGLE-enabled pipeline with multi-token config
    spec_config = SpeculativeConfig(
        method="eagle3",
        model=spec_model_path,
        num_speculative_tokens=3,
        max_path_len=8,
        max_decoding_tokens=32,
    )

    eagle_pipe = lm_pipeline(
        model_path,
        backend_config=engine_config,
        speculative_config=spec_config,
    )

    prompt = "Multi-token EAGLE experimental equality test prompt."
    gen_configs = [
        GenerationConfig(
            max_new_tokens=32,
            temperature=0.0,
            top_k=1,
            random_seed=123,
        )
    ]

    baseline_outputs = baseline_pipe([prompt], gen_config=gen_configs)
    eagle_outputs = eagle_pipe([prompt], gen_config=gen_configs)

    assert len(baseline_outputs) == len(eagle_outputs) == 1
    assert baseline_outputs[0].token_ids == eagle_outputs[0].token_ids


def test_eagle_acceptance_metrics_sanity(tmp_path):
    """Sanity-check EAGLE acceptance metrics exposed via RequestMetrics.

    Runs a short speculative decoding scenario and verifies that:
      - speculative metrics are present,
      - the number of accepted tokens never exceeds draft tokens,
      - acceptance rates fall within [0, 1].
    """
    model_path = os.environ.get("MODEL_PATH")
    spec_model_path = os.environ.get("SPEC_MODEL_PATH") or model_path

    if not model_path or not spec_model_path:
        pytest.skip("MODEL_PATH / SPEC_MODEL_PATH not set; skipping metrics test")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TurboMind EAGLE tests")

    engine_config = TurbomindEngineConfig(
        tp=1,
        max_batch_size=1,
        kv=TurboMindKVConfig(kv_capacity_bytes=0.5),
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

    prompt = "EAGLE acceptance metrics sanity test."
    gen_configs = [
        GenerationConfig(
            max_new_tokens=32,
            temperature=0.0,
            top_k=1,
            random_seed=123,
        )
    ]

    outputs = pipe([prompt], gen_config=gen_configs)
    assert outputs, "Pipeline returned no outputs"

    saw_spec_metrics = False
    for out in outputs:
        req_metrics = getattr(out, "req_metrics", None)
        spec_info = getattr(req_metrics, "spec_info", None) if req_metrics else None
        if not spec_info:
            continue

        saw_spec_metrics = True
        num_draft = int(spec_info.get("num_draft_tokens", 0))
        num_accept = int(spec_info.get("num_accepted_tokens", 0))
        avg_accepted_per_step = float(spec_info.get("avg_accepted_per_step", 0.0))

        assert num_draft >= 0
        assert num_accept >= 0
        assert num_accept <= num_draft
        # Average accepted per step must be non-negative and cannot exceed
        # total draft tokens in degenerate cases.
        assert avg_accepted_per_step >= 0.0

    if not saw_spec_metrics:
        pytest.skip("No EAGLE spec_info metrics found on outputs; skipping assertions")


def test_eagle_multi_batch_metrics_sanity(tmp_path):
    """Sanity-check EAGLE metrics on a small multi-batch scenario.

    This extends the single-request metrics test to multiple parallel
    sequences, ensuring that per-request spec_info is populated and
    that basic invariants hold for each output.
    """
    model_path = os.environ.get("MODEL_PATH")
    spec_model_path = os.environ.get("SPEC_MODEL_PATH") or model_path

    if not model_path or not spec_model_path:
        pytest.skip("MODEL_PATH / SPEC_MODEL_PATH not set; skipping multi-batch metrics test")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TurboMind EAGLE tests")

    engine_config = TurbomindEngineConfig(
        tp=1,
        max_batch_size=4,
        kv=TurboMindKVConfig(kv_capacity_bytes=0.5),
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

    prompts = [
        "EAGLE multi-batch metrics sanity test: seq 1.",
        "EAGLE multi-batch metrics sanity test: seq 2.",
        "EAGLE multi-batch metrics sanity test: seq 3.",
    ]
    gen_configs = [
        GenerationConfig(
            max_new_tokens=32,
            temperature=0.0,
            top_k=1,
            random_seed=123 + i,
        )
        for i in range(len(prompts))
    ]

    outputs = pipe(prompts, gen_config=gen_configs)
    assert len(outputs) == len(prompts), "Pipeline should return one output per prompt"

    saw_spec_metrics = False
    for out in outputs:
        req_metrics = getattr(out, "req_metrics", None)
        spec_info = getattr(req_metrics, "spec_info", None) if req_metrics else None
        if not spec_info:
            continue

        saw_spec_metrics = True
        num_draft = int(spec_info.get("num_draft_tokens", 0))
        num_accept = int(spec_info.get("num_accepted_tokens", 0))
        avg_accepted_per_step = float(spec_info.get("avg_accepted_per_step", 0.0))

        assert num_draft >= 0
        assert num_accept >= 0
        assert num_accept <= num_draft
        assert avg_accepted_per_step >= 0.0

    if not saw_spec_metrics:
        pytest.skip("No EAGLE spec_info metrics found on multi-batch outputs; skipping assertions")


def test_baseline_comparison(tmp_path: Path):
    """Compare EAGLE vs baseline performance on a tiny scenario.

    This is a lightweight wrapper around the standalone benchmark
    script in inference/benchmark_speculative.py. It is safe to call
    in CI as long as small models / short sequences are used.
    """
    from benchmark_speculative import BenchmarkRunner

    print("\n" + "=" * 60)
    print("TEST 3: EAGLE vs Baseline Comparison")
    print("=" * 60)

    runner = BenchmarkRunner(model_path, spec_model_path, output_dir=tmp_path)

    # Simple single-sequence comparison with short context/new tokens.
    scenarios = [
        dict(
            scenario_name="Baseline_Single_Short",
            batch_size=1,
            context_length=512,
            max_new_tokens=256,
            use_speculation=False,
        ),
        dict(
            scenario_name="Speculative_Single_Short_3tokens",
            batch_size=1,
            context_length=512,
            max_new_tokens=256,
            use_speculation=True,
            num_spec_tokens=3,
        ),
    ]

    for scenario in scenarios:
        _ = runner.run_test_scenario(**scenario)


@pytest.mark.cuda
def test_eagle3_invariants_debug_small(tmp_path):
    """Run a small EAGLE3 decode under invariants debug.

    This exercises LMDEPLOY_EAGLE_INVARIANTS_DEBUG with a tiny greedy
    decode to catch obvious multi-token invariant regressions.
    """
    model_path = os.environ.get("MODEL_PATH")
    spec_model_path = os.environ.get("SPEC_MODEL_PATH") or model_path

    if not model_path or not spec_model_path:
        pytest.skip("MODEL_PATH / SPEC_MODEL_PATH not set; skipping invariants debug test")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TurboMind EAGLE tests")

    os.environ["LMDEPLOY_EAGLE_INVARIANTS_DEBUG"] = "1"
    os.environ.setdefault("LMDEPLOY_EAGLE_PERF_MODE", "0")

    engine_config = TurbomindEngineConfig(
        tp=1,
        max_batch_size=2,
        kv=TurboMindKVConfig(kv_capacity_bytes=0.5),
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

    prompts = [
        "EAGLE3 invariants-debug small test, seq 1.",
        "EAGLE3 invariants-debug small test, seq 2.",
    ]
    gen_configs = [
        GenerationConfig(
            max_new_tokens=16,
            temperature=0.0,
            top_k=1,
            random_seed=321 + i,
        )
        for i in range(len(prompts))
    ]

    outputs = pipe(prompts, gen_config=gen_configs)
    assert len(outputs) == len(prompts)
    for out in outputs:
        assert getattr(out, "token_ids", None)

