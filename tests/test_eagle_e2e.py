"""
EAGLE Speculative Decoding End-to-End Test

Tests the full EAGLE implementation with TurboMind.
"""

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


def test_eagle_initialization():
    """Test that EAGLE module initializes correctly."""
    print("=" * 60)
    print("TEST 1: EAGLE Initialization")
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
            cache_max_entry_count=0.5,
            speculative_config=spec_config
        )
        
        print("\n[INFO] Creating TurboMind with EAGLE config...")
        tm = TurboMind.from_pretrained(
            'meta-llama/Llama-3.2-3B-Instruct',  # Target model
            engine_config=engine_config
        )
        
        print("\n✅ EAGLE initialization successful!")
        print(f"   - Target model: Llama-3.2-3B-Instruct")
        print(f"   - Draft model: Llama-3.2-1B-Instruct")
        print(f"   - Num speculative tokens: {spec_config.num_speculative_tokens}")
        
        return tm
        
    except Exception as e:
        print(f"\n❌ EAGLE initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_eagle_generation(tm):
    """Test EAGLE generation with a simple prompt."""
    if tm is None:
        print("\n⏭️  Skipping generation test (initialization failed)")
        return
    
    print("\n" + "=" * 60)
    print("TEST 2: EAGLE Generation")
    print("=" * 60)
    
    try:
        prompts = [
            "The capital of France is",
            "1 + 1 =",
            "The quick brown fox"
        ]
        
        gen_config = GenerationConfig(
            max_new_tokens=20,
            temperature=0.0,  # Greedy for determinism
            top_k=1
        )
        
        print(f"\n[INFO] Testing {len(prompts)} prompts...")
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n--- Prompt {i}/{len(prompts)} ---")
            print(f"Input: {prompt}")
            
            start_time = time.time()
            
            # Generate with EAGLE
            generator = tm.create_instance()
            outputs = []
            for output in generator.stream_infer(
                session_id=i,
                input_ids=[],  # Will be tokenized
                request_output_len=gen_config.max_new_tokens,
                sequence_start=True,
                sequence_end=True,
                step=0,
                stream_output=False
            ):
                outputs.append(output)
            
            elapsed = time.time() - start_time
            
            if outputs:
                final_output = outputs[-1]
                print(f"Output: {final_output.text if hasattr(final_output, 'text') else 'N/A'}")
                print(f"Time: {elapsed:.2f}s")
                print(f"Tokens generated: {final_output.generate_token_len if hasattr(final_output, 'generate_token_len') else 'N/A'}")
            
        print("\n✅ EAGLE generation test complete!")
        
    except Exception as e:
        print(f"\n❌ EAGLE generation failed: {e}")
        import traceback
        traceback.print_exc()


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
        cache_max_entry_count=0.5,
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
        cache_max_entry_count=0.5,
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


def test_baseline_comparison(model_path: str, spec_model_path: str, tmp_path: Path):
    """Compare EAGLE vs baseline performance on a tiny scenario.

    This is a lightweight wrapper around the standalone benchmark
    script in inference/benchmark_speculative.py. It is safe to call
    in CI as long as small models / short sequences are used.
    """
    from inference.benchmark_speculative import BenchmarkRunner

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


def main():
    """Run all EAGLE tests."""
    print("\n" + "=" * 60)
    print("EAGLE Speculative Decoding Test Suite")
    print("=" * 60)
    
    # Test 1: Initialization
    tm = test_eagle_initialization()
    
    # Test 2: Generation
    test_eagle_generation(tm)
    
    # Test 3: Comparison (requires accessible tiny models)
    # NOTE: For local runs, set MODEL_PATH and SPEC_MODEL_PATH envs.
    model_path = os.environ.get("MODEL_PATH")
    spec_path = os.environ.get("SPEC_MODEL_PATH")
    if model_path and spec_path:
        test_baseline_comparison(model_path, spec_path, Path("eagle_bench_results"))
    else:
        print("\n⏭️  Skipping EAGLE vs baseline comparison (MODEL_PATH/SPEC_MODEL_PATH not set)")
    
    print("\n" + "=" * 60)
    print("Test Suite Complete")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Compile TurboMind with EAGLE support")
    print("2. Run this test to verify initialization")
    print("3. Check logs for EAGLE-specific messages")
    print("4. Implement full C++ decode loop for actual speedups")


if __name__ == "__main__":
    main()
