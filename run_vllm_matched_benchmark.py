#!/usr/bin/env python3
"""
VLLM-matched benchmark script for testing EAGLE3 checkpoints.
Uses exact VLLM parameters: batch_size=8, context_length=8192, max_new_tokens=24576, session_len=32768
"""

import sys
import os
from datetime import datetime
sys.path.insert(0, "/workspace/aimo/VERSIONS/lmdeploy_8da9555d")

from benchmark_speculative import BenchmarkRunner

def run_vllm_matched_benchmark(model_path, spec_model_path, output_dir, scenario_name, num_spec_tokens=1):
    """Run VLLM-matched benchmark with given spec model."""
    print(f"\n{'='*60}")
    print(f"Running: {scenario_name}")
    print(f"  MODEL_PATH: {model_path}")
    print(f"  SPEC_MODEL_PATH: {spec_model_path}")
    print(f"  NUM_SPEC_TOKENS: {num_spec_tokens}")
    print(f"  BATCH_SIZE: 8, CONTEXT_LENGTH: 8192, MAX_NEW_TOKENS: 24576, SESSION_LEN: 32768")
    print(f"{'='*60}")
    
    runner = BenchmarkRunner(model_path, spec_model_path, output_dir)
    
    # Override create_pipeline to use session_len=32768
    original_create = runner.create_pipeline
    def create_pipeline_with_session_len(use_speculation=False, num_spec_tokens=3, max_batch_size=8, session_len=8192):
        return original_create(use_speculation=use_speculation, num_spec_tokens=num_spec_tokens, max_batch_size=max_batch_size, session_len=32768)
    runner.create_pipeline = create_pipeline_with_session_len
    
    # Run VLLM-matched scenario
    runner.run_test_scenario(
        scenario_name,
        batch_size=8,
        context_length=8192,
        max_new_tokens=24576,
        use_speculation=spec_model_path is not None,
        num_spec_tokens=num_spec_tokens,
        warmup_runs=1,
        measurement_runs=1,
    )

def main():
    model_path = "/workspace/aimo/models/gpt-oss-120b"
    output_dir = "/workspace/aimo/VERSIONS/lmdeploy_8da9555d/results_eagle3_micro_vllm_match"
    
    # Test scenarios
    scenarios = [
        # Normal EAGLE3 checkpoint with multiple spec tokens
        ("/workspace/aimo/models/gpt-oss-120b-eagle3", "Speculative_Batch8_Context8K_1tokens_EAGLE3_normal_VLLM", 1),
        ("/workspace/aimo/models/gpt-oss-120b-eagle3", "Speculative_Batch8_Context8K_3tokens_EAGLE3_normal_VLLM", 3),
        ("/workspace/aimo/models/gpt-oss-120b-eagle3", "Speculative_Batch8_Context8K_4tokens_EAGLE3_normal_VLLM", 4),
        ("/workspace/aimo/models/gpt-oss-120b-eagle3", "Speculative_Batch8_Context8K_5tokens_EAGLE3_normal_VLLM", 5),
        
        # Converted throughput EAGLE3 checkpoint (only spec=1 for throughput optimization)
        ("/workspace/aimo/models/turbomind_eagle_draft_gpt-oss-120b-Eagle3-throughput", "Speculative_Batch8_Context8K_1tokens_EAGLE3_throughput_converted_VLLM", 1),
    ]
    
    print(f"VLLM-Matched EAGLE3 Benchmark Suite - {datetime.now().isoformat()}")
    print(f"Total scenarios to run: {len(scenarios)}")
    
    for spec_model_path, scenario_name, num_spec_tokens in scenarios:
        try:
            run_vllm_matched_benchmark(model_path, spec_model_path, output_dir, scenario_name, num_spec_tokens)
            print(f"✅ COMPLETED: {scenario_name}")
        except Exception as e:
            print(f"❌ FAILED: {scenario_name} - {e}")
    
    print(f"\n{'='*60}")
    print("All benchmarks completed!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()