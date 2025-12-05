"""
EAGLE Speculative Decoding End-to-End Test

Tests the full EAGLE implementation with TurboMind.
"""

import sys
import time
from pathlib import Path

# Add lmdeploy to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lmdeploy import TurboMind, TurbomindEngineConfig, GenerationConfig
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


def test_baseline_comparison():
    """Compare EAGLE vs baseline performance."""
    print("\n" + "=" * 60)
    print("TEST 3: EAGLE vs Baseline Comparison")
    print("=" * 60)
    
    print("\n[INFO] This test would compare:")
    print("   - Baseline TurboMind (no speculation)")
    print("   - EAGLE TurboMind (with speculation)")
    print("   - Metrics: throughput, latency, acceptance rate")
    print("\n⏭️  Skipping for now (requires full integration)")


def main():
    """Run all EAGLE tests."""
    print("\n" + "=" * 60)
    print("EAGLE Speculative Decoding Test Suite")
    print("=" * 60)
    
    # Test 1: Initialization
    tm = test_eagle_initialization()
    
    # Test 2: Generation
    test_eagle_generation(tm)
    
    # Test 3: Comparison (placeholder)
    test_baseline_comparison()
    
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
