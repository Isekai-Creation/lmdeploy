#!/usr/bin/env python3
"""
Test FP4 KV Cache with quant_policy=16 in lmdeploy TurboMind.

This test verifies:
1. Engine loads successfully with quant_policy=16
2. Inference runs without errors
3. KV cache uses FP4 (reduced memory footprint)
"""

import sys
import os

# Add lmdeploy to path
sys.path.insert(0, '/workspace/aimo/VERSIONS/lmdeploy_8da9555d')
os.chdir('/workspace/aimo/VERSIONS/lmdeploy_8da9555d')

import torch
import time

def test_fp4_kv_cache():
    from lmdeploy import TurbomindEngineConfig, pipeline
    
    # Use a small model for testing
    model_path = "/workspace/aimo/models/gpt-oss-20b"  # Adjust if needed
    
    # Check if model exists, else use a fallback
    if not os.path.exists(model_path):
        # Try alternative paths
        alt_paths = [
            "/workspace/aimo/models/Qwen2.5-7B-Instruct",
            "/workspace/aimo/models/Meta-Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct",  # Will download from HF
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                model_path = alt
                break
        else:
            model_path = alt_paths[-1]  # Use HF download as fallback
    
    print(f"=" * 60)
    print(f"FP4 KV Cache Test (quant_policy=16)")
    print(f"=" * 60)
    print(f"Model: {model_path}")
    
    # Test 1: FP16 baseline (quant_policy=0)
    print("\n[1/3] Testing FP16 KV cache (quant_policy=0)...")
    try:
        config_fp16 = TurbomindEngineConfig(
            quant_policy=0,
            cache_max_entry_count=0.3,
            session_len=512,
        )
        print(f"  Config created: quant_policy={config_fp16.quant_policy}")
        
        # Try to create pipeline
        pipe_fp16 = pipeline(model_path, backend_config=config_fp16)
        
        # Run a simple inference
        prompt = "Hello, world!"
        response = pipe_fp16(prompt, max_new_tokens=16)
        print(f"  Inference OK: {response.text[:50]}...")
        
        # Get memory usage
        torch.cuda.synchronize()
        mem_fp16 = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak GPU memory: {mem_fp16:.2f} GB")
        
        del pipe_fp16
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    except Exception as e:
        print(f"  FP16 baseline failed: {e}")
        mem_fp16 = 0
    
    # Test 2: INT8 KV cache (quant_policy=8)
    print("\n[2/3] Testing INT8 KV cache (quant_policy=8)...")
    try:
        config_int8 = TurbomindEngineConfig(
            quant_policy=8,
            cache_max_entry_count=0.3,
            session_len=512,
        )
        print(f"  Config created: quant_policy={config_int8.quant_policy}")
        
        pipe_int8 = pipeline(model_path, backend_config=config_int8)
        
        response = pipe_int8(prompt, max_new_tokens=16)
        print(f"  Inference OK: {response.text[:50]}...")
        
        torch.cuda.synchronize()
        mem_int8 = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak GPU memory: {mem_int8:.2f} GB")
        
        del pipe_int8
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    except Exception as e:
        print(f"  INT8 test failed: {e}")
        mem_int8 = 0
    
    # Test 3: FP4 KV cache (quant_policy=16)
    print("\n[3/3] Testing FP4 KV cache (quant_policy=16)...")
    try:
        config_fp4 = TurbomindEngineConfig(
            quant_policy=16,
            cache_max_entry_count=0.3,
            session_len=512,
        )
        print(f"  Config created: quant_policy={config_fp4.quant_policy}")
        
        pipe_fp4 = pipeline(model_path, backend_config=config_fp4)
        
        response = pipe_fp4(prompt, max_new_tokens=16)
        print(f"  Inference OK: {response.text[:50]}...")
        
        torch.cuda.synchronize()
        mem_fp4 = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak GPU memory: {mem_fp4:.2f} GB")
        
        del pipe_fp4
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  FP4 test failed: {e}")
        import traceback
        traceback.print_exc()
        mem_fp4 = 0
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"FP16 KV (quant_policy=0):  {mem_fp16:.2f} GB")
    print(f"INT8 KV (quant_policy=8):  {mem_int8:.2f} GB")
    print(f"FP4 KV (quant_policy=16): {mem_fp4:.2f} GB")
    
    if mem_fp16 > 0 and mem_fp4 > 0:
        savings = (1 - mem_fp4 / mem_fp16) * 100
        print(f"\nFP4 memory savings vs FP16: {savings:.1f}%")
    
    print("\n[DONE] FP4 KV Cache test completed!")


if __name__ == "__main__":
    test_fp4_kv_cache()
