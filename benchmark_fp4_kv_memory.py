#!/usr/bin/env python3
"""
FP4 vs INT8 KV Cache Memory Benchmark

Compares memory pressure during decode at:
- 8192 tokens (8K)
- 32768 tokens (32K)
- 65536 tokens (64K)

Using quant_policy:
- 8  = INT8 KV cache
- 16 = FP4 KV cache (new implementation)
"""

import sys
import os
import gc
import torch
import time
from datetime import datetime

# Ensure lmdeploy is in path
LMDEPLOY_PATH = '/workspace/aimo/VERSIONS/lmdeploy_8da9555d'
sys.path.insert(0, LMDEPLOY_PATH)
os.chdir(LMDEPLOY_PATH)

def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    return allocated, reserved

def reset_gpu():
    """Reset GPU memory stats."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

def run_decode_benchmark(model_path, quant_policy, session_len, max_new_tokens=2048):
    """
    Run decode benchmark with specified session length.
    Focus on memory pressure during decoding.
    """
    from lmdeploy import TurbomindEngineConfig, pipeline
    
    reset_gpu()
    
    try:
        config = TurbomindEngineConfig(
            quant_policy=quant_policy,
            cache_max_entry_count=0.8,  # Use most of available GPU memory
            session_len=session_len,
        )
        
        print(f"      Loading engine (session_len={session_len})...")
        start = time.time()
        pipe = pipeline(model_path, backend_config=config)
        load_time = time.time() - start
        
        _, mem_after_load = get_gpu_memory()
        
        # Single long prompt to maximize decode memory pressure
        prompt = "Write an extremely detailed technical documentation about CUDA kernel optimization, " \
                 "covering memory coalescing, warp divergence, shared memory banking, occupancy optimization, " \
                 "and register pressure. Provide extensive code examples for each technique."
        
        print(f"      Running decode (max_tokens={max_new_tokens})...")
        start = time.time()
        
        # Run generation - this will allocate KV cache
        response = pipe(prompt, max_new_tokens=max_new_tokens)
        
        decode_time = time.time() - start
        
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        
        output_tokens = len(response.text.split())
        
        del pipe
        reset_gpu()
        
        return {
            'success': True,
            'load_time': load_time,
            'decode_time': decode_time,
            'peak_memory_gb': peak_mem,
            'mem_after_load_gb': mem_after_load,
            'output_tokens': output_tokens,
            'tokens_per_sec': output_tokens / decode_time if decode_time > 0 else 0,
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        reset_gpu()
        return {
            'success': False,
            'error': str(e),
            'peak_memory_gb': 0,
        }


def main():
    print("=" * 70)
    print("FP4 vs INT8 KV Cache Memory Benchmark")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Use turbomind-converted model
    model_path = "/workspace/aimo/models/gpt-oss-120b-turbomind"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return 1
    
    print(f"Model: {model_path}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Benchmark configurations - focus on decode pressure
    session_lengths = [8192, 32768, 65536]  # 8K, 32K, 64K
    max_tokens_per_session = {8192: 4096, 32768: 8192, 65536: 16384}  # Decode heavy
    
    quant_policies = {
        8: "INT8 KV",
        16: "FP4 KV",
    }
    
    results = {}
    
    for session_len in session_lengths:
        max_tokens = max_tokens_per_session[session_len]
        print(f"\n{'='*70}")
        print(f"Session Length: {session_len} tokens ({session_len//1024}K)")
        print(f"Max Decode Tokens: {max_tokens}")
        print("=" * 70)
        
        for qp, qp_name in quant_policies.items():
            print(f"\n  [{qp_name}] quant_policy={qp}")
            
            result = run_decode_benchmark(
                model_path=model_path,
                quant_policy=qp,
                session_len=session_len,
                max_new_tokens=max_tokens
            )
            
            key = f"{session_len}_{qp}"
            results[key] = result
            
            if result['success']:
                print(f"      Peak Memory: {result['peak_memory_gb']:.2f} GB")
                print(f"      Decode Time: {result['decode_time']:.1f}s")
                print(f"      Tokens/sec: {result['tokens_per_sec']:.1f}")
            else:
                print(f"      FAILED: {result['error']}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Peak Memory Usage (GB)")
    print("=" * 70)
    print(f"{'Session':<12} {'INT8 KV':<18} {'FP4 KV':<18} {'Savings':<12}")
    print("-" * 70)
    
    for session_len in session_lengths:
        int8_result = results.get(f"{session_len}_8", {})
        fp4_result = results.get(f"{session_len}_16", {})
        
        int8_mem = int8_result.get('peak_memory_gb', 0)
        fp4_mem = fp4_result.get('peak_memory_gb', 0)
        
        if int8_mem > 0 and fp4_mem > 0:
            savings = (1 - fp4_mem / int8_mem) * 100
            savings_str = f"{savings:+.1f}%"
        else:
            savings_str = "N/A"
        
        int8_str = f"{int8_mem:.2f} GB" if int8_mem > 0 else "FAILED"
        fp4_str = f"{fp4_mem:.2f} GB" if fp4_mem > 0 else "FAILED"
        
        print(f"{session_len//1024}K{'':<8} {int8_str:<18} {fp4_str:<18} {savings_str:<12}")
    
    print("=" * 70)
    print("[DONE] Benchmark completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
