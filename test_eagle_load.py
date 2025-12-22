#!/usr/bin/env python3
"""Minimal test to check if baseline and EAGLE3 loading works."""

import sys
import gc
import torch

MODEL_PATH = "/workspace/aimo/models/gpt-oss-120b"
EAGLE_PATH = "/workspace/aimo/models/gpt-oss-120b-eagle3"


def test_baseline():
    """Test baseline loading (no EAGLE)"""
    print("=" * 60)
    print("TEST 1: Baseline loading (no EAGLE)")
    print("=" * 60)

    from lmdeploy import pipeline, TurbomindEngineConfig

    config = TurbomindEngineConfig(
        session_len=8192,
        cache_max_entry_count=0.5,
        tp=1,
    )

    print("Creating baseline pipeline...")
    pipe = pipeline(MODEL_PATH, backend_config=config)
    print("Pipeline created successfully!")

    print("Running single inference...")
    outputs = pipe(["Hello, how are you?"], max_new_tokens=10)
    print(f"Output: {outputs[0].text[:100]}...")

    print("Deleting pipeline...")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print("Baseline test PASSED!")
    return True


def test_eagle3():
    """Test EAGLE3 loading"""
    print("=" * 60)
    print("TEST 2: EAGLE3 loading")
    print("=" * 60)

    from lmdeploy import pipeline, TurbomindEngineConfig

    config = TurbomindEngineConfig(
        session_len=8192,
        cache_max_entry_count=0.5,
        tp=1,
        eagle_model_path=EAGLE_PATH,
        num_tokens_per_iter=3,
        max_prefill_token_num=3,
    )

    print("Creating EAGLE3 pipeline...")
    pipe = pipeline(MODEL_PATH, backend_config=config)
    print("Pipeline created successfully!")

    print("Running single inference...")
    outputs = pipe(["Hello, how are you?"], max_new_tokens=10)
    print(f"Output: {outputs[0].text[:100]}...")

    print("Deleting pipeline...")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print("EAGLE3 test PASSED!")
    return True


if __name__ == "__main__":
    print("Starting minimal EAGLE3 vs baseline test...\n")

    # Test baseline first
    try:
        baseline_ok = test_baseline()
    except Exception as e:
        print(f"BASELINE FAILED: {e}")
        baseline_ok = False

    print("\n" + "=" * 60 + "\n")

    # Test EAGLE3
    try:
        eagle_ok = test_eagle3()
    except Exception as e:
        print(f"EAGLE3 FAILED: {e}")
        eagle_ok = False

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Baseline: {'PASS' if baseline_ok else 'FAIL'}")
    print(f"  EAGLE3:   {'PASS' if eagle_ok else 'FAIL'}")
    print("=" * 60)
