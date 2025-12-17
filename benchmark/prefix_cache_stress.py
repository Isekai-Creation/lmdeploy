#!/usr/bin/env python3
"""Prefix cache exercise for DriftEngine.

The script repeatedly sends the same prompt to warm the prefix cache and
prints DriftMetrics deltas (hits/misses/evictions, KV usage) in between to
validate reuse behaviour end-to-end.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from lmdeploy import DriftEngineConfig, GenerationConfig, pipeline as lm_pipeline


def _ensure_drift_backend(pipe) -> Any:
    engine = getattr(pipe, "engine", None)
    if engine is None:
        raise RuntimeError("AsyncEngine missing underlying TurboMind engine")
    if hasattr(engine, "_ensure_drift_engine"):
        engine._ensure_drift_engine()
    drift = getattr(engine, "_drift_engine", None)
    if drift is None:
        raise RuntimeError("Drift backend not available on this pipeline")
    return drift


def _collect_metrics(pipe) -> Dict[str, Any]:
    drift = _ensure_drift_backend(pipe)
    return drift.get_metrics()


def _build_pipeline(model_path: str, session_len: int, max_batch: int):
    drift_cfg = DriftEngineConfig(
        model_path=model_path,
        session_len=session_len,
        max_batch_size=max_batch,
    )
    drift_cfg.enable_prefix_caching = True
    return lm_pipeline(model_path, backend_config=drift_cfg)


def run_prefix_stress(pipe, prompt: str, iterations: int, max_new: int, output: Path):
    gen_cfg = GenerationConfig(max_new_tokens=max_new)
    before = _collect_metrics(pipe)
    print(f"[prefix] initial metrics: {json.dumps(before, indent=2)}")

    for i in range(iterations):
        responses = pipe([prompt], gen_config=[gen_cfg])
        print(f"  iteration {i}: generated {responses[0].generate_token_len} tokens")

    after = _collect_metrics(pipe)
    print(f"[prefix] final metrics: {json.dumps(after, indent=2)}")
    output.write_text(json.dumps({"before": before, "after": after}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Prefix cache stress harness")
    parser.add_argument("model_path", help="Model path for DriftEngine")
    parser.add_argument("--prompt", default="The history of GPUs is")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-new", type=int, default=16)
    parser.add_argument("--session-len", type=int, default=32768)
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("prefix_cache_metrics.json"))
    args = parser.parse_args()

    pipe = _build_pipeline(args.model_path, args.session_len, args.max_batch)
    run_prefix_stress(pipe, args.prompt, args.iterations, args.max_new, args.output)


if __name__ == "__main__":
    main()
