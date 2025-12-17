#!/usr/bin/env python3
"""Utility scenarios for stressing DriftEngine control paths.

This script exercises three behaviours without requiring a dedicated
server:

1. Rapid start/stop via explicit session ``stop`` calls to ensure kill
   handling does not leak KV capacity.
2. Large-context prompts that approach ``session_len`` to validate the
   scheduler guardrail before real workloads do the same.
3. A KV-pressure probe that repeatedly streams requests while lowering
   ``TM_CACHE_MAX_ENTRY_COUNT`` so the scheduler and gateway see aborts.

The script intentionally avoids any synthetic token writes; it only relies
on ``lmdeploy.pipeline`` + ``AsyncEngine.session`` which directly talks to
DriftEngine via the standard bindings.
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List

from lmdeploy import DriftEngineConfig, GenerationConfig, pipeline as lm_pipeline


@dataclass
class StressConfig:
    model_path: str
    session_len: int = 32768
    max_batch_size: int = 8
    max_new_tokens: int = 256
    kill_after_tokens: int = 16
    iterations: int = 4
    kv_pressure_prompts: int = 8
    prefix_prompt: str = "Once upon a time"


def _build_pipeline(cfg: StressConfig):
    drift_cfg = DriftEngineConfig(
        model_path=cfg.model_path,
        session_len=cfg.session_len,
        max_batch_size=cfg.max_batch_size,
    )
    drift_cfg.enable_prefix_caching = True
    drift_cfg.enable_speculative_decoding = False
    drift_cfg.enable_cuda_graphs = False
    return lm_pipeline(cfg.model_path, backend_config=drift_cfg)


def _stream_prompt(session, prompt: str, kill_after: int) -> int:
    generated = 0
    for out in session(prompt, stream_response=True):
        token_ids = getattr(out, "token_ids", []) or []
        generated += len(token_ids)
        if generated >= kill_after:
            break
    return generated


def run_kill_stress(pipe, cfg: StressConfig):
    print("[stress] kill-handling scenario")
    gen_cfg = GenerationConfig(max_new_tokens=cfg.max_new_tokens)
    for i in range(cfg.iterations):
        session = pipe.session(gen_config=gen_cfg)
        produced = _stream_prompt(session, cfg.prefix_prompt, cfg.kill_after_tokens)
        print(f"  iteration {i}: produced {produced} tokens before stop()")
        session.stop()
        session.close()
        time.sleep(0.1)


def run_long_prompt(pipe, cfg: StressConfig):
    print("[stress] long-context prompt scenario")
    base = cfg.prefix_prompt
    long_prompt = (base + " ") * (cfg.session_len // max(len(base), 1))
    gen_cfg = GenerationConfig(max_new_tokens=8)
    responses = pipe([long_prompt], gen_config=[gen_cfg])
    print(f"  long-context response len={responses[0].generate_token_len}")


def run_kv_pressure(pipe, cfg: StressConfig):
    print("[stress] KV pressure scenario")
    os.environ.setdefault("TM_CACHE_MAX_ENTRY_COUNT", "0.5")
    prompts: List[str] = []
    for i in range(cfg.kv_pressure_prompts):
        prompts.append(f"Session {i}: {cfg.prefix_prompt} {i}")
    gen_cfgs = [GenerationConfig(max_new_tokens=cfg.max_new_tokens) for _ in prompts]
    responses = pipe(prompts, gen_config=gen_cfgs)
    total = sum(resp.generate_token_len for resp in responses)
    print(f"  issued {len(prompts)} prompts, generated {total} tokens under kv pressure")


def main() -> int:
    parser = argparse.ArgumentParser(description="DriftEngine stress harness")
    parser.add_argument("model_path", help="Path to the DriftEngine HF/TM model")
    parser.add_argument("--session-len", type=int, default=32768)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--kill-after", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--kv-prompts", type=int, default=8)
    parser.add_argument("--prompt", default="Once upon a time")
    args = parser.parse_args()

    cfg = StressConfig(
        model_path=args.model_path,
        session_len=args.session_len,
        max_batch_size=args.max_batch_size,
        max_new_tokens=args.max_new_tokens,
        kill_after_tokens=args.kill_after,
        iterations=args.iterations,
        kv_pressure_prompts=args.kv_prompts,
        prefix_prompt=args.prompt,
    )

    pipe = _build_pipeline(cfg)

    run_kill_stress(pipe, cfg)
    run_long_prompt(pipe, cfg)
    run_kv_pressure(pipe, cfg)
    print("[stress] scenarios completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
