#!/usr/bin/env python3
"""
FMHA sweep driver for Eagle3 on sm120.

This script runs micro benchmarks (no full large-context runs) across a
grid of FMHA tiling/configuration parameters and records throughput +
acceptance metrics into build/fmha_sweep_sm120.json.

It does not change any defaults by itself; it only automates data
collection so defaults can be chosen from real measurements.
"""

import argparse
import json
import os
import subprocess
from itertools import product
from pathlib import Path
from typing import Any, Dict, Tuple


def run_once(
    model_path: str,
    spec_model_path: str,
    scenario: str,
    output_dir: Path,
    kv_tile: int,
    max_tiles: int,
    block: int,
    heads_per_cta: int,
) -> Tuple[Path, Dict[str, Any]]:
    env = os.environ.copy()
    env.update(
        {
            "LMDEPLOY_EAGLE_PERF_MODE": "1",
            "LMDEPLOY_EAGLE_MICRO_STEPS": "512" if scenario == "single" else "128",
            "TM_EAGLE3_FMHA_KV_TILE": str(kv_tile),
            "TM_EAGLE3_FMHA_MAX_TILES": str(max_tiles),
            "TM_EAGLE3_FMHA_BLOCK": str(block),
            "TM_EAGLE3_FMHA_HEADS_PER_CTA": str(heads_per_cta),
        }
    )

    # Use a dedicated results subdir per config to avoid clobbering.
    tag = f"kv{kv_tile}_mt{max_tiles}_b{block}_hcta{heads_per_cta}_{scenario}"
    run_out_dir = output_dir / f"sweep_{tag}"
    run_out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "benchmark_speculative.py",
        "--model-path",
        model_path,
        "--spec-model-path",
        spec_model_path,
        "--output-dir",
        str(run_out_dir),
        "--scenario",
        "single" if scenario == "single" else "large-context",
        "--warmup-runs",
        "0",
        "--measurement-runs",
        "1",
    ]
    # Run from the repository root (parents[3]) so benchmark_speculative.py
    # is resolved correctly when this script lives under LM/lmdeploy/tools.
    subprocess.run(cmd, check=True, cwd=str(Path(__file__).resolve().parents[3]), env=env)

    # Pick the relevant JSON for this scenario.
    if scenario == "single":
        spec_name = "Speculative_Single_Context32K_3tokens.json"
    else:
        spec_name = "Speculative_Batch4_Context16K_3tokens.json"
    spec_path = run_out_dir / spec_name
    with spec_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return spec_path, data


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Eagle3 FMHA configs on sm120 (micro only).")
    parser.add_argument("--model-path", required=True, help="Path to main model.")
    parser.add_argument("--spec-model-path", required=True, help="Path to Eagle3 draft model.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/fmha_sweep_sm120.json"),
        help="Output JSON file for sweep results.",
    )
    args = parser.parse_args()

    combos = list(
        product(
            [64, 96, 128, 192, 256],  # KV_TILE
            [4, 8, 12, 16],  # MAX_TILES
            [128, 256],  # BLOCK
            [1, 2, 4],  # HEADS_PER_CTA
        )
    )

    results = []
    out_root = args.output.parent
    out_root.mkdir(parents=True, exist_ok=True)

    for kv_tile, max_tiles, block, heads_per_cta in combos:
        for scenario in ("single", "batch4"):
            spec_path, data = run_once(
                args.model_path,
                args.spec_model_path,
                scenario,
                out_root,
                kv_tile,
                max_tiles,
                block,
                heads_per_cta,
            )
            spec = data.get("eagle_speculation", {})
            entry = {
                "scenario": data.get("scenario", {}),
                "system": data.get("system", {}),
                "throughput_tokens_per_sec": data.get("throughput_tokens_per_sec", {}),
                "eagle_speculation": spec,
                "config": {
                    "kv_tile": kv_tile,
                    "max_tiles": max_tiles,
                    "block": block,
                    "heads_per_cta": heads_per_cta,
                },
                "source_json": str(spec_path),
            }
            results.append(entry)

    # Sort best-first for each scenario type by throughput.
    def key_fn(entry: Dict[str, Any]) -> float:
        return float(entry.get("throughput_tokens_per_sec", {}).get("mean", 0.0))

    single = [e for e in results if e.get("scenario", {}).get("name", "").startswith("Speculative_Single_Context32K")]
    batch4 = [e for e in results if e.get("scenario", {}).get("name", "").startswith("Speculative_Batch4_Context16K")]
    single.sort(key=key_fn, reverse=True)
    batch4.sort(key=key_fn, reverse=True)

    with args.output.open("w", encoding="utf-8") as f:
        json.dump({"single32k": single, "batch4_16k": batch4}, f, indent=2)


if __name__ == "__main__":
    main()
