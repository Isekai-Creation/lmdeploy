#!/usr/bin/env python3
"""
Check micro perf gates for Eagle3 TurboMind runs.

This script reads baseline + speculative JSON outputs from
benchmark_speculative.py and enforces the micro gates:

  - 32K single: spec >= 2.0x baseline, mean_accept_len >= 3.0
  - 16K batch4: spec >= 1.0x baseline, mean_accept_len >= 2.2
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def gate_single32k(baseline: Dict[str, Any], spec: Dict[str, Any]) -> None:
    base_tp = float(baseline["throughput_tokens_per_sec"]["mean"])
    spec_tp = float(spec["throughput_tokens_per_sec"]["mean"])
    ratio = spec_tp / max(base_tp, 1e-6)

    spec_info = spec.get("eagle_speculation", {})
    mean_len = float(spec_info.get("mean_acceptance_length", 0.0))

    if ratio < 2.0 or mean_len < 3.0:
        raise SystemExit(
            f"32K single gate FAILED: ratio={ratio:.3f} (>=2.0), "
            f"mean_accept_len={mean_len:.3f} (>=3.0)"
        )


def gate_batch4_16k(baseline: Dict[str, Any], spec: Dict[str, Any]) -> None:
    base_tp = float(baseline["throughput_tokens_per_sec"]["mean"])
    spec_tp = float(spec["throughput_tokens_per_sec"]["mean"])
    ratio = spec_tp / max(base_tp, 1e-6)

    spec_info = spec.get("eagle_speculation", {})
    mean_len = float(spec_info.get("mean_acceptance_length", 0.0))

    if ratio < 1.0 or mean_len < 2.2:
        raise SystemExit(
            f"16K batch4 gate FAILED: ratio={ratio:.3f} (>=1.0), "
            f"mean_accept_len={mean_len:.3f} (>=2.2)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Eagle3 micro perf gates.")
    parser.add_argument("--mode", choices=["single32k", "batch4_16k"], required=True)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--spec-json", type=Path, required=True)
    args = parser.parse_args()

    baseline = load(args.baseline_json)
    spec = load(args.spec_json)

    if not spec.get("scenario", {}).get("micro_run", False):
        raise SystemExit("Expected micro_run=true in spec JSON.")

    if args.mode == "single32k":
        gate_single32k(baseline, spec)
    else:
        gate_batch4_16k(baseline, spec)

    print(f"{args.mode} gates PASSED.")


if __name__ == "__main__":
    main()

