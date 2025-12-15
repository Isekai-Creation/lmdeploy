#!/usr/bin/env python3
"""
Utility to inspect Eagle3 GEMM shapes and prepare tuning inputs.

This script reads the JSON file emitted by the TurboMind runtime
when LMDEPLOY_EAGLE_GEMM_SHAPE_LOG=1 and LMDEPLOY_EAGLE_PERF_MODE=1
are enabled and aggregates unique shapes by tag.

It does not run the gemm2 tuner itself; instead it provides a
machine-readable summary that can be fed into existing tuning
pipelines (e.g. via TM_GEMM_TUNE / TM_GEMM_EXPORT and Warmup()).
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Eagle3 GEMM shapes from PERF_MODE runs."
    )
    parser.add_argument(
        "--shapes",
        type=Path,
        default=Path("build/eagle3_gemm_shapes_sm120.json"),
        help="Path to GEMM shapes JSON file exported by TurboMind.",
    )
    args = parser.parse_args()

    if not args.shapes.is_file():
        raise SystemExit(f"Shapes file not found: {args.shapes}")

    with args.shapes.open("r", encoding="utf-8") as f:
        shapes = json.load(f)

    if not isinstance(shapes, list):
        raise SystemExit(f"Expected a list in {args.shapes}, got {type(shapes)!r}")

    by_tag: Counter[str] = Counter()
    for s in shapes:
        tag = s.get("tag", "UNKNOWN")
        by_tag[tag] += int(s.get("count", 0) or 0)

    print(f"Loaded {len(shapes)} unique GEMM shapes from {args.shapes}")
    print("Total counts by tag:")
    for tag, cnt in sorted(by_tag.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {tag:24s} {cnt:10d}")


if __name__ == "__main__":
    main()

