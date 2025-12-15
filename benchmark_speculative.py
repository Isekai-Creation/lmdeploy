#!/usr/bin/env python3
"""
Comprehensive benchmark script for speculative decoding validation.

Tests multiple configurations and collects detailed metrics.
"""

import argparse
import json
import time
import torch
import psutil
import os
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

# LMDeploy imports
from lmdeploy import pipeline as lm_pipeline
from lmdeploy import TurbomindEngineConfig, GenerationConfig
from lmdeploy.lib import _turbomind as _tm
from lmdeploy.metrics.stats import SpeculativeDecodingStats, EagleMetricsSummary
from lmdeploy.speculative_config import SpeculativeConfig, validate_eagle_runtime_config


class BenchmarkRunner:
    """Run benchmarks with detailed metric collection."""

    def __init__(
        self, model_path: str, spec_model_path: str = None, output_dir: str = "results"
    ):
        self.model_path = model_path
        self.spec_model_path = spec_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # GPU info
        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory_total = (
                torch.cuda.get_device_properties(0).total_memory / 1e9
            )
        else:
            self.gpu_name = "No GPU"
            self.gpu_memory_total = 0

    def create_pipeline(
        self,
        use_speculation: bool = False,
        num_spec_tokens: int = 3,
        max_batch_size: int = 8,
        session_len: int = 32768,
    ):
        """Create LMDeploy pipeline with specified config."""
        # Allow overriding TurboMind KV cache fraction via environment.
        # This controls how much of device memory can be used for KV
        # cache entries. Default to 0.75 to reduce OOM risk on large
        # models while keeping long-context benchmarks feasible.
        cache_max_entry = 0.75
        cache_env = os.getenv("TM_CACHE_MAX_ENTRY_COUNT")
        if cache_env:
            try:
                v = float(cache_env)
                if 0.0 < v <= 1.0:
                    cache_max_entry = v
            except ValueError:
                pass

        engine_config = TurbomindEngineConfig(
            tp=1,
            quant_policy=8,
            session_len=session_len,
            max_batch_size=max_batch_size,
            cache_max_entry_count=cache_max_entry,
            enable_prefix_caching=False,
        )

        spec_cfg = None
        if use_speculation and self.spec_model_path:
            eagle_debug_env = os.getenv("LMDEPLOY_EAGLE_DEBUG", "").strip().lower()
            eagle_metrics_debug_env = os.getenv("LMDEPLOY_EAGLE_METRICS_DEBUG", "").strip().lower()
            eagle_debug = eagle_debug_env in ("1", "true", "yes", "on")
            eagle_metrics_debug = eagle_metrics_debug_env in ("1", "true", "yes", "on")
            spec_cfg = SpeculativeConfig(
                method="eagle3",
                num_speculative_tokens=num_spec_tokens,
                model=self.spec_model_path,
                eagle_debug=eagle_debug,
                eagle_metrics_debug=eagle_metrics_debug,
            )
            # Validate TurboMind + EAGLE3 runtime config before we touch
            # the GPU engine, so obvious misconfigurations fail fast with
            # a Python exception instead of a C++ segfault.
            validate_eagle_runtime_config(engine_config, spec_cfg)

        pipe = lm_pipeline(
            self.model_path,
            backend_config=engine_config,
            speculative_config=spec_cfg,
        )

        return pipe

    def measure_memory(self) -> Dict[str, float]:
        """Measure current memory usage."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "utilization_pct": (allocated / self.gpu_memory_total) * 100,
            }
        return {"allocated_gb": 0, "reserved_gb": 0, "utilization_pct": 0}

    def run_benchmark(
        self,
        pipe,
        prompts: List[str],
        gen_configs: List[GenerationConfig],
        warmup_runs: int = 2,
        measurement_runs: int = 2,
        num_spec_tokens: int | None = None,
    ) -> Dict[str, Any]:
        """Run benchmark and collect metrics."""
        # Allow callers to override run counts via environment variables so
        # stability sweeps can run quickly without changing CLI signatures.
        try:
            warmup_runs = int(os.getenv("SPEC_SUITE_WARMUP_RUNS", warmup_runs))
            measurement_runs = int(
                os.getenv("SPEC_SUITE_MEASUREMENT_RUNS", measurement_runs)
            )
        except ValueError:
            pass

        # Warmup
        print(f"  Warmup ({warmup_runs} runs)...")
        for _ in range(warmup_runs):
            _ = pipe(prompts, gen_config=gen_configs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Measurement
        print(f"  Measurement ({measurement_runs} runs)...")
        latencies = []
        throughputs = []
        memory_usages = []

        # Aggregate EAGLE speculative decoding metrics across all runs
        # when TurboMind populates req_metrics.spec_info. Only create
        # stats when speculation is actually enabled so baseline runs
        # do not allocate or emit EAGLE summaries.
        spec_stats: SpeculativeDecodingStats | None = None
        if num_spec_tokens is not None and num_spec_tokens > 0:
            spec_stats = SpeculativeDecodingStats(num_spec_tokens=num_spec_tokens)

        for run_idx in range(measurement_runs):
            # Measure memory before
            mem_before = self.measure_memory()

            # Run inference
            start_time = time.time()
            responses = pipe(prompts, gen_config=gen_configs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            # Measure memory after
            mem_after = self.measure_memory()

            # Calculate core latency/throughput metrics.
            total_time = end_time - start_time
            total_tokens = sum(len(r.token_ids) for r in responses)

            latency_per_token = (total_time / total_tokens) * 1000  # ms
            throughput = total_tokens / total_time  # tokens/sec

            latencies.append(latency_per_token)
            throughputs.append(throughput)
            memory_usages.append(mem_after)

            # Consume speculative decoding metrics if present on outputs.
            if spec_stats is not None:
                for r in responses:
                    if getattr(r, "req_metrics", None) is not None:
                        spec_stats.update_from_output(r)

            print(
                f"    Run {run_idx + 1}: {throughput:.1f} tok/s, "
                f"{latency_per_token:.2f} ms/tok, "
                f"{mem_after['allocated_gb']:.2f} GB"
            )

        # Aggregate results
        results: Dict[str, Any] = {
            "latency_ms_per_token": {
                "mean": sum(latencies) / len(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "values": latencies,
            },
            "throughput_tokens_per_sec": {
                "mean": sum(throughputs) / len(throughputs),
                "min": min(throughputs),
                "max": max(throughputs),
                "values": throughputs,
            },
            "memory_gb": {
                "mean": sum(m["allocated_gb"] for m in memory_usages)
                / len(memory_usages),
                "max": max(m["allocated_gb"] for m in memory_usages),
                "utilization_pct": sum(m["utilization_pct"] for m in memory_usages)
                / len(memory_usages),
            },
            "total_runs": measurement_runs,
        }

        # Attach an EAGLE-specific metrics summary only when the engine
        # reported speculative decoding stats (i.e. EAGLE was enabled).
        if (
            spec_stats is not None
            and spec_stats.num_drafts > 0
            and spec_stats.num_draft_tokens > 0
        ):
            summary = EagleMetricsSummary.from_stats(spec_stats)
            spec_dict = summary.to_dict()
            spec_dict["enabled"] = True
            results["eagle_speculation"] = spec_dict

        return results

    def run_test_scenario(
        self,
        scenario_name: str,
        batch_size: int,
        context_length: int,
        max_new_tokens: int,
        use_speculation: bool,
        num_spec_tokens: int = 3,
        warmup_runs: int = 2,
        measurement_runs: int = 2,
    ) -> Dict[str, Any]:
        """Run a complete test scenario."""

        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print(f"  Batch size: {batch_size}")
        print(f"  Context length: {context_length}")
        print(f"  Max new tokens: {max_new_tokens}")
        print(
            f"  Speculation: {'ON (' + str(num_spec_tokens) + ' tokens)' if use_speculation else 'OFF'}"
        )
        perf_mode_env = os.getenv("LMDEPLOY_EAGLE_PERF_MODE", "").strip().lower()
        perf_mode = perf_mode_env in ("1", "true", "yes", "on")
        if perf_mode:
            print("  PERF_MODE=1 stop_words=OFF gating=OFF")
        print(f"{'='*60}")

        # Optional escape hatch: when LMDEPLOY_EAGLE_DISABLE_BASELINE is set,
        # skip running non-speculative (baseline) scenarios entirely. This is
        # useful when the baseline path is temporarily broken but we still
        # need to exercise EAGLE3 speculative decode and collect GEMM/FMHA
        # shape telemetry. The stub result keeps JSON structure consistent
        # without touching the GPU engine.
        disable_baseline_env = os.getenv(
            "LMDEPLOY_EAGLE_DISABLE_BASELINE", ""
        ).strip().lower()
        disable_baseline = disable_baseline_env in ("1", "true", "yes", "on")
        if disable_baseline and not use_speculation:
            print(
                "LMDEPLOY_EAGLE_DISABLE_BASELINE=1 and use_speculation=False; "
                "skipping baseline scenario without running the engine."
            )

            micro_steps_env = os.getenv("LMDEPLOY_EAGLE_MICRO_STEPS", "").strip()
            micro_steps = None
            if micro_steps_env:
                try:
                    v = int(micro_steps_env)
                    if v > 0:
                        micro_steps = v
                except ValueError:
                    micro_steps = None

            results: Dict[str, Any] = {
                "latency_ms_per_token": {
                    "mean": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "values": [],
                },
                "throughput_tokens_per_sec": {
                    "mean": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "values": [],
                },
                "memory_gb": {
                    "mean": 0.0,
                    "max": 0.0,
                    "utilization_pct": 0.0,
                },
                "total_runs": 0,
            }

            results["scenario"] = {
                "name": scenario_name,
                "batch_size": batch_size,
                "context_length": context_length,
                "max_new_tokens": max_new_tokens,
                "use_speculation": use_speculation,
                "num_spec_tokens": num_spec_tokens if use_speculation else 0,
                "perf_mode": perf_mode,
                "micro_run": micro_steps is not None,
            }

            results["system"] = {
                "gpu_name": self.gpu_name,
                "gpu_memory_total_gb": self.gpu_memory_total,
                "timestamp": datetime.now().isoformat(),
                "turbomind_build_id": _tm.build_id(),
            }

            filename = f"{scenario_name.replace(' ', '_').lower()}.json"
            filepath = self.output_dir / filename
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2)

            print(f"\nBaseline scenario skipped; stub results saved to: {filepath}")
            return results

        # Create pipeline
        print("Creating pipeline...")
        pipe = self.create_pipeline(
            use_speculation=use_speculation,
            num_spec_tokens=num_spec_tokens,
            max_batch_size=batch_size,
        )

        # Generate prompts
        prompts = [
            "Explain the concept of quantum computing in detail."
            * (context_length // 50)
        ] * batch_size

        perf_mode_env = os.getenv("LMDEPLOY_EAGLE_PERF_MODE", "").strip().lower()
        perf_mode = perf_mode_env in ("1", "true", "yes", "on")

        # Generation configs. Allow sampling parameters to be overridden
        # via environment so we can sweep temperature / top_p / top_k /
        # min_p without changing the CLI or code in this file. In PERF_MODE
        # we force greedy decoding (no sampling) so that the verifier's
        # target tokens (argmax) match DynamicDecode's committed tokens.
        if perf_mode:
            temperature = 0.0
            top_k = 0
            top_p = 1.0
            min_p = 0.0
        else:
            temperature = float(os.getenv("SPEC_SUITE_TEMPERATURE", "0.0"))
            top_k = int(os.getenv("SPEC_SUITE_TOP_K", "20"))
            top_p = float(os.getenv("SPEC_SUITE_TOP_P", "0.8"))
            min_p = float(os.getenv("SPEC_SUITE_MIN_P", "0.0"))

        # Enable sampling when any stochastic knob is active.
        do_sample = temperature > 0.0 or top_p < 1.0 or top_k > 0 or min_p > 0.0

        stop_words = None
        bad_words = None
        stop_token_ids = None
        bad_token_ids = None
        if perf_mode:
            # In perf mode we want GPU tail to stay enabled and avoid
            # host-side stop-criteria paths. Force stop/bad word lists
            # and token-id stop-lists to be empty for these benchmarks.
            stop_words = []
            bad_words = []
            stop_token_ids = []
            bad_token_ids = []

        gen_configs = [
            GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                do_sample=do_sample,
                stop_words=stop_words,
                bad_words=bad_words,
                stop_token_ids=stop_token_ids,
                bad_token_ids=bad_token_ids,
            )
            for _ in range(batch_size)
        ]

        # Run benchmark
        results = self.run_benchmark(
            pipe,
            prompts,
            gen_configs,
            warmup_runs=warmup_runs,
            measurement_runs=measurement_runs,
            num_spec_tokens=num_spec_tokens if use_speculation else None,
        )

        # Add scenario info
        micro_steps_env = os.getenv("LMDEPLOY_EAGLE_MICRO_STEPS", "").strip()
        micro_steps = None
        if micro_steps_env:
            try:
                v = int(micro_steps_env)
                if v > 0:
                    micro_steps = v
            except ValueError:
                micro_steps = None

        results["scenario"] = {
            "name": scenario_name,
            "batch_size": batch_size,
            "context_length": context_length,
            "max_new_tokens": max_new_tokens,
            "use_speculation": use_speculation,
            "num_spec_tokens": num_spec_tokens if use_speculation else 0,
            "perf_mode": perf_mode,
            "micro_run": micro_steps is not None,
        }

        results["system"] = {
            "gpu_name": self.gpu_name,
            "gpu_memory_total_gb": self.gpu_memory_total,
            "timestamp": datetime.now().isoformat(),
            "turbomind_build_id": _tm.build_id(),
        }

        # Save results
        filename = f"{scenario_name.replace(' ', '_').lower()}.json"
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {filepath}")
        print(f"  Throughput: {results['throughput_tokens_per_sec']['mean']:.1f} tok/s")
        print(f"  Latency: {results['latency_ms_per_token']['mean']:.2f} ms/tok")
        print(f"  Memory: {results['memory_gb']['mean']:.2f} GB")
        if "eagle_speculation" in results:
            spec = results["eagle_speculation"]
            mean_rate = spec.get("mean_acceptance_rate", 0.0)
            mean_len = spec.get("mean_acceptance_length", 0.0)
            print(
                "  EAGLE: "
                f"mean_acceptance_rate={mean_rate:.3f}, "
                f"mean_acceptance_length={mean_len:.3f}"
            )

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark speculative decoding performance"
    )
    parser.add_argument("--model-path", required=True, help="Path to main model")
    parser.add_argument("--spec-model-path", help="Path to speculation model (EAGLE3)")
    parser.add_argument(
        "--output-dir", default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--scenario",
        choices=["all", "baseline", "single", "batch", "large-context", "stress"],
        default="all",
        help="Which scenario(s) to run",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Warmup runs per scenario",
    )
    parser.add_argument(
        "--measurement-runs",
        type=int,
        default=2,
        help="Measurement runs per scenario",
    )

    args = parser.parse_args()

    build_id = _tm.build_id()
    print(f"TURBOMIND build_id: {build_id}")

    runner = BenchmarkRunner(args.model_path, args.spec_model_path, args.output_dir)

    scenarios = []

    # Baseline scenarios (no speculation)
    if args.scenario in ["all", "baseline"]:
        scenarios.extend(
            [
                {
                    "scenario_name": "Baseline_Single_Context8K",
                    "batch_size": 1,
                    "context_length": 8192,
                    "max_new_tokens": 8192,
                    "use_speculation": False,
                },
                {
                    "scenario_name": "Speculative_Single_Context8K_2tokens",
                    "batch_size": 1,
                    "context_length": 8192,
                    "max_new_tokens": 8192,
                    "use_speculation": True,
                    "num_spec_tokens": 2,
                },
                {
                    "scenario_name": "Speculative_Single_Context8K_3tokens",
                    "batch_size": 1,
                    "context_length": 8192,
                    "max_new_tokens": 8192,
                    "use_speculation": True,
                    "num_spec_tokens": 3,
                },
                {
                    "scenario_name": "Speculative_Single_Context8K_4tokens",
                    "batch_size": 1,
                    "context_length": 8192,
                    "max_new_tokens": 8192,
                    "use_speculation": True,
                    "num_spec_tokens": 4,
                },
                {
                    "scenario_name": "Speculative_Single_Context8K_5tokens",
                    "batch_size": 1,
                    "context_length": 8192,
                    "max_new_tokens": 8192,
                    "use_speculation": True,
                    "num_spec_tokens": 5,
                },
            ]
        )

    # Single batch scenarios
    if args.scenario in ["all", "single"]:
        # Optional micro-run mode for single-context debugging so we can
        # exercise 32K single behaviour without always running full 32K
        # new tokens while tuning EAGLE3 kernels.
        micro_steps_env = os.getenv("LMDEPLOY_EAGLE_MICRO_STEPS", "").strip()
        micro_steps = None
        if micro_steps_env:
            try:
                v = int(micro_steps_env)
                if v > 0:
                    micro_steps = v
            except ValueError:
                micro_steps = None

        scenarios.extend(
            [
                {
                    "scenario_name": "Baseline_Single_Context32K",
                    "batch_size": 1,
                    "context_length": 32768,
                    "max_new_tokens": micro_steps if micro_steps is not None else 32768,
                    "use_speculation": False,
                },
                {
                    "scenario_name": "Speculative_Single_Context32K_2tokens",
                    "batch_size": 1,
                    "context_length": 32768,
                    "max_new_tokens": micro_steps if micro_steps is not None else 32768,
                    "use_speculation": True,
                    "num_spec_tokens": 2,
                },
                {
                    "scenario_name": "Speculative_Single_Context32K_3tokens",
                    "batch_size": 1,
                    "context_length": 32768,
                    "max_new_tokens": micro_steps if micro_steps is not None else 32768,
                    "use_speculation": True,
                    "num_spec_tokens": 3,
                },
                {
                    "scenario_name": "Speculative_Single_Context32K_4tokens",
                    "batch_size": 1,
                    "context_length": 32768,
                    "max_new_tokens": micro_steps if micro_steps is not None else 32768,
                    "use_speculation": True,
                    "num_spec_tokens": 4,
                },
                {
                    "scenario_name": "Speculative_Single_Context32K_5tokens",
                    "batch_size": 1,
                    "context_length": 32768,
                    "max_new_tokens": micro_steps if micro_steps is not None else 32768,
                    "use_speculation": True,
                    "num_spec_tokens": 5,
                },
            ]
        )

    # Batch scenarios
    if args.scenario in ["all", "batch"]:
        scenarios.extend(
            [
                {
                    "scenario_name": "Baseline_Batch8_Context8K",
                    "batch_size": 8,
                    "context_length": 8192,
                    "max_new_tokens": 8192,
                    "use_speculation": False,
                },
                {
                    "scenario_name": "Speculative_Batch8_Context8K_3tokens",
                    "batch_size": 8,
                    "context_length": 8192,
                    "max_new_tokens": 8192,
                    "use_speculation": True,
                    "num_spec_tokens": 3,
                },
                {
                    "scenario_name": "Speculative_Batch8_Context8K_5tokens",
                    "batch_size": 8,
                    "context_length": 8192,
                    "max_new_tokens": 8192,
                    "use_speculation": True,
                    "num_spec_tokens": 5,
                },
            ]
        )

    # Large context scenarios
    if args.scenario in ["all", "large-context"]:
        # Optional micro-run mode for large-context debugging so we can
        # exercise 16K batch4 behaviour without running full 16K steps.
        micro_steps_env = os.getenv("LMDEPLOY_EAGLE_MICRO_STEPS", "").strip()
        micro_steps = None
        if micro_steps_env:
            try:
                v = int(micro_steps_env)
                if v > 0:
                    micro_steps = v
            except ValueError:
                micro_steps = None

        scenarios.extend(
            [
                {
                    "scenario_name": "Baseline_Batch4_Context16K",
                    "batch_size": 4,
                    "context_length": 16384,
                    "max_new_tokens": micro_steps if micro_steps is not None else 16384,
                    "use_speculation": False,
                },
                {
                    "scenario_name": "Speculative_Batch4_Context16K_2tokens",
                    "batch_size": 4,
                    "context_length": 16384,
                    "max_new_tokens": micro_steps if micro_steps is not None else 16384,
                    "use_speculation": True,
                    "num_spec_tokens": 2,
                },
                {
                    "scenario_name": "Speculative_Batch4_Context16K_3tokens",
                    "batch_size": 4,
                    "context_length": 16384,
                    "max_new_tokens": micro_steps if micro_steps is not None else 16384,
                    "use_speculation": True,
                    "num_spec_tokens": 3,
                },
                {
                    "scenario_name": "Speculative_Batch4_Context16K_4tokens",
                    "batch_size": 4,
                    "context_length": 16384,
                    "max_new_tokens": micro_steps if micro_steps is not None else 16384,
                    "use_speculation": True,
                    "num_spec_tokens": 4,
                },
                {
                    "scenario_name": "Speculative_Batch4_Context16K_5tokens",
                    "batch_size": 4,
                    "context_length": 16384,
                    "max_new_tokens": micro_steps if micro_steps is not None else 16384,
                    "use_speculation": True,
                    "num_spec_tokens": 5,
                },
            ]
        )

    # Stress test scenarios
    if args.scenario in ["all", "stress"]:
        scenarios.extend(
            [
                {
                    "scenario_name": "Baseline_Batch16_LongGen",
                    "batch_size": 8,
                    "context_length": 16384,
                    "max_new_tokens": 16384,
                    "use_speculation": False,
                },
                {
                    "scenario_name": "Speculative_Batch16_LongGen_3tokens",
                    "batch_size": 8,
                    "context_length": 16384,
                    "max_new_tokens": 16384,
                    "use_speculation": True,
                    "num_spec_tokens": 3,
                },
            ]
        )

    # Run all scenarios
    all_results = []
    for scenario in scenarios:
        result = runner.run_test_scenario(
            **scenario,
            warmup_runs=args.warmup_runs,
            measurement_runs=args.measurement_runs,
        )
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for result in all_results:
        name = result["scenario"]["name"]
        throughput = result["throughput_tokens_per_sec"]["mean"]
        latency = result["latency_ms_per_token"]["mean"]
        print(f"{name:40s} {throughput:8.1f} tok/s  {latency:6.2f} ms/tok")


if __name__ == "__main__":
    main()
