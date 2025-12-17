#!/usr/bin/env python3
"""
DriftEngine baseline benchmark runner.
Speculative decoding is intentionally disabled; all scenarios run the drift backend.
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from lmdeploy import DriftEngineConfig, GenerationConfig, pipeline as lm_pipeline


def resolve_max_new_tokens(default_len: int, cli_override: Optional[int] = None) -> int:
    """Resolve max_new_tokens using CLI override or LMDEPLOY_EAGLE_MICRO_STEPS."""
    if cli_override is not None and cli_override > 0:
        return cli_override
    env_value = os.getenv("LMDEPLOY_EAGLE_MICRO_STEPS", "").strip()
    if env_value:
        try:
            env_tokens = int(env_value)
            if env_tokens > 0:
                return env_tokens
        except ValueError:
            pass
    return default_len


class BenchmarkRunner:
    """Run DriftEngine baseline benchmarks with simple metrics."""

    def __init__(self, model_path: str, output_dir: str = "results"):
        if os.getenv("DRIFT_USE_STUB_EXECUTOR"):
            raise RuntimeError("DRIFT_USE_STUB_EXECUTOR is not allowed when running DriftEngine benchmarks.")

        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            self.gpu_name = "No GPU"
            self.gpu_memory_total = 0

    def create_pipeline(self, max_batch_size: int = 8, session_len: int = 32768):
        """Create a drift pipeline with non-spec config."""
        cache_max_entry = 0.75
        cache_env = os.getenv("TM_CACHE_MAX_ENTRY_COUNT")
        if cache_env:
            try:
                v = float(cache_env)
                if 0.0 < v <= 1.0:
                    cache_max_entry = v
            except ValueError:
                pass

        cfg = DriftEngineConfig(
            model_path=self.model_path,
            tp=1,
            session_len=session_len,
            max_batch_size=max_batch_size,
        )
        # Force v1 non-spec behaviour
        if hasattr(cfg, "enable_speculative_decoding"):
            cfg.enable_speculative_decoding = False
        if hasattr(cfg, "enable_cuda_graphs"):
            cfg.enable_cuda_graphs = False
        if hasattr(cfg, "enable_prefix_caching"):
            cfg.enable_prefix_caching = False
        if hasattr(cfg, "kv"):
            if getattr(cfg.kv, "kv_page_size", None) is None:
                cfg.kv.kv_page_size = 128
            cfg.kv.kv_capacity_bytes = getattr(cfg.kv, "kv_capacity_bytes", None) or 0
            if hasattr(cfg.kv, "prefix_cache_enabled"):
                cfg.kv.prefix_cache_enabled = False
        if hasattr(cfg, "scheduler"):
            cfg.scheduler.enable_chunked_prefill = getattr(cfg.scheduler, "enable_chunked_prefill", True)
            cfg.scheduler.prefer_decode_over_prefill = getattr(cfg.scheduler, "prefer_decode_over_prefill", True)
        if hasattr(cfg, "cache_max_entry_count"):
            cfg.cache_max_entry_count = cache_max_entry

        return lm_pipeline(self.model_path, backend_config=cfg)

    def measure_memory(self) -> Dict[str, float]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "utilization_pct": (allocated / self.gpu_memory_total) * 100 if self.gpu_memory_total else 0,
            }
        return {"allocated_gb": 0, "reserved_gb": 0, "utilization_pct": 0}

    def collect_drift_metrics(self, pipe) -> Optional[Dict[str, Any]]:
        """Return the latest DriftMetrics snapshot if available."""
        engine = getattr(pipe, "engine", None)
        if engine is None:
            return None
        if hasattr(engine, "_ensure_drift_engine"):
            try:
                engine._ensure_drift_engine()
            except Exception:
                return None
        drift = getattr(engine, "_drift_engine", None)
        if drift is None or not hasattr(drift, "get_metrics"):
            return None
        try:
            metrics = drift.get_metrics()
        except Exception:
            return None
        return metrics if isinstance(metrics, dict) else None

    def ensure_live_backend(self, pipe) -> None:
        engine = getattr(pipe, "engine", None)
        if engine is None:
            raise RuntimeError("Pipeline missing engine handle; cannot verify DriftEngine state.")
        drift_engine = getattr(engine, "_drift_engine", None)
        if drift_engine is None:
            raise RuntimeError("Drift pipeline did not expose a live DriftEngine backend.")

    def run_benchmark(
        self,
        pipe,
        prompts: List[str],
        gen_configs: List[GenerationConfig],
        warmup_runs: int = 2,
        measurement_runs: int = 2,
    ) -> Dict[str, Any]:
        try:
            warmup_runs = int(os.getenv("SPEC_SUITE_WARMUP_RUNS", warmup_runs))
            measurement_runs = int(os.getenv("SPEC_SUITE_MEASUREMENT_RUNS", measurement_runs))
        except ValueError:
            pass

        self.ensure_live_backend(pipe)
 
        print(f"  Warmup ({warmup_runs} runs)...")

        for _ in range(warmup_runs):
            _ = pipe(prompts, gen_config=gen_configs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  Measurement ({measurement_runs} runs)...")
        latencies: List[float] = []
        throughputs: List[float] = []
        memory_usages: List[Dict[str, float]] = []

        for run_idx in range(measurement_runs):
            start_time = time.time()
            responses = pipe(prompts, gen_config=gen_configs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            if not responses:
                raise RuntimeError("Benchmark run produced no responses; DriftEngine is not executing.")

            total_tokens = sum(len(getattr(r, "token_ids", [])) for r in responses)
            if total_tokens == 0:
                raise RuntimeError("Benchmark run produced zero tokens; aborting.")
            if any(len(getattr(r, "token_ids", [])) == 0 for r in responses):
                raise RuntimeError("Benchmark run returned a response without generated tokens.")
            if all(all(token == 0 for token in getattr(r, "token_ids", [])) for r in responses):
                raise RuntimeError("Benchmark run produced only zero tokens; DriftEngine decode path is not live.")

            total_time = end_time - start_time

            latency_per_token = (total_time / total_tokens) * 1000 if total_tokens else 0.0
            throughput = total_tokens / total_time if total_time > 0 else 0.0

            latencies.append(latency_per_token)
            throughputs.append(throughput)
            mem_after = self.measure_memory()
            memory_usages.append(mem_after)

            print(
                f"    Run {run_idx + 1}: {throughput:.1f} tok/s, "
                f"{latency_per_token:.2f} ms/tok, "
                f"{mem_after['allocated_gb']:.2f} GB"
            )

        results = {
            "latency_ms_per_token": {
                "mean": sum(latencies) / len(latencies) if latencies else 0.0,
                "min": min(latencies) if latencies else 0.0,
                "max": max(latencies) if latencies else 0.0,
                "values": latencies,
            },
            "throughput_tokens_per_sec": {
                "mean": sum(throughputs) / len(throughputs) if throughputs else 0.0,
                "min": min(throughputs) if throughputs else 0.0,
                "max": max(throughputs) if throughputs else 0.0,
                "values": throughputs,
            },
            "memory_gb": {
                "mean": sum(m["allocated_gb"] for m in memory_usages) / len(memory_usages) if memory_usages else 0.0,
                "max": max(m["allocated_gb"] for m in memory_usages) if memory_usages else 0.0,
                "utilization_pct": sum(m["utilization_pct"] for m in memory_usages) / len(memory_usages) if memory_usages else 0.0,
            },
            "total_runs": measurement_runs,
        }

        drift_metrics = self.collect_drift_metrics(pipe)
        if drift_metrics:
            results["drift_metrics"] = drift_metrics

        return results

    def run_test_scenario(
        self,
        scenario_name: str,
        batch_size: int,
        context_length: int,
        max_new_tokens: int,
        warmup_runs: int = 2,
        measurement_runs: int = 2,
    ) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print(f"  Batch size: {batch_size}")
        print(f"  Context length: {context_length}")
        print(f"  Max new tokens: {max_new_tokens}")
        print(f"{'='*60}")

        pipe = self.create_pipeline(max_batch_size=batch_size, session_len=context_length)

        prompts = ["Explain the concept of quantum computing in detail." * (context_length // 50)] * batch_size

        temperature = float(os.getenv("SPEC_SUITE_TEMPERATURE", "0.0"))
        top_k = int(os.getenv("SPEC_SUITE_TOP_K", "20"))
        top_p = float(os.getenv("SPEC_SUITE_TOP_P", "0.8"))
        min_p = float(os.getenv("SPEC_SUITE_MIN_P", "0.0"))
        do_sample = temperature > 0.0 or top_p < 1.0 or top_k > 0 or min_p > 0.0

        gen_configs = [
            GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                do_sample=do_sample,
            )
            for _ in range(batch_size)
        ]

        results = self.run_benchmark(
            pipe,
            prompts,
            gen_configs,
            warmup_runs=warmup_runs,
            measurement_runs=measurement_runs,
        )

        results["scenario"] = {
            "name": scenario_name,
            "batch_size": batch_size,
            "context_length": context_length,
            "max_new_tokens": max_new_tokens,
            "use_speculation": False,
            "num_spec_tokens": 0,
        }

        results["system"] = {
            "gpu_name": self.gpu_name,
            "gpu_memory_total_gb": self.gpu_memory_total,
            "timestamp": datetime.now().isoformat(),
            "engine": "drift",
        }

        filename = f"{scenario_name.replace(' ', '_').lower()}.json"
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {filepath}")
        print(f"  Throughput: {results['throughput_tokens_per_sec']['mean']:.1f} tok/s")
        print(f"  Latency: {results['latency_ms_per_token']['mean']:.2f} ms/tok")
        print(f"  Memory: {results['memory_gb']['mean']:.2f} GB")
        return results


def main():
    parser = argparse.ArgumentParser(description="DriftEngine baseline benchmark")
    parser.add_argument("--model-path", required=True, help="Path to main model")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument(
        "--scenario",
        choices=["all", "baseline", "single", "batch", "large-context", "stress"],
        default="all",
        help="Which scenario(s) to run",
    )
    parser.add_argument(
        "--baseline-contexts",
        type=int,
        nargs="+",
        default=[8192, 16384, 32768],
        help="Context lengths (tokens) for baseline matrix scenarios",
    )
    parser.add_argument(
        "--baseline-batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8],
        help="Batch sizes to sweep for baseline matrix scenarios",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override max_new_tokens for all scenarios (defaults to context length)",
    )
    parser.add_argument("--warmup-runs", type=int, default=2, help="Warmup runs per scenario")
    parser.add_argument("--measurement-runs", type=int, default=2, help="Measurement runs per scenario")
    args = parser.parse_args()

    runner = BenchmarkRunner(args.model_path, args.output_dir)
    scenarios = []

    def build_baseline_matrix() -> List[Dict[str, Any]]:
        built: List[Dict[str, Any]] = []
        contexts = sorted({int(c) for c in args.baseline_contexts})
        batches = sorted({int(b) for b in args.baseline_batch_sizes})
        for ctx in contexts:
            ctx_label = f"{ctx // 1024}K" if ctx % 1024 == 0 else str(ctx)
            max_new = resolve_max_new_tokens(ctx, args.max_new_tokens)
            for batch in batches:
                built.append(
                    {
                        "scenario_name": f"Baseline_Batch{batch}_Context{ctx_label}",
                        "batch_size": batch,
                        "context_length": ctx,
                        "max_new_tokens": max_new,
                    }
                )
        return built

    if args.scenario in ["all", "baseline"]:
        scenarios.extend(build_baseline_matrix())

    if args.scenario == "single":
        max_new_single = resolve_max_new_tokens(32768, args.max_new_tokens)
        scenarios.append(
            {
                "scenario_name": "Baseline_Single_Context32K",
                "batch_size": 1,
                "context_length": 32768,
                "max_new_tokens": max_new_single,
            }
        )

    if args.scenario == "batch":
        max_new_batch = resolve_max_new_tokens(8192, args.max_new_tokens)
        scenarios.append(
            {
                "scenario_name": "Baseline_Batch8_Context8K",
                "batch_size": 8,
                "context_length": 8192,
                "max_new_tokens": max_new_batch,
            }
        )

    if args.scenario == "large-context":
        max_new_large = resolve_max_new_tokens(16384, args.max_new_tokens)
        scenarios.append(
            {
                "scenario_name": "Baseline_Batch4_Context16K",
                "batch_size": 4,
                "context_length": 16384,
                "max_new_tokens": max_new_large,
            }
        )

    if args.scenario == "stress":
        max_new_stress = resolve_max_new_tokens(16384, args.max_new_tokens)
        scenarios.append(
            {
                "scenario_name": "Baseline_Batch8_LongGen",
                "batch_size": 8,
                "context_length": 16384,
                "max_new_tokens": max_new_stress,
            }
        )

    all_results = []
    for scenario in scenarios:
        result = runner.run_test_scenario(
            **scenario,
            warmup_runs=args.warmup_runs,
            measurement_runs=args.measurement_runs,
        )
        all_results.append(result)

    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    summary_rows: List[Dict[str, Any]] = []
    for result in all_results:
        name = result["scenario"]["name"]
        throughput = result["throughput_tokens_per_sec"]["mean"]
        latency = result["latency_ms_per_token"]["mean"]
        print(f"{name:40s} {throughput:8.1f} tok/s  {latency:6.2f} ms/tok")

        row: Dict[str, Any] = {
            "scenario": name,
            "batch_size": result["scenario"]["batch_size"],
            "context_length": result["scenario"]["context_length"],
            "max_new_tokens": result["scenario"]["max_new_tokens"],
            "throughput_tokens_per_sec": throughput,
            "latency_ms_per_token": latency,
            "memory_gb": result["memory_gb"]["mean"],
        }
        metrics = result.get("drift_metrics") or {}
        metric_fields = [
            "ema_tokens_per_second",
            "ema_p50_latency_ms",
            "ema_p95_latency_ms",
            "step_prefill_tokens",
            "step_decode_tokens",
            "queued_prefill",
            "queued_decode",
            "active_requests",
            "kv_total_pages",
            "kv_used_pages",
            "kv_free_pages",
            "kv_blocked",
            "kv_rejected",
            "prefix_hits",
            "prefix_misses",
            "prefix_evictions",
            "prefix_bytes_evicted",
        ]
        for field in metric_fields:
            row[field] = metrics.get(field)
        summary_rows.append(row)

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        summary_path = runner.output_dir / "benchmark_summary.csv"
        with summary_path.open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary CSV written to {summary_path}")


if __name__ == "__main__":
    main()
