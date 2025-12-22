import os
import subprocess
import json
import time
from pathlib import Path

# Configuration
VERSION_DIR = "/workspace/aimo/VERSIONS/lmdeploy_8da9555d"
MODEL_PATH = "/workspace/aimo/models/gpt-oss-120b"
LOG_DIR = Path("/workspace/aimo/logs/eagle3_final_bench")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Test configs
EAGLE3_MODEL = "/workspace/aimo/models/gpt-oss-120b-eagle3"
THROUGHPUT_MODEL = "/workspace/aimo/models/gpt-oss-120b-Eagle3-throughput"

# SPEC_SWEEP = [3, 4, 5, 6, 7]
SPEC_SWEEP = [3]

CASES = [
    # (name, session_len, batch_size, input_len)
    ("SINGLE", 8192, 1, 0),
    ("LARGE", 65536, 1, 0),
    ("HUGE", 65536, 8, 0),
    ("VLLM_COMP", 32768, 8, 8192)
]

def run_bench(spec_model, num_spec, session_len, batch_size, input_len, name, cache_fraction=0.85):
    print(f"\n>>> Running {name}: model={spec_model}, spec_tokens={num_spec}, session={session_len}, bs={batch_size}, cache={cache_fraction}")
    
    max_new_tokens = session_len - input_len
    if max_new_tokens <= 0: max_new_tokens = 1
    
    env = os.environ.copy()
    env["LMDEPLOY_EAGLE_PERF_MODE"] = "1"
    env["TM_CACHE_MAX_ENTRY_COUNT"] = str(cache_fraction)
    
    # Create a temporary scenario script for exact prompt control
    scenario_script = LOG_DIR / f"run_{name}_{num_spec}_{cache_fraction}.py"
    with open(scenario_script, "w") as f:
        f.write(f"""
import os
import sys
import torch
sys.path.append("{VERSION_DIR}")
from benchmark_speculative import BenchmarkRunner

class CustomRunner(BenchmarkRunner):
    def run_test_scenario_custom(self, scenario_name, batch_size, context_length, max_new_tokens, use_speculation, num_spec_tokens, input_len):
        print(f"Custom Scenario: {{scenario_name}}")
        
        pipe = self.create_pipeline(
            use_speculation=use_speculation,
            num_spec_tokens=num_spec_tokens,
            max_batch_size=batch_size,
            session_len=context_length,
        )
        
        # Approximate tokens based on character length. 
        # For Llama tokenizer, approx 4 chars per token.
        # We want input_len tokens.
        if input_len > 0:
            prompt_text = "token " * input_len
        else:
            prompt_text = "Explain the concept of quantum computing."
            
        prompts = [prompt_text] * batch_size
        
        from lmdeploy import GenerationConfig
        gen_configs = [
            GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_k=0,
                top_p=1.0,
                do_sample=False,
                stop_words=[],
                bad_words=[],
                stop_token_ids=[],
                bad_token_ids=[],
            )
            for _ in range(batch_size)
        ]
        
        results = self.run_benchmark(
            pipe,
            prompts,
            gen_configs,
            warmup_runs=1,
            measurement_runs=1,
            num_spec_tokens=num_spec_tokens
        )
        
        # Save results
        filename = f"{{scenario_name}}.json"
        filepath = self.output_dir / filename
        import json
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {{filepath}}")

runner = CustomRunner("{MODEL_PATH}", "{spec_model}", "{LOG_DIR}")
runner.run_test_scenario_custom(
    scenario_name="{name}_{num_spec}",
    batch_size={batch_size},
    context_length={session_len},
    max_new_tokens={max_new_tokens},
    use_speculation=True,
    num_spec_tokens={num_spec},
    input_len={input_len}
)
""")
    
    python_bin = "/workspace/aimo/miniconda/envs/lmdeploy/bin/python"
    try:
        subprocess.run([python_bin, str(scenario_script)], env=env, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark {name} with {cache_fraction}: {e}")
        return False

# Execution
results_summary = []

# Sweep 1: Eagle3 model
for n in SPEC_SWEEP:
    for name, session, bs, in_len in CASES:
        full_name = f"EAGLE3_{name}_{n}"
        success = run_bench(EAGLE3_MODEL, n, session, bs, in_len, full_name, 0.85)
        if not success:
            print(f"Retrying {full_name} with 0.75 cache...")
            run_bench(EAGLE3_MODEL, n, session, bs, in_len, full_name, 0.75)

# Sweep 2: Throughput model
for name, session, bs, in_len in CASES:
    full_name = f"THROUGHPUT_{name}_1"
    success = run_bench(THROUGHPUT_MODEL, 1, session, bs, in_len, full_name, 0.85)
    if not success:
        print(f"Retrying {full_name} with 0.75 cache...")
        run_bench(THROUGHPUT_MODEL, 1, session, bs, in_len, full_name, 0.75)

print("\nAll benchmarks completed.")
