import os
import subprocess
from typing import Dict, List, Any

import fire
import yaml

# Try to import DriftEngineConfig
try:
    from lmdeploy import DriftEngineConfig
except ImportError:
    DriftEngineConfig = None
    print("Warning: DriftEngineConfig not available")


    def get_cmd(model_path, backend, engine_config, data_config):
    assert backend in ['turbomind', 'pytorch', 'drift']
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Handle config object vs dict
    if hasattr(engine_config, '__dict__'):
        engine_config = engine_config.__dict__.copy()
    else:
        engine_config = engine_config.copy()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = data_config.pop('dataset_path')
    data_config.pop('dataset_name')

    cmd = ['python3', f'{current_dir}/profile_pipeline_api.py', dataset_path, model_path, '--backend', backend]
    for key, value in engine_config.items():
        # profile_pipeline_api.py uses "--concurrency" to pass the "max_batch_size" value
        if key == 'max_batch_size':
            key = 'concurrency'
        # change the key like 'cache_max_entry_count' to 'cache-max-entry-count' to suit the optional
        # arguments in "python3 benchmark/profile_pipeline_api.py"
        key = key.replace('_', '-')
        cmd.append(f'--{key}')
        cmd.append(str(value))

    for key, value in data_config.items():
        # change the key like 'sharegpt_output_len' to 'sharegpt-output-len' to suit the optional
        # arguments in "python3 benchmark/profile_pipeline_api.py"
        key = key.replace('_', '-')
        cmd.append(f'--{key}')
        cmd.append(str(value))
    return cmd


def benchmark(model_path, backend, engine_config, data_config):
    """Benchmark the performance with the given configuration.

    Args:
        model_path: Path to the model.
    :param backend: Backend to use.
    :param engine_config: Configuration for the inference engine.
    :param data_config: Configuration for the data.
    """
    model_name = os.path.basename(model_path)
    bs = engine_config['max_batch_size']
    cach_ratio = engine_config.get('cache_max_entry_count', 0.8)
    tp = engine_config.get('tp', 1)
    output_file = f'benchmark_pipeline_{model_name}_{backend}_bs{bs}_tp{tp}_cache{cach_ratio}.csv'
    try:
        if isinstance(data_config, Dict):
            data_config = [data_config]
        assert isinstance(data_config, List) and all(isinstance(d, Dict) for d in data_config)
        for _data_config in data_config:
            _data_config['csv'] = output_file
            cmd = get_cmd(model_path, backend, engine_config, _data_config)
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
    except Exception as e:
        print(f'exception happened, {e}')


def main(model_path=None, backend=None, config_path=None, config_name=None, scenario=None, output_dir=None, warmup_runs=None, measurement_runs=None, csv=None):
    # Handle case where config_path is provided (from run_engine_suite.sh)
    if config_path and os.path.exists(config_path):
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract engine and data configurations
        if config_name:
            config = config.get(config_name, {})
        else:
            # Fallback to first engine config if no config_name specified
            engine_configs = config.get('engine', [{}])
            config = engine_configs[0] if engine_configs else {}
        
        engine_config = config.get('engine_config', {})
        data_config = config.get('data_config', {})
        
        # Override with command line arguments
        if model_path:
            if backend == 'drift' and hasattr(engine_config, 'model_path'):
                engine_config.model_path = model_path
            else:
                engine_config['model_path'] = model_path
        
        if warmup_runs:
            data_config['warmup_runs'] = warmup_runs
        
        if measurement_runs:
            data_config['measurement_runs'] = measurement_runs
        
        if output_dir:
            data_config['output_dir'] = output_dir
        
        # Handle csv parameter
        if csv:
            data_config['csv'] = csv
        
        print(f"Running benchmark: backend={backend}, config_name={config_name}, engine_config={engine_config}, data_config={data_config}")
        benchmark(model_path or engine_config.get('model_path', ''), backend, engine_config, data_config)
        return
    
    # Fallback to simple configuration for direct testing
    engine_config = {}
    data_config = {}
    
    # Create basic engine config based on backend
    if backend == 'turbomind':
        engine_config = {
            'max_batch_size': 128,
            'cache_max_entry_count': 0.8,
            'session_len': 8192,
            'tp': 1,
            'dp': 1,
        }
    elif backend == 'drift':
        # Use conservative DriftEngine config
        try:
            from lmdeploy import DriftEngineConfig
            engine_config = DriftEngineConfig.conservative_baseline()
            engine_config.max_batch_size = 128
        except ImportError:
            print("Warning: DriftEngineConfig not available, using basic config")
            engine_config = {
                'max_batch_size': 128,
                'cache_max_entry_count': 0.8,
                'session_len': 8192,
                'tp': 1,
                'dp': 1,
            }
    
    # Set up data config for synthetic data since we don't have the real dataset
    data_config = {
        'dataset_name': 'random',
        'random_input_len': 256,  # Smaller for testing
        'random_output_len': 64,   # Smaller for testing
        'num_prompts': 50,  # Reduced for testing
        'csv': csv or 'benchmark_results.csv'
    }
    
    # Override with command line arguments
    if model_path:
        engine_config['model_path'] = model_path
    
    # Run benchmark
    print(f"Running benchmark: backend={backend}, engine_config={engine_config}, data_config={data_config}")
    benchmark(model_path or engine_config.get('model_path', ''), backend, engine_config, data_config)

def get_default_config(config_name: str, backend: str) -> Dict[str, Any]:
    """Get default configuration based on config name and backend."""
    
    # Default data configuration
    data_config = {
        'dataset_path': '/nvme1/shared/ShareGPT_V3_unfiltered_cleaned_split.json',
        'dataset_name': 'sharegpt',
        'num_prompts': 1000,
        'sharegpt_output_len': 2048,
    }
    
    # Default engine configuration based on config_name
    if backend == 'turbomind':
        engine_config = {
            'max_batch_size': 128,
            'cache_max_entry_count': 0.8,
            'max_prefill_token_num': 4096,
            'tp': 1,
            'dp': 1,
            'session_len': 8192,
        }
    elif backend == 'drift':
        if config_name == 'drift_baseline':
            engine_config = {
                'prefer_high_throughput': False,
                'enable_cuda_graphs': False,
                'decode_microbatch_size': None,
                'prefill_microbatch_size': None,
                'target_latency_ms_p50': 100,
                'target_latency_ms_p95': 200,
                'max_queued_requests': 4096,
                'max_batch_size': 128,
                'cache_max_entry_count': 0.8,
                'max_prefill_token_num': 4096,
                'tp': 1,
                'dp': 1,
                'session_len': 8192,
            }
        elif config_name == 'drift_optimized':
            engine_config = {
                'prefer_high_throughput': True,
                'enable_cuda_graphs': True,
                'decode_microbatch_size': 64,
                'prefill_microbatch_size': 128,
                'target_latency_ms_p50': 50,
                'target_latency_ms_p95': 150,
                'max_queued_requests': 8192,
                'max_batch_size': 128,
                'cache_max_entry_count': 0.8,
                'max_prefill_token_num': 4096,
                'tp': 1,
                'dp': 1,
                'session_len': 8192,
            }
        else:
            engine_config = {}
    else:
        engine_config = {}
    
    return {
        'engine_config': engine_config,
        'data_config': data_config
    }


if __name__ == '__main__':
    fire.Fire(main)
