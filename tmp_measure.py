from time import time
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.speculative_config import SpeculativeConfig, validate_eagle_runtime_config

model_path = '/workspace/aimo/models/gpt-oss-120b'
spec_model_path = '/workspace/aimo/models/gpt-oss-120b-eagle3'

engine_cfg = TurbomindEngineConfig(
    tp=1,
    session_len=32768,
    max_batch_size=1,
    quant_policy=8,
    cache_max_entry_count=0.75,
    enable_prefix_caching=False,
)

prompt = 'Explain the concept of transformers in deep learning in detail.'

# Baseline
base_pipe = pipeline(model_path, backend_config=engine_cfg)
base_cfg = GenerationConfig(max_new_tokens=8192, temperature=0.0, top_k=0, top_p=1.0)

print('=== BASELINE (no speculation) ===', flush=True)
start = time()
base_resps = base_pipe([prompt], gen_config=base_cfg)
elapsed = time() - start
n_tokens = len(base_resps[0].token_ids)
print('BASE tokens:', n_tokens, flush=True)
print('BASE elapsed_s:', elapsed, flush=True)
print('BASE tok_per_s:', n_tokens / elapsed if elapsed > 0 else 0.0, flush=True)

# Speculative
spec_cfg = SpeculativeConfig(
    method='eagle3',
    num_speculative_tokens=3,
    model=spec_model_path,
    eagle_debug=False,
    eagle_metrics_debug=False,
)
validate_eagle_runtime_config(engine_cfg, spec_cfg)

spec_pipe = pipeline(model_path, backend_config=engine_cfg, speculative_config=spec_cfg)

print('=== SPECULATIVE (EAGLE3) ===', flush=True)
start = time()
spec_resps = spec_pipe([prompt], gen_config=base_cfg)
elapsed = time() - start
n_tokens = len(spec_resps[0].token_ids)
print('SPEC tokens:', n_tokens, flush=True)
print('SPEC elapsed_s:', elapsed, flush=True)
print('SPEC tok_per_s:', n_tokens / elapsed if elapsed > 0 else 0.0, flush=True)
