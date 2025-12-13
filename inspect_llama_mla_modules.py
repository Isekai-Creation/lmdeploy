import argparse
import glob
import os

import torch
from safetensors.torch import safe_open
from transformers import AutoConfig, AutoModelForCausalLM


def _inspect_attn(layer, layer_idx: int):
    print(f'Layer{layer_idx}.self_attn type:', type(layer))
    for name in ['q_a_proj', 'q_b_proj', 'q_proj', 'kv_a_proj_with_mqa', 'kv_b_proj', 'o_proj']:
        mod = getattr(layer, name, None)
        print(f'  {name}:', type(mod))
        if isinstance(mod, torch.nn.Module):
            w = getattr(mod, 'weight', None)
            if w is not None:
                print(f'    weight.shape={tuple(w.shape)}')

    for name in ['q_a_layernorm', 'kv_a_layernorm']:
        ln = getattr(layer, name, None)
        print(f'  {name}:', type(ln))
        if isinstance(ln, torch.nn.Module):
            w = getattr(ln, 'weight', None)
            if w is not None:
                print(f'    weight.shape={tuple(w.shape)}')


def _inspect_mlp(layer, layer_idx: int):
    mlp = getattr(layer, 'mlp', None)
    print(f'Layer{layer_idx}.mlp type:', type(mlp))
    if mlp is None:
        return
    attrs = [a for a in dir(mlp) if not a.startswith('_')]
    interesting = [
        a for a in attrs
        if any(k in a for k in ['gate_proj', 'up_proj', 'down_proj', 'experts', 'router', 'w1', 'w2', 'w3'])
    ]
    print('  mlp attrs (filtered):', interesting)
    for name in ['gate_proj', 'up_proj', 'down_proj', 'w1', 'w2', 'w3']:
        mod = getattr(mlp, name, None)
        if isinstance(mod, torch.nn.Module):
            w = getattr(mod, 'weight', None)
            if w is not None:
                print(f'  {name}.weight.shape={tuple(w.shape)}')
    if hasattr(mlp, 'experts'):
        try:
            experts = mlp.experts
            print('  experts:', type(experts), 'len=', len(experts))
        except Exception:
            print('  experts: present but not iterable')
    if hasattr(mlp, 'router'):
        router = mlp.router
        print('  router type:', type(router))
        if isinstance(router, torch.nn.Module):
            for attr in ['weight', 'bias']:
                w = getattr(router, attr, None)
                if w is not None:
                    print(f'  router.{attr}.shape={tuple(w.shape)}')


def _enumerate_checkpoint_keys(model_path: str):
    print('--- Checking checkpoint keys on disk ---')
    pattern = os.path.join(model_path, '*.safetensors')
    files = sorted(glob.glob(pattern))
    if not files:
        print('No .safetensors files found under', model_path)
        return
    want_substrings = [
        'mlp.gate_proj.weight',
        'mlp.up_proj.weight',
        'mlp.down_proj.weight',
        'mlp.experts',
        'mlp.router',
    ]
    found_counts = {k: 0 for k in want_substrings}
    for f in files:
        print('Scanning file:', os.path.basename(f))
        with safe_open(f, framework='pt', device='cpu') as sf:
            for key in sf.keys():
                for pat in want_substrings:
                    if pat in key:
                        found_counts[pat] += 1
                        print('  matched:', key)
    print('MLP-related key counts across shards:')
    for pat, cnt in found_counts.items():
        print(f'  {pat}: {cnt}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path',
                        type=str,
                        default='models/gpt-oss-120b-mla-dsa',
                        help='Path to LlamaMLAForCausalLM checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    args = parser.parse_args()

    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    print('Config class:', cfg.__class__.__name__)
    print('kv_lora_rank:', getattr(cfg, 'kv_lora_rank', None))
    print('qk_rope_head_dim:', getattr(cfg, 'qk_rope_head_dim', None))
    print('qk_nope_head_dim:', getattr(cfg, 'qk_nope_head_dim', None))
    print('v_head_dim:', getattr(cfg, 'v_head_dim', None))

    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map={'': args.device})
    model.eval()

    layers = list(model.model.layers)
    print('num_hidden_layers:', len(layers))

    for idx in [0, len(layers) // 2]:
        print(f'=== Inspect layer {idx} ===')
        layer = layers[idx]
        _inspect_attn(layer.self_attn, idx)
        _inspect_mlp(layer, idx)

    _enumerate_checkpoint_keys(args.model_path)


if __name__ == '__main__':
    main()
