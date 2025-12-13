import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def dump_past(prefix: str, past):
    if past is None:
        print(f'{prefix}: past_key_values=None')
        return
    print(f'{prefix}: past_key_values: {len(past)} layers')
    for li, layer_past in enumerate(past):
        print(f'  Layer {li:02d}: type={type(layer_past)}')
        if isinstance(layer_past, (tuple, list)):
            print(f'    n_parts={len(layer_past)}')
            for pi, part in enumerate(layer_past):
                if torch.is_tensor(part):
                    print(
                        f'    part{pi}: Tensor shape={tuple(part.shape)} '
                        f'dtype={part.dtype} device={part.device} stride={part.stride()}'
                    )
                else:
                    has_shape = hasattr(part, 'shape')
                    desc = part if not has_shape else '<tensor-like>'
                    print(f'    part{pi}: {type(part)} value={desc}')
        else:
            print('    (non-tuple past entry)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path',
                        type=str,
                        default='models/gpt-oss-120b-mla-dsa',
                        help='Path to LlamaMLAForCausalLM checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    args = parser.parse_args()

    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    print('Config arch:', cfg.__class__.__name__)
    print('architectures:', getattr(cfg, 'architectures', None))
    print('kv_lora_rank:', getattr(cfg, 'kv_lora_rank', None))
    print('qk_rope_head_dim:', getattr(cfg, 'qk_rope_head_dim', None))
    print('qk_nope_head_dim:', getattr(cfg, 'qk_nope_head_dim', None))
    print('v_head_dim:', getattr(cfg, 'v_head_dim', None))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map={'': args.device})
    model.eval()

    prompt = 'Hello, this is an MLA past_key_values inspection.'
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    with torch.no_grad():
        out1 = model(**inputs, use_cache=True, return_dict=True)

    past1 = out1.past_key_values if hasattr(out1, 'past_key_values') else out1['past_key_values']
    print('past_key_values type:', type(past1))
    # best-effort Cache introspection
    try:
        has_dict = hasattr(past1, '__dict__')
        print('past_key_values has __dict__:', has_dict)
        if has_dict:
            print('past_key_values.__dict__ keys:', list(past1.__dict__.keys()))
        attrs = [a for a in dir(past1) if not a.startswith('_')]
        interesting = [a for a in attrs if any(k in a.lower() for k in ('cache', 'key', 'value', 'layer', 'kv'))]
        print('past_key_values attrs (filtered):', interesting)
        for name in interesting:
            try:
                val = getattr(past1, name)
            except Exception:
                continue
            print(f'  attr {name}: type={type(val)}')
            if isinstance(val, (list, tuple)):
                for i, item in enumerate(val[:4]):
                    if torch.is_tensor(item):
                        print(f'    [{i}] tensor shape={tuple(item.shape)} dtype={item.dtype} device={item.device}')
                    else:
                        print(f'    [{i}] type={type(item)}')
    except Exception as e:
        print('Cache introspection error:', repr(e))
    dump_past('=== step1 ===', past1)

    # Second step: append one token and reuse past
    next_input_ids = torch.full((inputs['input_ids'].shape[0], 1),
                                tokenizer.eos_token_id or tokenizer.pad_token_id,
                                dtype=inputs['input_ids'].dtype,
                                device=inputs['input_ids'].device)
    inputs2 = dict(input_ids=next_input_ids, use_cache=True, return_dict=True, past_key_values=past1)

    with torch.no_grad():
        out2 = model(**inputs2)

    past2 = out2.past_key_values if hasattr(out2, 'past_key_values') else out2['past_key_values']
    dump_past('=== step2 ===', past2)


if __name__ == '__main__':
    main()
