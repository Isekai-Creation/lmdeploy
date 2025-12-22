# Copyright (c) OpenMMLab. All rights reserved.

import re

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


def map_experts(str):
    s = re.sub(r'(experts.*proj)$', r'\1.weight', str)
    s = re.sub(r'(experts.*proj)_bias$', r'\1.bias', s)
    s = re.sub(r'(experts.*proj)_blocks$', r'\1.blocks', s)
    s = re.sub(r'(experts.*proj)_scales$', r'\1.scales', s)
    return s


class GptOssReader(LlamaReader):

    mappings = [map_experts]

    def moe_ffn_expert(self, e=None, i=None, kind=None):
        if not kind:
            return self.filter(r'experts')
        result = []
        for key in ['gate_up', 'down']:
            name = f'{self.attn_layer_prefix}.{i}.mlp.experts.{key}_proj.{kind}'
            tensor = self.params.get(name)[e]
            if kind == 'weight':  # experts in BF16 models are in M-major
                tensor = tensor.cuda().t()
            if key == 'gate_up':
                gate, up = tensor[::2], tensor[1::2]
                result.append(self.transform(gate, kind))
                result.append(self.transform(up, kind))
            else:
                result.append(self.transform(tensor, kind))
        return (result[0], result[2], result[1])

    def moe_ffn_gate(self, i, kind):
        return self.transform(self.params.get(f'{self.attn_layer_prefix}.{i}.mlp.router.{kind}'), kind)

    def attn_sinks(self, i):
        return self.params.get(f'{self.attn_layer_prefix}.{i}.self_attn.sinks')


@INPUT_MODELS.register_module(name='gpt-oss')
class GptOssModel(LlamaModel):

    Reader = GptOssReader

    def model_info(self):
        cfg = self.model_config
        # Handle missing layer_types - default to alternating sliding/full pattern
        num_layers = cfg.get('num_hidden_layers', 24)
        types = cfg.get('layer_types', None)
        if types is None:
            # Generate default alternating pattern: sliding_attention, full_attention, ...
            types = ['sliding_attention' if i % 2 == 0 else 'full_attention' for i in range(num_layers)]
        sliding_window = cfg.get('sliding_window', 128)  # Default sliding window size
        info = super().model_info()
        info.update(attn_bias=int(cfg.get('attention_bias', True)),
                    mlp_bias=True,
                    expert_router_bias=True,
                    expert_num=cfg.get('num_local_experts', 32),
                    expert_inter_size=cfg.get('intermediate_size', 2880),
                    experts_per_token=cfg.get('experts_per_token', 4),
                    norm_topk_prob=True,
                    inter_size=0,
                    window_size=[sliding_window if x == 'sliding_attention' else 0 for x in types],
                    attn_sink=True,
                    activation_type='gpt-oss')
        return info

