# Copyright (c) OpenMMLab. All rights reserved.

import re

from ..config import RopeParam
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
        types = cfg.get('layer_types', [])
        sliding_window = cfg.get('sliding_window', 0)
        info = super().model_info()
        # Base GPT-OSS MoE / routing / activation wiring.
        info.update(attn_bias=int(cfg['attention_bias']),
                    mlp_bias=True,
                    expert_router_bias=True,
                    expert_num=cfg['num_local_experts'],
                    expert_inter_size=cfg['intermediate_size'],
                    experts_per_token=cfg['experts_per_token'],
                    norm_topk_prob=True,
                    inter_size=0,
                    window_size=[sliding_window if x == 'sliding_attention' else 0 for x in types],
                    attn_sink=True,
                    activation_type='gpt-oss')

        # If this GPT-OSS config has MLA fields (as produced by TransMLA),
        # propagate them into the TurboMind model config so MLA geometry is
        # available on the TM side. This mirrors the DeepSeek2 path.
        kv_lora_rank = cfg.get('kv_lora_rank', 0) or 0
        if kv_lora_rank > 0:
            qk_rope_dim = cfg.get('qk_rope_head_dim', 0) or 0
            qk_nope_dim = cfg.get('qk_nope_head_dim', 0) or 0
            size_per_head = qk_rope_dim + qk_nope_dim or info['size_per_head']
            # Update MLA-related fields and override size_per_head to match MLA.
            info.update(
                kv_lora_rank=kv_lora_rank,
                q_lora_rank=cfg.get('q_lora_rank') or 0,
                qk_rope_dim=qk_rope_dim,
                v_head_dim=cfg.get('v_head_dim', info['size_per_head']),
                size_per_head=size_per_head,
                mla_enabled=True,
                mla_cache_impl='naive',
            )
            # Ensure RoPE dim reflects the MLA rope sub-dimension.
            rope_param: RopeParam = info['rope_param']
            rope_param.dim = qk_rope_dim or rope_param.dim
            info.update(rope_param=rope_param)

        return info
