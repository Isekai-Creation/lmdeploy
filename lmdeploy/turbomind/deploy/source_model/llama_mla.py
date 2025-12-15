import os

import torch
from transformers import AutoModelForCausalLM

from lmdeploy.archs import get_model_arch

from ..config import RopeParam
from ..loader import create_loader
from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class LlamaMLAReader(LlamaReader):
    """Reader for DeepSeek-style MLA attention on Llama blocks."""

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool, model_cfg: dict, policy):
        super().__init__(new_params, unused_params, last_bin, model_cfg, policy)
        self.hf_model = None

    def set_hf_model(self, hf_model):
        self.hf_model = hf_model

    def mla(self, i: int, kind: str):
        if not kind:
            return self.filter(r'self_attn.*proj')
        result = []
        for key in ['q_a_proj', 'q_b_proj', 'q_proj', 'kv_a_proj_with_mqa', 'kv_b_proj', 'o_proj']:
            tensor = self.params.get(f'{self.attn_layer_prefix}.{i}.self_attn.{key}.{kind}')
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def mla_norm(self, i: int):
        result = []
        for k in ['q', 'kv']:
            name = f'{self.attn_layer_prefix}.{i}.self_attn.{k}_a_layernorm.weight'
            result.append(self.params.get(name))
        return (*result, )

    def _ffn(self, i: int, kind: str):
        """Get ffn kind for layer i with MLA bring-up fallback.

        In TM_MLA_BRINGUP mode, if dense MLP weights are missing from
        safetensors, pull them from the instantiated HF model instead.
        """
        if not kind:
            return self.filter(self.ffn_pattern)
        try:
            return super()._ffn(i, kind)
        except KeyError:
            if os.getenv('TM_MLA_BRINGUP', '') != '1' or self.hf_model is None:
                raise
            layer = self.hf_model.model.layers[i].mlp
            result = []
            for name in ['gate_proj', 'down_proj', 'up_proj']:
                proj = getattr(layer, name, None)
                if proj is None or getattr(proj, 'weight', None) is None:
                    raise
                tensor = proj.weight.detach()
                tensor = self.transform(tensor, kind)
                result.append(tensor)
            return (*result, )


@INPUT_MODELS.register_module(name='llama-mla')
class LlamaMLAModel(LlamaModel):
    """Llama MLA model (DeepSeek-style MLA/DSA) in hf format."""

    Reader = LlamaMLAReader

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs: dict):
        super().__init__(model_path, tokenizer_path, **kwargs)
        # Refresh model_config from arch in case LlamaMLAConfig differs
        _, cfg = get_model_arch(model_path)
        self.model_config = cfg.to_dict()

    def readers(self):
        """Yield readers, optionally with HF model attached in bring-up mode."""
        mappings = getattr(self.Reader, 'mappings', [])
        loader = create_loader(self.model_path, self.Reader.attn_layer_patten, mappings)

        hf_model = None
        if os.getenv('TM_MLA_BRINGUP', '') == '1':
            try:
                hf_model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map={'': 'cuda'},
                )
                hf_model.eval()
            except Exception as e:
                print(f'[MLA][bringup] failed to load HF model for FFN export fallback: {e}')
                hf_model = None

        for i, param in loader.items():
            reader = self.Reader(param, {}, False, self.model_config, policy=self.policy)
            if hf_model is not None and hasattr(reader, 'set_hf_model'):
                reader.set_hf_model(hf_model)
            yield i, reader

        torch.cuda.empty_cache()

    def model_info(self):
        """Read model info, then override with MLA/DSA geometry."""
        cfg = self.model_config
        info = super().model_info()

        # MLA dims from HF config
        kv_lora_rank = cfg.get('kv_lora_rank', 0) or 0
        q_lora_rank = cfg.get('q_lora_rank', 0) or 0
        qk_rope_dim = cfg.get('qk_rope_head_dim', 0) or 0
        qk_nope_dim = cfg.get('qk_nope_head_dim', 0) or 0
        v_head_dim = cfg.get('v_head_dim', info.get('size_per_head', 0) or 0)

        # QK dim is rope + nope; treat this as size_per_head under MLA.
        size_per_head = info.get('size_per_head', 0)
        if qk_rope_dim and qk_nope_dim:
            size_per_head = qk_rope_dim + qk_nope_dim

        # Derive qk_nope_dim if only size_per_head / qk_rope_dim are known.
        if qk_rope_dim and not qk_nope_dim and size_per_head:
            qk_nope_dim = size_per_head - qk_rope_dim

        # Sliding / full attention pattern (DSA)
        num_layer = cfg.get('num_hidden_layers', info.get('num_layer', 0) or 0)
        layer_types = cfg.get('layer_types', None)
        sliding_window = cfg.get('sliding_window', 0)
        window_size = []
        if layer_types is None:
            window_size = [0] * num_layer
        else:
            types = list(layer_types)
            if len(types) < num_layer:
                types = types + ['full_attention'] * (num_layer - len(types))
            elif len(types) > num_layer:
                types = types[:num_layer]
            for t in types:
                if t == 'sliding_attention':
                    window_size.append(sliding_window)
                else:
                    window_size.append(0)

        # Cache implementation: allow config override, then env, default naive.
        mla_cache_impl = cfg.get('mla_cache_impl', None)
        if not mla_cache_impl:
            mla_cache_impl = os.getenv('LMDEPLOY_MLA_CACHE_IMPL', 'naive')
        if mla_cache_impl not in ('naive', 'absorb'):
            raise ValueError(f'Unsupported mla_cache_impl: {mla_cache_impl}')

        # Rope dim should match qk_rope_dim when available.
        rope_param: RopeParam = info['rope_param']
        if qk_rope_dim:
            rope_param.dim = qk_rope_dim

        info.update(
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            qk_rope_dim=qk_rope_dim,
            v_head_dim=v_head_dim,
            size_per_head=size_per_head,
            window_size=window_size,
            mla_enabled=True,
            mla_cache_impl=mla_cache_impl,
            rope_param=rope_param,
        )

        # Log a concise MLA summary once at model-info time.
        if kv_lora_rank > 0 and qk_rope_dim > 0 and v_head_dim > 0:
            # Avoid bringing logger here; model_info is called once per convert.
            print(
                f'[MLA] enabled: cache_impl={mla_cache_impl} '
                f'kv_lora_rank={kv_lora_rank} qk_rope={qk_rope_dim} '
                f'qk_nope={qk_nope_dim} qk_dim={size_per_head} v_dim={v_head_dim} '
                f'sliding_window={sliding_window}'
            )

        return info
