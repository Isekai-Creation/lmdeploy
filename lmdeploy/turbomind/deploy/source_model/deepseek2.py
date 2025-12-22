# Copyright (c) OpenMMLab. All rights reserved.
import math

from ..config import RopeParam
from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class DeepSeek2Reader(LlamaReader):

    def moe_ffn_gate(self, i, kind):
        return self.params.get(f"model.layers.{i}.mlp.gate.{kind}")

    def moe_ffn_expert(self, e=None, i=None, kind=None):
        if not kind:
            return self.filter(r"experts")
        result = []
        for key in ["gate", "down", "up"]:
            name = f"model.layers.{i}.mlp.experts.{e}.{key}_proj.{kind}"
            tensor = self.params.get(name)
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result,)

    def _ffn(self, i: int, kind: str):
        """Get ffn kind for layer i."""
        if not kind:
            return self.filter(r"mlp" if i == 0 else r"shared_expert\.")
        result = []
        for key in ["gate", "down", "up"]:
            name = f"model.layers.{i}.mlp.shared_experts.{key}_proj.{kind}"
            if i == 0:
                name = name.replace("shared_experts.", "")
            tensor = self.params.get(name)
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result,)

    def mla(self, i: int, kind: str):
        if not kind:
            return self.filter(r"self_attn.*proj")
        result = []
        for key in [
            "q_a_proj",
            "q_b_proj",
            "q_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj",
        ]:
            tensor = self.params.get(
                f"{self.attn_layer_prefix}.{i}.self_attn.{key}.{kind}"
            )
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result,)

    def mla_mxfp4(self, i: int):
        """Get MXFP4 quantized MLA weights and dequantize to BF16.

        For MXFP4 models, weights are stored as:
          - {proj}_blocks: packed uint8 (2x4bit per byte)
          - {proj}_scales: uint8 scales with 127 bias

        Returns dequantized weights in the same order as mla().
        """
        from ..mxfp4_utils import dequant_mxfp4_weight

        result = []
        for key in [
            "q_a_proj",
            "q_b_proj",
            "q_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj",
        ]:
            blocks_key = f"{self.attn_layer_prefix}.{i}.self_attn.{key}_blocks"
            scales_key = f"{self.attn_layer_prefix}.{i}.self_attn.{key}_scales"

            blocks = self.params.get(blocks_key)
            scales = self.params.get(scales_key)

            if blocks is not None and scales is not None:
                # Dequantize MXFP4 to BF16
                import torch

                tensor = dequant_mxfp4_weight(blocks, scales, torch.bfloat16)
            else:
                # Fall back to regular weight loading
                tensor = self.params.get(
                    f"{self.attn_layer_prefix}.{i}.self_attn.{key}.weight"
                )
                tensor = (
                    self.transform(tensor, "weight") if tensor is not None else None
                )

            result.append(tensor)
        return (*result,)

    def mla_norm(self, i: int):
        result = []
        for k in ["q", "kv"]:
            name = f"{self.attn_layer_prefix}.{i}.self_attn.{k}_a_layernorm.weight"  # noqa: E501
            result.append(self.params.get(name))
        return (*result,)

    def dsa_indexer(self, kind: str):
        """Get DSA indexer weights for DeepSeek-V32 sparse attention."""
        if not kind:
            return self.filter(r"indexer")
        result = []
        for key in ["wq_b", "wk", "k_norm", "weights_proj"]:
            tensor = self.params.get(f"model.indexer.{key}.{kind}")
            tensor = self.transform(tensor, kind) if tensor is not None else None
            result.append(tensor)
        return (*result,)


def get_yarn_params(rope_scaling: dict):

    scaling_factor = float(rope_scaling["factor"])
    mscale = rope_scaling["mscale"]
    mscale_all_dim = rope_scaling["mscale_all_dim"]

    def yarn_get_mscale(scale=1, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    _mscale = float(
        yarn_get_mscale(scaling_factor, mscale)
        / yarn_get_mscale(scaling_factor, mscale_all_dim)
    )

    softmax_scale = 0
    if mscale_all_dim:
        scale = yarn_get_mscale(scaling_factor, mscale_all_dim)
        softmax_scale = scale * scale

    return _mscale, softmax_scale


@INPUT_MODELS.register_module(name="deepseek2")
@INPUT_MODELS.register_module(name="deepseek_v32")
class DeepSeek2Model(LlamaModel):

    Reader = DeepSeek2Reader

    def model_info(self):
        cfg = self.model_config
        info = super().model_info()
        qk_nope_dim = cfg["qk_nope_head_dim"]
        qk_rope_dim = cfg["qk_rope_head_dim"]
        num_layer = cfg["num_hidden_layers"]
        expert_num = cfg.get("n_routed_experts", 0)
        expert_num = [expert_num] * num_layer if expert_num else [0] * num_layer
        if expert_num[0]:
            expert_num[0] = 0
        n_shared_experts = cfg.get("n_shared_experts", 0) or 0
        expert_inter_size = cfg.get("moe_intermediate_size", 0)
        experts_per_token = cfg.get("num_experts_per_tok", 0)
        inter_size = (
            [n_shared_experts * expert_inter_size] * num_layer
            if n_shared_experts
            else [cfg["intermediate_size"]] * num_layer
        )
        if n_shared_experts:
            inter_size[0] = cfg["intermediate_size"]
        norm_topk_prob = cfg.get("norm_topk_prob", True)
        size_per_head = qk_rope_dim + qk_nope_dim
        info.update(
            kv_lora_rank=cfg["kv_lora_rank"],
            q_lora_rank=cfg.get("q_lora_rank") or 0,
            qk_rope_dim=qk_rope_dim,
            v_head_dim=cfg["v_head_dim"],
            size_per_head=size_per_head,
            expert_num=expert_num,
            expert_inter_size=expert_inter_size,
            experts_per_token=experts_per_token,
            inter_size=inter_size,
            norm_topk_prob=norm_topk_prob,
            routed_scale=cfg.get("routed_scaling_factor", 1.0),
            topk_method=cfg.get("topk_method", "greedy"),
            topk_group=cfg.get("topk_group", 1),
            moe_group_num=cfg.get("n_group", 1),
            tune_layer_num=2,
        )

        # DSA indexer parameters for DeepSeek-V32 sparse attention
        index_topk = cfg.get("index_topk", 0)
        if index_topk:
            info.update(
                index_topk=index_topk,
                index_head_dim=cfg.get("index_head_dim", 128),
                index_n_heads=cfg.get("index_n_heads", 64),
            )

        rope_param: RopeParam = info["rope_param"]
        rope_param.dim = qk_rope_dim
        rope_scaling = cfg.get("rope_scaling")
        if rope_scaling and rope_scaling.get("type") == "yarn":
            attention_factor, softmax_scale = get_yarn_params(rope_scaling)
            softmax_scale *= size_per_head ** (-0.5)
            rope_param.max_position_embeddings = rope_scaling[
                "original_max_position_embeddings"
            ]
            rope_param.attention_factor = attention_factor
            info.update(rope_param=rope_param, softmax_scale=softmax_scale)
        return info
