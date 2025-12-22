# Copyright (c) OpenMMLab. All rights reserved.
"""MXFP4 Dequantization Utilities for DeepSeek_V32 MLA Weight Loading.

This module provides CPU-based dequantization for MXFP4 weights during
model conversion/loading. For GPU inference, the CUDA kernel in
mxfp4_dequant.cu is used.
"""

from typing import Tuple

import torch


# FP4 E2M1 lookup table (matches CUDA kernel and TransMLA convention)
FP4_LUT = torch.tensor(
    [
        +0.0,
        +0.5,
        +1.0,
        +1.5,
        +2.0,
        +3.0,
        +4.0,
        +6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def dequant_mxfp4_cpu(
    blocks: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Dequantize MXFP4 blocks + scales to dense tensor on CPU.

    Args:
        blocks: uint8 packed blocks, shape [..., groups, group_size//2]
                Low nibble = even index, high nibble = odd index
        scales: uint8 scales with 127 bias, shape [..., groups]
        dtype: Output dtype (bfloat16 or float16)

    Returns:
        Dequantized tensor, shape [..., groups * group_size]
    """
    *prefix_shape, num_groups, half_group = blocks.shape
    group_size = half_group * 2

    # Validate shapes
    assert scales.shape == (
        *prefix_shape,
        num_groups,
    ), f"scales shape {scales.shape} != expected {(*prefix_shape, num_groups)}"

    # Compute output shape
    out_cols = num_groups * group_size

    # Expand scales for broadcasting: [..., groups, 1]
    scales_expanded = scales.unsqueeze(-1).to(torch.int32) - 127
    scale_factors = torch.pow(2.0, scales_expanded.float())  # [..., groups, 1]

    # Unpack nibbles: blocks [..., groups, half_group]
    idx_lo = (blocks & 0x0F).long()  # [..., groups, half_group]
    idx_hi = (blocks >> 4).long()  # [..., groups, half_group]

    # Interleave: output [..., groups, group_size]
    out_unpacked = torch.zeros(
        *prefix_shape, num_groups, group_size, dtype=torch.float32, device=blocks.device
    )
    out_unpacked[..., 0::2] = FP4_LUT[idx_lo]
    out_unpacked[..., 1::2] = FP4_LUT[idx_hi]

    # Apply scaling
    out_scaled = out_unpacked * scale_factors

    # Reshape to final: [..., out_cols]
    out_final = out_scaled.reshape(*prefix_shape, out_cols)

    return out_final.to(dtype)


def dequant_mxfp4_weight(
    weight_blocks: torch.Tensor,
    weight_scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an MXFP4 weight tensor.

    Convenience wrapper for typical weight shape [out_features, groups, group_size//2].

    Args:
        weight_blocks: Packed weight blocks
        weight_scales: Weight scales
        dtype: Output dtype

    Returns:
        Dequantized weight tensor [out_features, in_features]
    """
    return dequant_mxfp4_cpu(weight_blocks, weight_scales, dtype)


def load_mxfp4_mla_weights(
    blocks_dict: dict, scales_dict: dict, dtype: torch.dtype = torch.bfloat16
) -> dict:
    """Load and dequantize all MLA MXFP4 weights.

    Args:
        blocks_dict: Dict of {layer_name: blocks_tensor}
        scales_dict: Dict of {layer_name: scales_tensor}
        dtype: Output dtype

    Returns:
        Dict of {layer_name: dequantized_weight}
    """
    result = {}

    mla_weight_names = [
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
    ]

    for name in mla_weight_names:
        blocks_key = f"{name}_blocks"
        scales_key = f"{name}_scales"

        if blocks_key in blocks_dict and scales_key in scales_dict:
            blocks = blocks_dict[blocks_key]
            scales = scales_dict[scales_key]
            weight = dequant_mxfp4_weight(blocks, scales, dtype)
            result[name] = weight

    return result
