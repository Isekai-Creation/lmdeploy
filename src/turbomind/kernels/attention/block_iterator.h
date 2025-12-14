// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "attention_params.h"
#include "block.h"

namespace turbomind {

template<class BlockHead, int CTA_S>
struct BlockIterator {

    BlockHead block_head_;
    char**    block_ptrs_;
    char**    scale_block_ptrs_{};

    // Optional FP4/NVFP4 scale pool metadata. When scale_block_ptrs_ is
    // non-null, these describe the logical layout needed by
    // get_fp4_mx_scale_base for this (layer, head) within the block pool.
    int head_idx_{};
    int head_num_{};
    int layer_id_{};
    int block_len_{};

    char* block_{};
    int   block_id_{};
    int   block_ti_{};

    __device__ BlockIterator(BlockHead block_head,
                             char**    block_ptrs,
                             char**    scale_block_ptrs,
                             int       head_idx,
                             int       head_num,
                             int       layer_id,
                             int       block_len):
        block_head_{block_head},
        block_ptrs_{block_ptrs},
        scale_block_ptrs_{scale_block_ptrs},
        head_idx_{head_idx},
        head_num_{head_num},
        layer_id_{layer_id},
        block_len_{block_len}
    {
    }

    __device__ void SetTile(int iter)
    {
        block_head_.get_block_coord(iter * CTA_S, block_id_, block_ti_);
        block_ = block_ptrs_[block_id_];
    }

    __device__ void Advance()
    {
        block_ti_ -= CTA_S;
        if (block_ti_ < 0) {
            block_ti_ += block_head_.block_len();
            block_id_ -= 1;
        }
        if (block_id_ >= 0) {
            block_ = block_ptrs_[block_id_];
        }
    }

    // Accessors for FP4 scale-pool plumbing.
    __device__ char** scale_blocks_seq() const
    {
        return scale_block_ptrs_;
    }

    __device__ int head_idx() const
    {
        return head_idx_;
    }

    __device__ int head_num() const
    {
        return head_num_;
    }

    __device__ int layer_id() const
    {
        return layer_id_;
    }

    __device__ int block_len() const
    {
        return block_len_;
    }

    template<int Index>
    __device__ auto OffsetPtr(int offset) const
    {
        if constexpr (Index == 0) {
            return block_head_.k_data(block_, block_ti_) + offset;
        }
        else if constexpr (Index == 1) {
            return block_head_.v_data(block_, block_ti_) + offset;
        }
        else if constexpr (Index == 2) {
            return block_head_.k_param(block_, block_ti_) + offset;
        }
        else if constexpr (Index == 3) {
            return block_head_.v_param(block_, block_ti_) + offset;
        }
        else {
            static_assert(Index != Index, "invalid index");
        }
    }
};

template<class T, class Tkv, class BlockLayout_, int CTA_S>
struct BlockIteratorFactory {
    using BlockLayout = BlockLayout_;

    BlockLayout_ block_layout_;
    char**       block_ptrs_;
    char**       scale_block_ptrs_;
    const int*   cu_block_nums_;
    int          layer_idx_;

    __device__ auto Create(int batch_idx, int head_idx)
    {
        block::Head<T, Tkv, BlockLayout> head{block_layout_, layer_idx_, head_idx};

        char** block_ptrs       = block_ptrs_ + cu_block_nums_[batch_idx];
        char** scale_block_ptrs = scale_block_ptrs_ ? scale_block_ptrs_ + cu_block_nums_[batch_idx] : nullptr;

        const int head_num  = block_layout_.config().head_num();
        const int block_len = block_layout_.config().block_len();

        return BlockIterator<block::Head<T, Tkv, BlockLayout>, CTA_S>{
            head, block_ptrs, scale_block_ptrs, head_idx, head_num, layer_idx_, block_len};
    }
};

template<class CacheIterFactory>
struct CreateCacheIterFactory<CacheIterFactory, std::void_t<typename CacheIterFactory::BlockLayout>> {
    template<class Param>
    static CacheIterFactory apply(const Param& param)
    {
        using BlockLayout = typename CacheIterFactory::BlockLayout;
        using BlockConfig = typename BlockLayout::Config;

        return {
            BlockLayout{BlockConfig{param.num_kv_heads, param.block_iter_params.block_len}},
            param.block_iter_params.block_ptrs,
            param.block_iter_params.scale_block_ptrs,
            param.block_iter_params.cu_block_nums,
            param.block_iter_params.layer_id,
        };
    }
};

template<class T, class Tkv, int CTA_S, int HeadDim>
using GetBlockIterFactory = BlockIteratorFactory<T, Tkv, block::Layout<block::Config<T, Tkv, HeadDim>>, CTA_S>;

}  // namespace turbomind
