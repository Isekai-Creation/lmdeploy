/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modified from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h

#pragma once

#include <vector>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/layers/DynamicDecodeLayer.h"
#include "src/turbomind/models/llama/LlamaBatch.h"
#include "src/turbomind/models/llama/LlamaWeight.h"
#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/unified_decoder.h"
#include "src/turbomind/models/llama/EagleModule.h"
#include "src/turbomind/models/llama/EagleBuffers.h"
#include "lmdeploy/turbomind/speculative_decoding_mode.h"

namespace turbomind {

class LlamaBatch;

class LlamaV2 {
public:
    struct SpecContext {
        int                 max_decoding_tokens_step{};
        const Sequence**    sequences{nullptr};
        const int*          d_sequence_lengths{nullptr};
        const int*          planned_tokens_per_seq{nullptr};
        EagleBuffers*       eagle_buffers{nullptr};
        bool                enable_eagle{false};
        bool                enable_eagle_target_tree{false};
    };

    LlamaV2(DataType                     dtype,
            const ModelParam&            model,
            const EngineParam&           engine,
            const AttentionParam&        attn,
            const MoeParam&              moe,
            const LoraParam&             lora,
            const Context&               ctx,
            int                          max_batch_size,
            std::shared_ptr<LlamaWeight> weights);

    size_t vocab_size() const noexcept
    {
        return vocab_size_;
    }

    bool isEagleEnabled() const noexcept
    {
        return spec_mode_.isEagle() && eagle_module_ && eagle_buffers_ && eagle_module_->isEnabled();
    }

    bool isTargetTreeDecodeEnabled() const noexcept
    {
        return engine_param_.enable_eagle_target_tree && target_tree_supported_;
    }

    bool isTargetTreeDecodeActiveStep() const noexcept
    {
        return isTargetTreeDecodeEnabled() && eagle_tree_target_tokens_valid_;
    }

private:
    void updateEmbedding(char*            decoder_input,
                         const int        bsz,
                         const int*       h_input_length,
                         const Sequence** sequences,
                         int              token_num,
                         int*             lora_mask,
                         bool*            have_embeddings);

    void Forward(Buffer_<int>     input_ids,
                 Tensor           hidden_states_out,
                 Tensor           decoder_out,
                 Buffer           kv_block_ptrs,
                 Buffer           cu_block_nums,
                 Buffer_<int>     h_input_length,
                 Buffer_<int>     h_context_length,
                 Buffer           rope_base,
                 MropeRope*       mrope,
                 Tensor           partial_ML,
                 Buffer           finished,
                 Buffer           local_token_nums,
                 Buffer           lora_mask,
                 int              decode_num,
                 int              prefil_num,
                 const Sequence** sequences);

    Tensor postDecodeEmbedding(const Tensor& features, Buffer local_logits);

    void dynamicDecode(Buffer token_ids,
                       Buffer finished,
                       Buffer sequence_length,
                       Tensor curand_state,
                       Tensor logits,
                       Buffer seq_limit_len,
                       Buffer init_context_length,
                       Buffer context_length,
                       Buffer prompt_length,
                       Buffer sampled_logprobs,  // <- indicator
                       Buffer sampled_indexes,
                       Buffer sampled_nums,
                       int    step,
                       int    max_context_len);

    void dynamicDecodeMultiStep(Buffer             token_ids,
                                Buffer             finished,
                                Buffer             sequence_length,
                                Tensor             curand_state,
                                Tensor             logits,
                                Buffer             seq_limit_len,
                                Buffer             init_context_length,
                                Buffer             context_length,
                                Buffer             prompt_length,
                                Buffer             sampled_logprobs,
                                Buffer             sampled_indexes,
                                Buffer             sampled_nums,
                                int                step,
                                int                max_context_len,
                                const ForcedTailContext* forced_ctx);

    void dynamicDecodeWithSpec(GenerationState& g,
                               Buffer           token_ids,
                               Buffer           finished,
                               Buffer           sequence_length,
                               Tensor           curand_state,
                               const Tensor&    decoder_features,
                               Tensor           logits,
                               Buffer           seq_limit_len,
                               Buffer           init_context_length,
                               Buffer           context_length,
                               Buffer           prompt_length,
                               Buffer           sampled_logprobs,
                               Buffer           sampled_indexes,
                               Buffer           sampled_nums,
                               int              max_context_len,
                               const SpecContext& spec_ctx);

    void dynamicDecodeWithSpecMulti(GenerationState& g,
                                    Buffer           token_ids,
                                    Buffer           finished,
                                    Buffer           sequence_length,
                                    Tensor           curand_state,
                                    const Tensor&    decoder_features,
                                    Tensor           logits,
                                    Buffer           seq_limit_len,
                                    Buffer           init_context_length,
                                    Buffer           context_length,
                                    Buffer           prompt_length,
                                    Buffer           sampled_logprobs,
                                    Buffer           sampled_indexes,
                                    Buffer           sampled_nums,
                                    int              max_context_len,
                                    const SpecContext& spec_ctx);

    // Per-step EAGLE acceptance summary. This is populated by the fused
    // dynamicDecodeWithSpec EAGLE branch and consumed by LlamaBatch for
    // metrics and (optional) multi-token advancement.
    void getEagleAcceptanceForStep(std::vector<int>& accepted_lens,
                                   std::vector<int>& accepted_tokens,
                                   int&              tokens_per_seq) const;

    // Run draft model (EagleNet) over the last-token hidden states to
    // produce draft logits / hidden states for speculative decoding.
    // This is a thin wrapper around EagleModule::forward so that
    // LlamaBatch does not need to access EagleModule directly.
    void eagleDraftForward(const Tensor& hidden_states,
                           Tensor&       draft_logits,
                           Tensor&       draft_hidden);

    // Experimental: run target-tree decode over the EAGLE speculation tree
    // for the current step. This prepares per-node target tokens for the
    // acceptance kernels by flattening draft_paths into generation inputs
    // and (in a later stage) running the base model on those tokens. When
    // disabled, TurboMind falls back to host-fabricated target_tokens.
    //
    // d_sequence_lengths is a device pointer to per-slot sequence lengths
    // (one entry per engine slot); it is used to derive position ids for
    // tree tokens. When nullptr, a base sequence length of 0 is assumed.
    void targetTreeDecode(int batch_size, const int* d_sequence_lengths);

    // Entry point for base-model target-tree decode. This runs the
    // staging kernel (`targetTreeDecode`) using the provided per-slot
    // device sequence lengths and then executes a dedicated decode pass
    // over the flattened tree tokens. The per-slot Sequence** array is
    // used only to query prefix KV layout; it is never mutated by the
    // tree path.
    void runEagleTargetTreeDecode(int batch_size,
                                  const int*       d_sequence_lengths,
                                  const Sequence** sequences);

    // Max engine tokens TurboMind should handle per decode step
    // when running in EAGLE speculative mode.
    int eagleMaxEngineTokensPerStep() const noexcept
    {
        return eagle_max_engine_tokens_per_step_;
    }

private:
    friend class LlamaBatch;

    const DataType dtype_;

    const ModelParam     param_;
    const AttentionParam attn_param_;
    const LoraParam      lora_param_;

    const Communicators* const comm_;

    const int    tp_size_;
    const int    tp_rank_;
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t hidden_units_;
    const size_t layer_num_;
    const size_t vocab_size_;
    const size_t vocab_size_padded_;
    const float  rmsnorm_eps_;
    const size_t local_head_num_;
    const size_t local_kv_head_num_;

    const std::shared_ptr<LlamaWeight> weights_;

    // Refs into `Context`, make the pointer constant (not the pointed objects)
    cudaStream_t const stream_;
    LlamaLinear&       linear_;

    // Optional pointer back to SequenceManager so that specialized
    // decode paths (e.g. target-tree decode) can reuse prefix KV block
    // pointers as read-only when building scratch decode passes.
    SequenceManager* sequence_manager_{nullptr};

    bool use_allgather_2d_{false};

    const bool debug_;

    std::unique_ptr<UnifiedDecoder>     unified_decoder_;
    std::unique_ptr<DynamicDecodeLayer> dynamic_decode_;
    
    // Speculative decoding (EAGLE)
    SpeculativeDecodingMode spec_mode_{SpeculativeDecodingMode::None()};
    std::unique_ptr<EagleModule>  eagle_module_;
    std::unique_ptr<EagleBuffers> eagle_buffers_;
    // Dedicated hidden-state and logits buffers for target-tree decode.
    // Hidden states follow the base model dtype; logits are kept in FP32
    // to keep argmax numerics stable over MXFP4 / BF16 compute.
    Tensor tree_hidden_states_;
    Buffer tree_logits_buffer_;
    int    max_tree_tokens_{0};
    // Concatenated hidden states captured from a small set of
    // decoder layers (e.g. the last 3) for Eagle3. This is
    // populated by UnifiedDecoder when speculative Eagle3 is
    // enabled and consumed by EagleModule in eagleDraftForward.
    Tensor eagle_capture_hidden_;
    const EngineParam engine_param_;
    int eagle_max_engine_tokens_per_step_{0};
    // Runtime gate for target-tree decode (may be disabled when
    // dtypes or shapes are incompatible with the current engine).
    bool target_tree_supported_{false};
    bool eagle_tree_target_tokens_valid_{false};

    // Cached per-step acceptance summary for EAGLE. These vectors live on
    // host and are updated once per decode step on the TP leader rank.
    std::vector<int> eagle_step_accepted_lens_;
    std::vector<int> eagle_step_accepted_tokens_;
    int              eagle_step_tokens_per_seq_{0};
    int              eagle_step_max_extra_{0};
};

}  // namespace turbomind
