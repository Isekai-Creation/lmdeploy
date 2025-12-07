// Copyright (c) OpenMMLab. All rights reserved.
// Adapted from TensorRT-LLM's EAGLE implementation

#pragma once

#include <memory>
#include <vector>
#include <string>
#include "src/turbomind/core/core.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "lmdeploy/turbomind/speculative_decoding_mode.h"

namespace turbomind {

class LlamaLinear;

struct EagleWeight {
    // Model level
    Tensor embed_tokens;
    Tensor fc;
    
    // Layer level (single layer for Eagle3)
    Tensor input_norm;
    Tensor hidden_norm;
    Tensor attn_qkv;
    Tensor attn_o;
    Tensor attn_norm; // post_attention_layernorm
    Tensor mlp_gate_up;
    Tensor mlp_down;
    
    // Output level
    Tensor output_norm;
    Tensor lm_head;
    
    // Mapping
    Tensor draft_id_to_target_id;
    
    bool is_initialized = false;
};

/**
 * @brief EAGLE module for speculative decoding
 * 
 * Holds configuration and default tree choices for EAGLE/EAGLE3.
 * Mirrors TensorRT-LLM's EagleModule but simplified for TurboMind.
 */
class EagleModule {
public:
    using SizeType = int32_t;
    bool isEnabled() const noexcept { return enabled_; }
    
    /**
     * @brief Construct EAGLE module with configuration
     * 
     * @param max_draft_path_len Maximum length of a single draft path
     * @param max_decoding_draft_tokens Maximum number of draft tokens per step
     * @param max_decoding_tokens Maximum total tokens (draft + accepted) per step
     * @param max_non_leaf_nodes Maximum non-leaf nodes per layer
     */
    EagleModule(
        SizeType max_draft_path_len,
        SizeType max_decoding_draft_tokens,
        SizeType max_decoding_tokens,
        SizeType max_non_leaf_nodes
    );
    
    ~EagleModule() = default;
    
    // Getters
    SizeType getMaxDraftPathLen() const { return max_draft_path_len_; }
    SizeType getMaxDecodingDraftTokens() const { return max_decoding_draft_tokens_; }
    SizeType getMaxDecodingTokens() const { return max_decoding_tokens_; }
    SizeType getMaxNonLeafNodes() const { return max_non_leaf_nodes_; }
    
    /**
     * @brief Load draft model weights from directory
     * @param model_dir Path to the draft model directory
     * @param device_id GPU device ID
     * @param stream CUDA stream
     */
    void load(const std::string& model_dir, int device_id, cudaStream_t stream);
    
    // Run draft model forward pass
    // input_ids: [batch_size]
    // hidden_states: [batch_size, hidden_units] (from target model)
    // output_logits: [batch_size, vocab_size]
    // output_hidden_states: [batch_size, hidden_units]
    void forward(const Tensor& input_ids,
                 const Tensor& hidden_states,
                 Tensor& output_logits,
                 Tensor& output_hidden_states,
                 LlamaLinear& linear,
                 cudaStream_t stream);

    // Lightweight accessors used by EagleBuffers and host-side tree builders.
    // These helpers are A-scope only and are not required for core decode.
    const EagleWeight& getWeights() const { return weights_; }

    /// Return the hidden size of the draft model loaded from config.yaml.
    int getHiddenUnits() const { return hidden_units_; }

    /// Return the vocab size of the draft model loaded from config.yaml.
    int getVocabSize() const { return vocab_size_; }

    /// Return the number of int32 mask elements per token row.
    SizeType getNumPackedMasks() const
    {
        return (max_decoding_tokens_ + 31) / 32;
    }

    /// Return a conservative upper bound on tree nodes per step
    /// (max_decoding_tokens * max_draft_path_len).
    SizeType getMaxTreeNodes() const { return max_decoding_tokens_ * max_draft_path_len_; }
    /**
     * @brief Get default EAGLE tree choices
     *
     * Returns the predefined tree structure for EAGLE3.
     * Format: choices[i] = list of child indices for node i.
     */
    const std::vector<std::vector<SizeType>>& getDefaultChoices() const {
        return default_eagle_choices_;
    }

private:
    SizeType max_draft_path_len_;
    SizeType max_decoding_draft_tokens_;
    SizeType max_decoding_tokens_;
    SizeType max_non_leaf_nodes_;

    bool enabled_{false};  // set true only after successful load + validation
    
    // Default EAGLE3 tree structure
    // This is a simplified tree; can be loaded from config later
    std::vector<std::vector<SizeType>> default_eagle_choices_;
    
    // Draft model weights
    EagleWeight weights_;

    // Draft LM head formatted for LlamaLinear
    LlamaDenseWeight lm_head_weight_;
    bool             lm_head_prepared_{false};

    // Draft mode and geometry metadata. For legacy EagleNet drafts we
    // treat everything as a single-hidden block. For Eagle3 drafts, the
    // converter exposes a 2×hidden QKV input and separate Q / KV sizes
    // so that the shallow attention path can mirror TensorRT‑LLM’s
    // Eagle3 geometry.
    enum class EagleMode
    {
        kEagleNet,
        kEagle3,
    };

    EagleMode eagle_mode_{EagleMode::kEagleNet};

    int eagle_q_size_{0};
    int eagle_kv_size_{0};
    int eagle_qkv_in_dim_{0};
    int eagle_qkv_in_factor_{0};

    // Shallow EagleNet block weights formatted for LlamaLinear
    // (single self‑attention + FC “MLP” style block).
    LlamaDenseWeight attn_qkv_weight_;
    LlamaDenseWeight attn_o_weight_;
    LlamaDenseWeight fc_weight_;
    bool             draft_block_prepared_{false};

    // Scratch buffers reused across forward calls to avoid per‑step
    // allocations in the speculative decode hot path.
    Tensor attn_input_scratch_;        // [batch, hidden]
    Tensor attn_qkv_input_scratch_;    // [batch, qkv_in_dim] for Eagle3
    Tensor qkv_scratch_;               // [batch, q_size + 2 * kv_size]
    Tensor attn_out_scratch_;          // [batch, hidden]
    Tensor mlp_input_scratch_;    // [batch, 2 * hidden]
    Tensor mlp_out_scratch_;      // [batch, hidden]
    Tensor normed_hidden_scratch_;
    Tensor logits_scratch_;

    // Cached model dims / dtype
    int      hidden_units_{0};
    int      vocab_size_{0};
    DataType weight_dtype_{kFloat16};
    
    void initializeDefaultChoices();
};

} // namespace turbomind
