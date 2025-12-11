// Copyright (c) OpenMMLab. All rights reserved.
// Adapted from TensorRT-LLM's EAGLE implementation

#pragma once

#include <memory>
#include <vector>
#include <string>
#include "src/turbomind/core/core.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/EagleDraftLayer.h"
#include "lmdeploy/turbomind/speculative_decoding_mode.h"

namespace turbomind {

// Forward declaration to avoid a hard include cycle between EagleModule
// and EagleBuffers. The full definition of EagleBuffers lives in
// src/turbomind/models/llama/EagleBuffers.h.
struct EagleBuffers;

class LlamaLinear;

struct EagleWeight {
    // Model level
    Tensor embed_tokens;
    Tensor fc;
    Tensor eagle_fc;

    // Layer level (single layer for Eagle3)
    Tensor input_norm;
    Tensor hidden_norm;
    Tensor attn_qkv;
    Tensor attn_o;
    Tensor attn_norm; // post_attention_layernorm
    Tensor mlp_gate_up;
    Tensor mlp_down;

    // Optional native Eagle3 midlayer projections (non‑LLaMA geometry).
    // When present, these hold midlayer.self_attn.{q,k,v,o}_proj weights
    // exactly as exported from the HF Eagle3 checkpoint and are consumed
    // by Eagle3AttentionLayer via Eagle3AttentionWeight.
    Tensor eagle_q_proj;
    Tensor eagle_k_proj;
    Tensor eagle_v_proj;
    Tensor eagle_o_proj;
    
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
    // input_ids: [batch_size] (per-slot draft tokens for this step)
    // last_hidden_states: [batch_size, hidden_units] (from target model)
    // captured_hidden_states: [batch_size, hidden_units * N] (optional,
    //   concatenated per-layer hidden states for Eagle3; may be empty)
    // output_logits: [batch_size, vocab_size]
    // output_hidden_states: [batch_size, hidden_units]
    void forward(const Tensor& input_ids,
                 const Tensor& last_hidden_states,
                 const Tensor& captured_hidden_states,
                 Tensor&       output_logits,
                 Tensor&       output_hidden_states,
                 LlamaLinear&  linear,
                 cudaStream_t  stream);

    // Backwards-compatible wrapper that ignores captured_hidden_states.
    void forward(const Tensor& input_ids,
                 const Tensor& hidden_states,
                 Tensor&       output_logits,
                 Tensor&       output_hidden_states,
                 LlamaLinear&  linear,
                 cudaStream_t  stream);

    // Lightweight accessors used by EagleBuffers and host-side tree builders.
    // These helpers are A-scope only and are not required for core decode.
    const EagleWeight& getWeights() const { return weights_; }

    /// Return the hidden size of the draft model loaded from config.yaml.
    int getHiddenUnits() const { return hidden_units_; }
    int getBaseHiddenUnits() const { return base_hidden_units_; }
    int getDraftHiddenUnits() const { return draft_hidden_units_; }

    /// Return the Eagle3 FC input dimension when available (0 otherwise).
    int getEagleFcInDim() const { return eagle_fc_in_dim_; }

    /// Return true when an Eagle3 draft layer has been prepared for this module.
    bool hasEagle3DraftLayer() const noexcept
    {
        return eagle_mode_ == EagleMode::kEagle3 && static_cast<bool>(eagle3_draft_layer_);
    }

    /// Optional capture layer ordering from config.yaml.
    const std::vector<int>& getCaptureLayers() const { return eagle_capture_layers_cfg_; }

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
    int              lm_head_input_dim_{0};
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

    // Base / draft hidden sizes. base_hidden_units_ tracks the runtime
    // hidden width used by the decoder (e.g. 2880 for GPT-OSS-120B),
    // while draft_hidden_units_ tracks the Eagle-3 midlayer FC output
    // width (e.g. 2880). hidden_units_ mirrors hidden_units from
    // config.yaml so that debug helpers can report it directly.
    int base_hidden_units_{0};
    int draft_hidden_units_{0};
    int eagle_q_size_{0};
    int eagle_kv_size_{0};
    int eagle_qkv_in_dim_{0};
    int eagle_qkv_in_factor_{0};
    int eagle_fc_in_dim_{0};
    int eagle_fc_in_factor_{0};
    std::vector<int> eagle_capture_layers_cfg_;

    // Shallow EagleNet block weights formatted for LlamaLinear
    // (single self‑attention + FC “MLP” style block).
    LlamaDenseWeight attn_qkv_weight_;
    LlamaDenseWeight attn_o_weight_;
    LlamaDenseWeight fc_weight_;
    LlamaDenseWeight eagle_fc_weight_;
    bool             draft_block_prepared_{false};

    // Scratch buffers reused across forward calls to avoid per‑step
    // allocations in the speculative decode hot path.
    Tensor attn_input_scratch_;        // [batch, hidden]
    Tensor attn_qkv_input_scratch_;    // [batch, qkv_in_dim] for Eagle3
    Tensor qkv_scratch_;               // [batch, q_size + 2 * kv_size]
    Tensor attn_out_scratch_;          // [batch, hidden]
    Tensor mlp_input_scratch_;         // unused after Eagle3 cleanup
    Tensor mlp_out_scratch_;           // unused after Eagle3 cleanup
    Tensor eagle_fc_out_scratch_;      // [batch, hidden] for Eagle3 pre-FC
    Tensor normed_hidden_scratch_;
    Tensor embed_input_scratch_;       // [batch, hidden] draft token embeddings
    Tensor embed_norm_scratch_;        // [batch, hidden] normalized embeddings
    Tensor logits_scratch_;
    Tensor residual_to_ffn_dim_scratch_; // [batch, draft_hidden_units_] for residual conversion
    Tensor lm_head_input_scratch_;       // [batch, lm_head_input_dim_] for LM head matmul

    // Optional Eagle3 draft layer weights. When present, this structurally
    // groups the fused QKV / Wo and MLP weights into LlamaAttentionWeight /
    // LlamaFfnWeight so a future, more faithful attention path can be wired
    // without changing EagleModule's public surface.
    std::unique_ptr<Eagle3DraftLayerWeight> eagle3_draft_layer_;

    friend class LlamaV2;

    // Cached model dims / dtype
    int      hidden_units_{0};        // legacy default (draft hidden unless overridden)
    int      vocab_size_{0};
    DataType weight_dtype_{kFloat16};
    float rope_base_{10000.0f};
    float rope_scale_{1.0f};

    // Debug views for eagle_forward_logits_debug / eagle_forward_debug.
    // These alias internal scratch buffers after the most recent forward
    // call when EAGLE debug is enabled.
    Tensor debug_fc_out_;
    Tensor debug_attn_input_;
    Tensor debug_attn_out_;
    Tensor debug_ffn_out_;
    Tensor debug_pre_head_hidden_;
    Tensor debug_logits_;

public:
    // Accessor for Eagle3 draft-layer weights; returns nullptr when
    // no Eagle3 draft layer has been prepared.
    const Eagle3DraftLayerWeight* eagle3_draft_layer() const noexcept
    {
        return eagle3_draft_layer_.get();
    }

    // Lightweight debug accessors; intended for tests / tooling only.
    const Tensor& debug_fc_out() const { return debug_fc_out_; }
    const Tensor& debug_attn_input() const { return debug_attn_input_; }
    const Tensor& debug_attn_out() const { return debug_attn_out_; }
    const Tensor& debug_ffn_out() const { return debug_ffn_out_; }
    const Tensor& debug_pre_head_hidden() const { return debug_pre_head_hidden_; }
    const Tensor& debug_logits() const { return debug_logits_; }

    // Tree-aware draft path used from LlamaV2. Current implementation
    // runs per-sequence logits only (no per-node logits); tree nodes
    // reuse per-slot last logits. TODO(Engineer C Task F): extend to
    // per-node draft logits using flattened tree inputs.
    void forward_draft_tree(const Tensor& last_hidden_states,
                            const Tensor& captured_hidden_states,
                            const Tensor& base_logits,
                            int           tokens_per_seq,
                            EagleBuffers& buffers,
                            Tensor&       draft_logits_buffer,
                            LlamaLinear&  linear,
                            cudaStream_t  stream);



    void initializeDefaultChoices();
};

} // namespace turbomind
