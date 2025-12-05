// Copyright (c) OpenMMLab. All rights reserved.
// Adapted from TensorRT-LLM's EAGLE implementation

#pragma once

#include <memory>
#include <vector>
#include "lmdeploy/turbomind/speculative_decoding_mode.h"

namespace turbomind {

/**
 * @brief EAGLE module for speculative decoding
 * 
 * Holds configuration and default tree choices for EAGLE/EAGLE3.
 * Mirrors TensorRT-LLM's EagleModule but simplified for TurboMind.
 */
class EagleModule {
public:
    using SizeType = int32_t;
    
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
     * @brief Get default EAGLE tree choices
     * 
     * Returns the predefined tree structure for EAGLE3.
     * Format: choices[i] = list of child indices for node i.
     */
    const std::vector<std::vector<SizeType>>& getDefaultChoices() const {
        return default_eagle_choices_;
    }
    
    /**
     * @brief Get number of packed mask elements needed
     * 
     * Packed masks compress boolean attention masks into int32.
     * Returns ceil(max_decoding_tokens / 32).
     */
    SizeType getNumPackedMasks() const {
        return (max_decoding_tokens_ + 31) / 32;
    }

private:
    SizeType max_draft_path_len_;
    SizeType max_decoding_draft_tokens_;
    SizeType max_decoding_tokens_;
    SizeType max_non_leaf_nodes_;
    
    // Default EAGLE3 tree structure
    // This is a simplified tree; can be loaded from config later
    std::vector<std::vector<SizeType>> default_eagle_choices_;
    
    void initializeDefaultChoices();
};

} // namespace turbomind
