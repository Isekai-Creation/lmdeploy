// Copyright (c) OpenMMLab. All rights reserved.
// Adapted from TensorRT-LLM's EAGLE implementation

#include "EagleModule.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {

EagleModule::EagleModule(
    SizeType max_draft_path_len,
    SizeType max_decoding_draft_tokens,
    SizeType max_decoding_tokens,
    SizeType max_non_leaf_nodes
)
    : max_draft_path_len_(max_draft_path_len)
    , max_decoding_draft_tokens_(max_decoding_draft_tokens)
    , max_decoding_tokens_(max_decoding_tokens)
    , max_non_leaf_nodes_(max_non_leaf_nodes)
{
    TM_LOG_INFO("Initializing EagleModule: max_draft_path_len=%d, max_decoding_draft_tokens=%d, "
                "max_decoding_tokens=%d, max_non_leaf_nodes=%d",
                max_draft_path_len_, max_decoding_draft_tokens_, 
                max_decoding_tokens_, max_non_leaf_nodes_);
    
    initializeDefaultChoices();
}

void EagleModule::initializeDefaultChoices() {
    // Default EAGLE3 tree structure (simplified)
    // This creates a binary tree with depth = max_draft_path_len
    // In production, this should be loaded from config or model
    
    // For now, create a simple linear path structure
    // Node 0 (root) -> Node 1 -> Node 2 -> ... -> Node N (leaf)
    default_eagle_choices_.clear();
    
    // Example: 5-level tree with 2 branches per non-leaf node
    // This matches typical EAGLE3 configurations
    SizeType num_nodes = max_decoding_tokens_;
    default_eagle_choices_.resize(num_nodes);
    
    // Simple binary tree structure
    for (SizeType i = 0; i < num_nodes; ++i) {
        SizeType left_child = 2 * i + 1;
        SizeType right_child = 2 * i + 2;
        
        if (left_child < num_nodes) {
            default_eagle_choices_[i].push_back(left_child);
        }
        if (right_child < num_nodes) {
            default_eagle_choices_[i].push_back(right_child);
        }
    }
    
    TM_LOG_INFO("Initialized EAGLE tree with %zu nodes", default_eagle_choices_.size());
}

} // namespace turbomind
