/*
 * Copyright (c) 2024, LMDeploy Contributors.
 * Based on TensorRT-LLM's EAGLE implementation
 *
 * EAGLE3 Tree Data Structures
 */

#pragma once

#include <vector>
#include <cstdint>
#include <cassert>

namespace turbomind {
namespace eagle {

using SizeType = int32_t;
using TokenIdType = int32_t;

/**
 * @brief Node in the speculation tree
 */
struct TreeNode {
    TokenIdType token_id;        // Token at this node
    SizeType parent_idx;         // Index of parent node (-1 for root)
    SizeType level;              // Depth in tree (0 = root)
    bool is_leaf;                // Whether this is a leaf node
    std::vector<SizeType> children;  // Indices of child nodes
    
    TreeNode() : token_id(-1), parent_idx(-1), level(0), is_leaf(true) {}
    
    TreeNode(TokenIdType token, SizeType parent, SizeType lv)
        : token_id(token), parent_idx(parent), level(lv), is_leaf(true) {}
};

/**
 * @brief Speculation tree for EAGLE3
 * 
 * Represents a tree of draft token sequences. Each path from root to leaf
 * is a candidate sequence that will be verified by the target model.
 */
class SpeculationTree {
public:
    SpeculationTree(SizeType max_depth = 5, SizeType max_paths = 10)
        : max_depth_(max_depth), max_paths_(max_paths) {
        nodes_.reserve(max_paths * max_depth);
    }
    
    /**
     * @brief Build tree from draft tokens
     * 
     * @param draft_tokens [num_draft_tokens] array of token IDs
     * @param num_tokens number of draft tokens
     */
    void buildTree(TokenIdType const* draft_tokens, SizeType num_tokens);

    /**
     * @brief Build tree from draft tokens following an explicit choices table.
     *
     * The choices table encodes children per node, where
     *   choices[i] = {child_node_indices...}
     * and node index 0 is reserved for the root. Tokens are assigned in
     * sequence to the created nodes up to @p num_tokens, respecting
     * @p max_depth_ and @p max_paths_.
     *
     * This is a C++ counterpart to the EAGLE choices loaded in
     * EagleModule (e.g. from eagle_tree.yaml) and allows offline-tuned
     * trees to be reflected in the host-side representation.
     *
     * @param draft_tokens [num_draft_tokens] array of token IDs
     * @param num_tokens number of draft tokens
     * @param choices per-node child indices (node 0 is the root)
     */
    void buildTreeWithChoices(
        TokenIdType const* draft_tokens,
        SizeType num_tokens,
        std::vector<std::vector<SizeType>> const& choices);
    
    /**
     * @brief Extract all paths from root to leaves
     * 
     * Paths are stored as [num_paths][max_path_len] with -1 padding.
     */
    void extractPaths();
    
    /**
     * @brief Find best path given acceptance results
     * 
     * @param accepted [num_paths] boolean array of accepted paths
     * @return index of best path
     */
    SizeType findBestPath(bool const* accepted) const;
    
    /**
     * @brief Get non-leaf nodes at specified level
     * 
     * @param level tree level (0 = root)
     * @return vector of node indices
     */
    std::vector<SizeType> getNonLeafNodes(SizeType level) const;
    
    /**
     * @brief Add new level to tree
     * 
     * @param parent_indices indices of parent nodes
     * @param tokens tokens to add as children
     * @param num_tokens number of tokens
     */
    void addLevel(
        SizeType const* parent_indices,
        TokenIdType const* tokens,
        SizeType num_tokens
    );
    
    /**
     * @brief Get paths as flat array
     * 
     * @return pointer to paths [num_paths][max_path_len]
     */
    SizeType const* getPathsFlat() const { return paths_flat_.data(); }
    
    /**
     * @brief Get number of paths
     */
    SizeType getNumPaths() const { return paths_.size(); }
    
    /**
     * @brief Get maximum tree depth
     */
    SizeType getMaxDepth() const { return max_depth_; }
    
    /**
     * @brief Clear tree for next iteration
     */
    void reset();

private:
    std::vector<TreeNode> nodes_;
    std::vector<std::vector<SizeType>> paths_;  // [num_paths][path_len]
    std::vector<SizeType> paths_flat_;          // Flattened paths
    
    SizeType max_depth_;
    SizeType max_paths_;
    
    // Helper: recursively extract paths
    void extractPathsRecursive(
        SizeType node_idx,
        std::vector<SizeType>& current_path
    );
};

/**
 * @brief Path representation for EAGLE3
 * 
 * Stores paths as [batch_size][max_decoding_tokens][max_path_len]
 * with -1 indicating end of path.
 */
struct PathRepresentation {
    std::vector<SizeType> data;  // Flat storage
    SizeType batch_size;
    SizeType max_decoding_tokens;
    SizeType max_path_len;
    
    PathRepresentation(SizeType bs, SizeType max_tokens, SizeType max_len)
        : batch_size(bs)
        , max_decoding_tokens(max_tokens)
        , max_path_len(max_len) {
        data.resize(bs * max_tokens * max_len, -1);
    }
    
    // Get path for specific batch and token
    SizeType* getPath(SizeType batch_idx, SizeType token_idx) {
        return data.data() + 
               (batch_idx * max_decoding_tokens + token_idx) * max_path_len;
    }
    
    SizeType const* getPath(SizeType batch_idx, SizeType token_idx) const {
        return data.data() + 
               (batch_idx * max_decoding_tokens + token_idx) * max_path_len;
    }
};

} // namespace eagle
} // namespace turbomind
