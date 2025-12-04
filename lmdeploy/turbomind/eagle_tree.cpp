/*
 * Copyright (c) 2024, LMDeploy Contributors.
 *
 * EAGLE3 Tree Implementation
 */

#include "eagle_tree.h"
#include <algorithm>
#include <queue>

namespace turbomind {
namespace eagle {

void SpeculationTree::buildTree(
    TokenIdType const* draft_tokens,
    SizeType num_tokens
) {
    reset();
    
    if (num_tokens == 0) return;
    
    // Create root node (golden token)
    nodes_.emplace_back(-1, -1, 0);  // Root has no token
    
    // Add draft tokens as tree nodes
    // For now, simple linear tree (will expand to branching later)
    for (SizeType i = 0; i < num_tokens && i < max_depth_; ++i) {
        SizeType parent_idx = i;  // Parent is previous node
        nodes_.emplace_back(draft_tokens[i], parent_idx, i + 1);
        
        // Add child to parent
        nodes_[parent_idx].children.push_back(nodes_.size() - 1);
        nodes_[parent_idx].is_leaf = false;
    }
}

void SpeculationTree::extractPaths() {
    paths_.clear();
    paths_flat_.clear();
    
    if (nodes_.empty()) return;
    
    // Start from root
    std::vector<SizeType> current_path;
    extractPathsRecursive(0, current_path);
    
    // Flatten paths for GPU
    for (auto const& path : paths_) {
        for (auto node_idx : path) {
            paths_flat_.push_back(node_idx);
        }
        // Pad to max_depth_
        for (SizeType i = path.size(); i < max_depth_; ++i) {
            paths_flat_.push_back(-1);
        }
    }
}

void SpeculationTree::extractPathsRecursive(
    SizeType node_idx,
    std::vector<SizeType>& current_path
) {
    current_path.push_back(node_idx);
    
    auto const& node = nodes_[node_idx];
    
    if (node.is_leaf) {
        // Reached leaf, save path
        paths_.push_back(current_path);
    } else {
        // Recurse to children
        for (auto child_idx : node.children) {
            extractPathsRecursive(child_idx, current_path);
        }
    }
    
    current_path.pop_back();
}

SizeType SpeculationTree::findBestPath(bool const* accepted) const {
    // Find longest accepted path
    SizeType best_path = 0;
    SizeType max_accepted = 0;
    
    for (SizeType i = 0; i < paths_.size(); ++i) {
        if (accepted[i]) {
            SizeType path_len = paths_[i].size();
            if (path_len > max_accepted) {
                max_accepted = path_len;
                best_path = i;
            }
        }
    }
    
    return best_path;
}

std::vector<SizeType> SpeculationTree::getNonLeafNodes(SizeType level) const {
    std::vector<SizeType> non_leaf_nodes;
    
    for (SizeType i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].level == level && !nodes_[i].is_leaf) {
            non_leaf_nodes.push_back(i);
        }
    }
    
    return non_leaf_nodes;
}

void SpeculationTree::addLevel(
    SizeType const* parent_indices,
    TokenIdType const* tokens,
    SizeType num_tokens
) {
    SizeType current_level = nodes_.empty() ? 0 : nodes_.back().level + 1;
    
    for (SizeType i = 0; i < num_tokens; ++i) {
        SizeType parent_idx = parent_indices[i];
        
        // Add new node
        nodes_.emplace_back(tokens[i], parent_idx, current_level);
        SizeType new_node_idx = nodes_.size() - 1;
        
        // Update parent
        nodes_[parent_idx].children.push_back(new_node_idx);
        nodes_[parent_idx].is_leaf = false;
    }
}

void SpeculationTree::reset() {
    nodes_.clear();
    paths_.clear();
    paths_flat_.clear();
}

} // namespace eagle
} // namespace turbomind
