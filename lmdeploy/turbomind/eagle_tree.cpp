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
    assert(max_depth_ > 0 && max_paths_ > 0);

    // Backwards-compatible helper: build a simple chain of nodes from the
    // root. This is still used when no explicit EAGLE choices are provided.
    reset();

    if (num_tokens == 0) {
        return;
    }

    // Create root node (golden token)
    nodes_.emplace_back(-1, -1, 0);  // Root has no token

    SizeType max_nodes = max_paths_ * max_depth_;
    SizeType next_token_idx = 0;

    SizeType parent_idx = 0;
    while (next_token_idx < num_tokens
           && static_cast<SizeType>(nodes_.size()) < max_nodes
           && nodes_[parent_idx].level + 1 < max_depth_) {
        TokenIdType token = draft_tokens[next_token_idx++];
        nodes_.emplace_back(token, parent_idx, nodes_[parent_idx].level + 1);
        SizeType child_idx = static_cast<SizeType>(nodes_.size() - 1);
        nodes_[parent_idx].children.push_back(child_idx);
        nodes_[parent_idx].is_leaf = false;
        parent_idx = child_idx;
    }

    // Invariants: number of nodes should not exceed max_paths_ * max_depth_.
    assert(static_cast<SizeType>(nodes_.size()) <= max_paths_ * max_depth_);
}

void SpeculationTree::buildTreeWithChoices(
    TokenIdType const* draft_tokens,
    SizeType num_tokens,
    std::vector<std::vector<SizeType>> const& choices
) {
    reset();

    if (num_tokens == 0) {
        return;
    }

    // Root node (index 0) has no token; all other nodes map to draft_tokens
    nodes_.emplace_back(-1, -1, 0);

    SizeType next_token_idx = 0;

    // Helper lambda to add a node and assign a token if available.
    auto add_child = [&](SizeType parent_idx, SizeType level) -> SizeType {
        TokenIdType token = -1;
        if (next_token_idx < num_tokens) {
            token = draft_tokens[next_token_idx++];
        }
        nodes_.emplace_back(token, parent_idx, level);
        auto child_idx = static_cast<SizeType>(nodes_.size() - 1);
        nodes_[parent_idx].children.push_back(child_idx);
        nodes_[parent_idx].is_leaf = false;
        return child_idx;
    };

    // Build nodes breadth-first following the choices table until we run
    // out of tokens, reach max depth, or exhaust the choices.
    assert(max_depth_ > 0 && max_paths_ > 0);

    SizeType max_nodes = max_paths_ * max_depth_;
    for (SizeType node_idx = 0; node_idx < static_cast<SizeType>(choices.size()); ++node_idx) {
        if (node_idx >= static_cast<SizeType>(nodes_.size())) {
            break;
        }
        auto const& node = nodes_[node_idx];
        if (node.level + 1 >= max_depth_) {
            continue;
        }
        auto const& children = choices[node_idx];
        for (SizeType child_symbol : children) {
            (void)child_symbol;  // currently unused; keeps the compiler happy

            if (next_token_idx >= num_tokens || static_cast<SizeType>(nodes_.size()) >= max_nodes) {
                break;
            }
            add_child(node_idx, node.level + 1);
        }
        if (next_token_idx >= num_tokens || static_cast<SizeType>(nodes_.size()) >= max_nodes) {
            break;
        }
    }

    // Invariants: number of nodes should not exceed max_paths_ * max_depth_.
    assert(static_cast<SizeType>(nodes_.size()) <= max_nodes);
}

void SpeculationTree::extractPaths() {
    paths_.clear();
    paths_flat_.clear();
    
    if (nodes_.empty()) {
        return;
    }
    
    // Start from root
    std::vector<SizeType> current_path;
    extractPathsRecursive(0, current_path);
    
    // Flatten paths for GPU
    for (auto const& path : paths_) {
        assert(!path.empty());
        assert(path.size() <= static_cast<size_t>(max_depth_));
        for (auto node_idx : path) {
            // Each entry should refer to a valid node.
            assert(node_idx >= 0 && node_idx < static_cast<SizeType>(nodes_.size()));
            paths_flat_.push_back(node_idx);
        }
        // Pad to max_depth_
        for (SizeType i = path.size(); i < max_depth_; ++i) {
            paths_flat_.push_back(-1);
        }
    }

    // Invariants: flattened size should equal num_paths * max_depth_.
    assert(paths_flat_.size() == static_cast<size_t>(paths_.size()) * static_cast<size_t>(max_depth_));
}

void SpeculationTree::extractPathsRecursive(
    SizeType node_idx,
    std::vector<SizeType>& current_path
) {
    assert(node_idx >= 0 && node_idx < static_cast<SizeType>(nodes_.size()));

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
