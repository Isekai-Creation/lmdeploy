/*
 * Copyright (c) 2024, LMDeploy Contributors.
 * EAGLE Speculative Decoding Implementation for LlamaV2
 */

#include "src/turbomind/models/llama/LlamaV2.h"
#include "lmdeploy/turbomind/eagle_tree.h"
#include "lmdeploy/turbomind/kernels/speculative_decoding/eagle_kernels.h"
#include "lmdeploy/turbomind/kernels/speculative_decoding/common.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {

void LlamaV2::eagleSpeculativeStep(
    Buffer_<int>     draft_tokens,
    Buffer_<int>     target_tokens,
    int              num_draft_tokens,
    Buffer_<int>     accepted_tokens,
    Buffer_<int>     accepted_lens,
    Buffer_<int>     num_accepted,
    const Sequence** sequences,
    int              batch_size
) {
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    
    if (!spec_mode_.isEagle() || !eagle_module_ || !eagle_buffers_) {
        TM_LOG_WARNING("[EAGLE] Speculative step called but EAGLE not initialized");
        if (num_accepted) {
            *num_accepted.data() = 0;
        }
        return;
    }
    
    if (num_draft_tokens == 0 || batch_size == 0) {
        if (num_accepted) {
            *num_accepted.data() = 0;
        }
        return;
    }
    
    TM_LOG_DEBUG("[EAGLE] Processing %d draft tokens for batch_size=%d",
                 num_draft_tokens, batch_size);

    // Interpret `draft_tokens` as flattened [batch_size, tokens_per_seq].
    // For now we still use only the first token per sequence when computing
    // one-step acceptance metrics, but we keep track of the layout so that
    // multi-token speculative steps can be wired later without changing this
    // interface.
    const int tokens_per_seq = std::max(1, num_draft_tokens / batch_size);
    const int expected_total = batch_size * tokens_per_seq;
    if (expected_total != num_draft_tokens) {
        TM_LOG_WARNING("[EAGLE] draft_tokens size (%d) is not a multiple of batch_size (%d); "
                       "interpreting as tokens_per_seq=%d and ignoring any trailing entries",
                       num_draft_tokens,
                       batch_size,
                       tokens_per_seq);
    }
    
    // ========== Step 1: Build EAGLE tree from draft tokens ==========
    eagle::SpeculationTree tree(
        engine_param_.spec_max_draft_path_len,
        engine_param_.spec_max_decoding_tokens
    );
    
    // Prefer the configured/default EAGLE choices from EagleModule when
    // available so the host-side tree matches the device-side layout.
    const auto& choices = eagle_module_->getDefaultChoices();
    if (!choices.empty()) {
        tree.buildTreeWithChoices(draft_tokens.data(), num_draft_tokens, choices);
    } else {
        tree.buildTree(draft_tokens.data(), num_draft_tokens);
    }
    tree.extractPaths();
    
    const int* paths_flat = tree.getPathsFlat();
    const int  num_paths  = tree.getNumPaths();
    const int  max_path_len = engine_param_.spec_max_draft_path_len;
    const int  max_decoding_tokens = engine_param_.spec_max_decoding_tokens;

    if (num_paths == 0) {
        TM_LOG_WARNING("[EAGLE] No valid paths in tree");
        if (num_accepted) {
            *num_accepted.data() = 0;
        }
        return;
    }
    
    TM_LOG_DEBUG("[EAGLE] Built tree with %d paths, max_depth=%d",
                 num_paths, max_path_len);
    
    // ========== Step 2: Prepare context for EagleNet ==========
    // Copy draft tokens to GPU buffers
    cudaMemcpyAsync(
        eagle_buffers_->inputs.draft_tokens,
        draft_tokens.data(),
        num_draft_tokens * sizeof(int),
        cudaMemcpyHostToDevice,
        stream_
    );
    
    // Prepare per-sequence path buffers. The current tree is single-sequence,
    // so we replicate its paths across all batch slots. This keeps the device
    // layout consistent with [batch_size, max_decoding_tokens, max_path_len]
    // while we still operate in a one-token-per-sequence regime.
    std::vector<int> host_paths(batch_size * max_decoding_tokens * max_path_len, -1);
    const int paths_per_seq = std::min(num_paths, max_decoding_tokens);

    for (int b = 0; b < batch_size; ++b) {
        for (int p = 0; p < paths_per_seq; ++p) {
            const int* src = paths_flat + p * max_path_len;
            int* dst = host_paths.data()
                       + (b * max_decoding_tokens + p) * max_path_len;
            std::copy(src, src + max_path_len, dst);
        }
    }

    cudaMemcpyAsync(
        eagle_buffers_->inputs.draft_paths,
        host_paths.data(),
        host_paths.size() * sizeof(int),
        cudaMemcpyHostToDevice,
        stream_);
    
    sync_check_cuda_error();
    
    // ========== Step 3: Generate leaf mask and packed masks ==========
    using namespace turbomind::kernels::eagle;
    
    // Build leaf mask (distinguish leaf from non-leaf nodes)
    invokeBuildLeafMask(
        eagle_buffers_->inputs.leaf_mask,
        eagle_buffers_->inputs.draft_paths,
        batch_size,
        engine_param_.spec_max_decoding_tokens,
        engine_param_.spec_max_draft_path_len,
        stream_
    );
    
    // Generate packed attention masks from paths (tree structure).
    std::vector<int> batch_slots(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        batch_slots[i] = i;
    }
    
    int* d_batch_slots = nullptr;
    check_cuda_error(cudaMalloc(&d_batch_slots, batch_size * sizeof(int)));
    check_cuda_error(cudaMemcpyAsync(
        d_batch_slots,
        batch_slots.data(),
        batch_size * sizeof(int),
        cudaMemcpyHostToDevice,
        stream_));
    
    invokeGetPackedMaskFromPath(
        eagle_buffers_->inputs.packed_masks,
        d_batch_slots,
        eagle_buffers_->inputs.draft_paths,
        engine_param_.spec_max_decoding_tokens,
        engine_param_.spec_max_draft_path_len,
        stream_
    );
    
    sync_check_cuda_error();
    
    TM_LOG_DEBUG("[EAGLE] Generated masks for %d paths", num_paths);
    
    // ========== Step 4/5: Accept/reject draft tokens (host-side, tree-based) ==========
    //
    // Implement a tree-based acceptance rule per sequence using the
    // speculation tree built above. For each sequence, we walk every path,
    // count how many tokens would be appended under the rule
    //
    //   accept while draft_token == target_token;
    //   on first mismatch, append the target token and stop;
    //
    // and pick the path with the longest accepted length.

    using namespace turbomind::kernels::speculative_decoding;

    std::vector<int> h_draft(num_draft_tokens);
    std::vector<int> h_target(num_draft_tokens);
    std::copy_n(draft_tokens.data(), num_draft_tokens, h_draft.data());
    std::copy_n(target_tokens.data(), num_draft_tokens, h_target.data());

    std::vector<int> h_accepted_lens(batch_size, 0);
    std::vector<int> h_accepted_tokens(batch_size * max_path_len, -1);
    std::vector<int> host_best_path_ids(batch_size, 0);

    int total_accepted = 0;

    for (int b = 0; b < batch_size; ++b) {
        int best_path_idx   = 0;
        int best_accept_len = 0;

        for (int p = 0; p < paths_per_seq; ++p) {
            int accepted = 0;

            for (int d = 0; d < max_path_len; ++d) {
                const int node_idx = paths_flat[p * max_path_len + d];
                if (node_idx <= 0) {  // skip root (0) and stop at -1
                    if (node_idx < 0) {
                        break;
                    }
                    continue;
                }

                const int token_idx = node_idx - 1;
                if (token_idx < 0 || token_idx >= num_draft_tokens) {
                    break;
                }

                const int draft_id  = h_draft[token_idx];
                const int target_id = h_target[token_idx];

                accepted += 1;

                if (draft_id != target_id) {
                    break;
                }
            }

            if (accepted > best_accept_len) {
                best_accept_len = accepted;
                best_path_idx   = p;
            }
        }

        h_accepted_lens[b] = best_accept_len;
        host_best_path_ids[b] = best_path_idx;
        total_accepted += best_accept_len;

        int written = 0;
        if (best_accept_len > 0) {
            for (int d = 0; d < max_path_len && written < best_accept_len; ++d) {
                const int node_idx = paths_flat[best_path_idx * max_path_len + d];
                if (node_idx <= 0) {
                    if (node_idx < 0) {
                        break;
                    }
                    continue;
                }
                const int token_idx = node_idx - 1;
                if (token_idx < 0 || token_idx >= num_draft_tokens) {
                    break;
                }

                const int target_id = h_target[token_idx];
                h_accepted_tokens[b * max_path_len + written] = target_id;
                written += 1;
            }
        }
    }

    if (accepted_lens) {
        TM_CHECK(accepted_lens.size() >= batch_size);
        std::copy(h_accepted_lens.begin(), h_accepted_lens.end(), accepted_lens.data());
    }

    if (accepted_tokens && max_path_len > 0) {
        const int max_tokens = batch_size * max_path_len;
        TM_CHECK(accepted_tokens.size() >= max_tokens);
        std::copy_n(h_accepted_tokens.data(), max_tokens, accepted_tokens.data());
    }

    if (num_accepted) {
        *num_accepted.data() = total_accepted;
    }

    if (eagle_buffers_ && eagle_buffers_->isAllocated()) {
        const int max_tokens = batch_size * max_path_len;

        check_cuda_error(cudaMemcpyAsync(eagle_buffers_->outputs.accepted_tokens,
                                         h_accepted_tokens.data(),
                                         max_tokens * sizeof(int),
                                         cudaMemcpyHostToDevice,
                                         stream_));
        check_cuda_error(cudaMemcpyAsync(eagle_buffers_->outputs.accepted_lens,
                                         h_accepted_lens.data(),
                                         batch_size * sizeof(int),
                                         cudaMemcpyHostToDevice,
                                         stream_));
        check_cuda_error(cudaMemcpyAsync(eagle_buffers_->inputs.best_path_ids,
                                         host_best_path_ids.data(),
                                         batch_size * sizeof(int),
                                         cudaMemcpyHostToDevice,
                                         stream_));

        invokePackAcceptedPaths(
            eagle_buffers_->outputs.accepted_lengths_cumsum,
            eagle_buffers_->outputs.accepted_path_offsets,
            eagle_buffers_->outputs.accepted_lens,
            eagle_buffers_->inputs.best_path_ids,
            eagle_buffers_->inputs.draft_paths,
            reinterpret_cast<SizeType const*>(d_batch_slots),
            batch_size,
            batch_size,
            paths_per_seq,
            max_path_len,
            stream_);
    }
    
    // ========== Step 6: Rewind KV cache for rejected tokens ==========
    const int effective_draft_tokens = batch_size;  // one token considered per seq
    if (total_accepted < effective_draft_tokens) {
        // Rewind KV cache to accepted_count position
        for (int b = 0; b < batch_size; ++b) {
            if (sequences[b]) {
                // In full implementation, update sequence->cache_len
                // to reflect only accepted tokens
                TM_LOG_DEBUG("[EAGLE] Would rewind KV cache for sequence %d", b);
            }
        }
    }
    
    // Log acceptance statistics
    float acceptance_rate = (effective_draft_tokens > 0)
                                ? static_cast<float>(total_accepted) / static_cast<float>(effective_draft_tokens)
                                : 0.0f;
    TM_LOG_INFO("[EAGLE] Accepted %d/%d draft tokens (%.1f%% acceptance rate)", 
                total_accepted, effective_draft_tokens, acceptance_rate * 100.0f);
    
    // Store acceptance rate in buffers for metrics
    float h_acceptance_rate = acceptance_rate;
    cudaMemcpyAsync(
        eagle_buffers_->outputs.acceptance_rate,
        &h_acceptance_rate,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream_
    );
    
    sync_check_cuda_error();
    
    TM_LOG_DEBUG("[EAGLE] Speculative step complete");
}

} // namespace turbomind
