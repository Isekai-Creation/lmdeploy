// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <functional>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/sequence.h" // Add include for new Sequence.h

#include "src/turbomind/models/llama/BlockManager.h"
#include "src/turbomind/models/llama/BlockTrie.h"

namespace turbomind {

class SequenceManager {
public:
    // clang-format off
    struct BlockConfig {
        int head_dim_;
        int head_num_;
        int block_len_;
        int t_bits_;
        int q_bits_;
        int t_bits() const { return t_bits_; }
        int q_bits() const { return q_bits_; }
        int head_dim() const { return head_dim_; }
        int head_num() const { return head_num_; }
        int block_len() const { return block_len_; }
    };
    // clang-format on

    explicit SequenceManager(size_t             layer_num,
                             const BlockConfig& block_config,
                             double             block_count,
                             int                chunk_size,
                             bool               enable_prefix_caching,
                             int                rank,
                             int                attn_cp_size,
                             core::Allocator    allocator,
                             GetFreeMemSize     get_free_size,
                             size_t             scale_block_size = 0);

    SequenceManager(const SequenceManager&)     = delete;
    SequenceManager(SequenceManager&&) noexcept = default;

    [[nodiscard]] const Sequence* Create(uint64_t id);

    [[nodiscard]] const Sequence* Get(uint64_t id);

    [[nodiscard]] bool Contains(uint64_t id);

    [[nodiscard]] bool Erase(uint64_t id);

    void UpdateAndSetUnlock(const Sequence& seq);

    struct Outcome {
        int allocation;
        int swap_in;
        int swap_out;
    };

    using AdjustInputCount = std::function<int(const Sequences&, const std::vector<int>&)>;

    [[nodiscard]] Outcome Materialize(Sequences                    sequences,
                                      std::vector<int>             context_lengths,
                                      const std::vector<uint64_t>& priorities,
                                      int                          step_length,
                                      AdjustInputCount             adjust);

    /** @brief cache the input prompt tokens of each seq in sequences[0:active_size-1]
     *
     * @param sequences The sequence list
     * @param active_size the number of active sequences in the list
     */
    void CachePrompt(const Sequences& sequences, int active_size);

    /** @brief cache the generated tokens of a given sequence
     *
     * @param sequence the given sequence
     *
     * @note This function can only be called after the sequence finish generation
     * and all tokens including the prompt tokens and generated tokens have been put to
     * `seq.tokens`
     */
    void CacheGeneration(const Sequence& sequence);

    [[nodiscard]] void* GetBlockPtr(int block_id)
    {
        return block_manager_->block(block_id).data;
    }

    int max_block_count() const noexcept
    {
        return block_manager_->max_block_count();
    }

    int total_count() const noexcept
    {
        return block_manager_->total_count();
    }

    int active_count() const noexcept
    {
        return block_manager_->active_count();
    }

    int free_count() const noexcept
    {
        return block_manager_->free_count();
    }

    int cached_count() const noexcept
    {
        return block_manager_->cached_count();
    }

    // Unlock the given blocks in the underlying BlockManager, decreasing
    // their use_count and moving them from the active set to the cached
    // set when no sequences reference them any longer.
    void UnlockBlocks(const BlockIds& ids);

    // return #total_seq, #active_seq, #cached_seq
    std::tuple<int, int, int> seq_stats() const noexcept;

    // Optional: bind a secondary BlockManager that stores FP4/NVFP4
    // per-block scale factors. When present, all allocation / lock /
    // unlock / eviction operations are mirrored onto this manager
    // using the same block ids so that data and scale pools form a
    // single logical allocation unit.
    void AttachScaleBlockManager(std::shared_ptr<BlockManager> scale_block_manager);

    [[nodiscard]] void* GetScaleBlockPtr(int block_id)
    {
        return scale_block_manager_ ? scale_block_manager_->block(block_id).data : nullptr;
    }

private:
    void Erase(std::map<uint64_t, Sequence>::iterator& it);

    void CommitUnlockAndFree();

    void VerifyAndLockCached(const Sequences& sequences);

    std::vector<int> CountRequiredBlocks(const Sequences&        sequences,  //
                                         const std::vector<int>& context_lengths,
                                         int                     step_length);

    static void SortByPriority(Sequences&                   sequences,  //
                               std::vector<int>&            context_lengths,
                               const std::vector<uint64_t>& priorities);

    static void AssignAndActivate(const Sequences&        sequences,  //
                                  const std::vector<int>& counts,
                                  const BlockIds&         blocks,
                                  const UniqueIds&        unique_ids);

    void PrefixMatch(Sequences& sequences);

private:
    int block_seq_len_;
    int rank_;
    int attn_cp_size_;

    // Use `std::map` to avoid reference invalidation
    std::map<uint64_t, Sequence> sequences_;

    std::shared_ptr<BlockManager> block_manager_;
    // Optional FP4/NVFP4 scale pool. When non-null, this BlockManager
    // must have the same max_block_count() as block_manager_, and all
    // block ids managed by SequenceManager are interpreted identically
    // in both pools.
    std::shared_ptr<BlockManager> scale_block_manager_;
    std::shared_ptr<BlockTrie>    block_trie_;

    BlockIds unlocked_;
    BlockIds freed_;
};

inline std::ostream& operator<<(std::ostream& os, const SequenceManager::Outcome& oc)
{
    os << "allocation: " << oc.allocation << ", swap-in: " << oc.swap_in << ", swap-out: " << oc.swap_out;
    return os;
}

}  // namespace turbomind
