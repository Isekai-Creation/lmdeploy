// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <curand_kernel.h>

#include "src/turbomind/core/core.h"

#include "src/turbomind/engine/gateway.h"
#include "src/turbomind/engine/request.h"

#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_params.h"

#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/metrics.h"

namespace turbomind {

struct MropeRope {
    int          stride{};
    Tensor_<int> position_ids;
    Buffer_<int> position_delta;
    Buffer_<int> length;
};

struct BatchState {

    Buffer_<int>  h_prompt_length;  // history + input, ignore generated
    Buffer_<int>  h_context_length;
    Buffer_<bool> h_finished;

    MropeRope mrope;

    Tensor_<uint8_t> curand_state;  // [n, sizeof(curandState_t)]

    Tensor_<int> output_ids;  // output ids in [B, S]

    Buffer_<float> h_rope_theta;

    std::vector<int> seq_len_limit;

    std::vector<const Sequence*>          sequences;
    std::vector<std::shared_ptr<Request>> requests;

    std::vector<int> errors;

    // |<-- existing -->|<-- swap-in -->|
    // |<----------- active ----------->|<-- inactive -->|
    int active_size;
    int size;
};

class LlamaV2;

struct GenerationState {
    int max_init_ctx_len;
    int step;

    int partial;
    int partial_context_legnth;

    std::vector<uint64_t> unique_ids;

    bool skip_init_sampling;

    // min tokens per iter for satisfying `max_prefill_iters` constraint
    std::deque<int> min_input_count;

    int finished_count;
};

class LlamaBatch {
public:
    void AllocateBuffer(ssize_t batch_size, ssize_t session_len, int cache_block_seq_len);

    void AllocSymmBuffers();
    void FreeSymmBuffers();

    void FreeBuffer();

    using Requests = std::vector<std::shared_ptr<Request>>;
    using Signal   = std::function<void()>;

    void DisableInvalidRequests(Requests& infer_reqs, Requests& kill_reqs);

    void ProcessKillRequests(const Requests& reqs, std::vector<Signal>& signals);

    void ProcessInferRequests(const Requests& reqs, std::vector<Signal>& signals);

    int AdjustMaxInputCount(GenerationState&                    g,
                            const std::vector<const Sequence*>& sequences,
                            const std::vector<int>&             context_length);

    void Initialize(GenerationState& g);

    void InitializeSampling(const GenerationState& g);

    bool Forward(GenerationState& g);

    void Finish(GenerationState& g, std::vector<Signal>& signals);

    [[nodiscard]] Signal Interrupt(int index, bool force_stop = false);

    void ComputeAndOutputLogits(const Tensor& hidden_states, int first, int last);

    void OutputLogits(const Tensor& logits, int first, int last, GenerationConfig::OutType out_type);

    void OutputLastHiddenState(const Tensor& hidden_states, int first, int last);

    explicit LlamaBatch(DataType                 data_type,
                        const EngineParam&       param,
                        std::unique_ptr<LlamaV2> model,
                        std::shared_ptr<Context> ctx,
                        std::shared_ptr<Gateway> gateway,
                        int                      device_id,
                        int                      dp_rank);

    ~LlamaBatch();

    void InitializeBufferAndKVCache();

    void FreeBufferAndKVCache();

    void Start();

    LlamaV2& model() noexcept
    {
        return *model_;
    }

    int session_len() const noexcept
    {
        return session_len_;
    }

    void Warmup();

    ScheduleMetrics getScheduleMetrics()
    {
        const std::lock_guard<std::mutex> lock(metrics_mutex_);
        return schedule_metrics_;
    }

    void advanceSequencesByEagleAcceptance(const std::vector<int>&  dynamic_tokens,
                                           const std::vector<bool>& finished_slots,
                                           int                      batch_size,
                                           int                      eagle_tokens_per_seq,
                                           const std::vector<int>&  eagle_accepted_lens,
                                           const std::vector<int>&  eagle_accepted_tokens,
                                           GenerationState&         g);

private:
    void FindCanceledIndices(std::vector<int>& indices);

    void ProcessCancelRequests(std::vector<int>& indices, std::vector<Signal>& signals);

    void InternalThreadEntry();

    void OutputThreadEntry();

    void CopyState(const std::vector<std::tuple<BatchState*, BatchState*, int, int>>& desc);

    template<class... Ts>
    void IndexedCopyImpl(const int* src_idx, const int* dst_idx, int count, const std::tuple<Ts*, Ts*, int>&... cpys)
    {
        if (!count) {
            return;
        }
        constexpr int N = sizeof...(Ts);
        static_assert((!std::is_same_v<Ts, void> && ...));
        std::array<void*, N> src_ptr{std::get<0>(cpys)...};
        std::array<void*, N> dst_ptr{std::get<1>(cpys)...};
        std::array<int, N>   elem_sz{int(sizeof(Ts) * std::get<2>(cpys))...};
        invokeIndexedCopy(src_ptr.data(),  //
                          dst_ptr.data(),
                          elem_sz.data(),
                          src_idx,
                          dst_idx,
                          count,
                          N,
                          stream_);
        sync_check_cuda_error();
    }

    template<class... Ts>
    void IndexedCopy(const std::vector<int>& src_idx,
                     const std::vector<int>& dst_idx,
                     const std::tuple<Ts*, Ts*, int>&... cpys)
    {
        // has the same size, or one is empty
        FT_CHECK(src_idx.size() == dst_idx.size() || (src_idx.empty() ^ dst_idx.empty()));
        IndexedCopyImpl(src_idx.empty() ? nullptr : src_idx.data(),
                        dst_idx.empty() ? nullptr : dst_idx.data(),
                        std::max(src_idx.size(), dst_idx.size()),
                        cpys...);
    }

    template<class... Ts>
    void IndexedCopy(int count, const std::tuple<Ts*, Ts*, int>&... cpys)
    {
        IndexedCopyImpl(nullptr, nullptr, count, cpys...);
    }

    void* SymmAlloc(size_t size, bool register_);

    void SymmFree(void* ptr, size_t size, bool deregister);

    void DestroyCommunicators();

    void UpdateMetrics();

    // Multi-token EAGLE gating helpers. These centralize the conditions under
    // which experimental multi-token advance/rewind are allowed.
    bool isEagleMultiTokenStepEnabled(const GenerationState& g) const;
    bool isEagleMultiTokenSlotEnabled(int slot) const;

    // EAGLE post-decode helpers to keep `Forward` readable.
    void collectEagleStepHostState(const GenerationState& g,
                                   int                    batch_size,
                                   std::vector<int>&      h_token_ids,
                                   std::vector<bool>&     h_finished_slots);

    void updateEagleMetricsAndKVLengths(const GenerationState&   g,
                                        const std::vector<int>&  h_token_ids,
                                        const std::vector<bool>& h_finished_slots,
                                        int                      batch_size,
                                        int                      max_path_len,
                                        int                      eagle_tokens_per_seq,
                                        const std::vector<int>&  eagle_accepted_lens,
                                        const std::vector<int>&  eagle_accepted_tokens,
                                        std::vector<int>&        kv_draft_lengths,
                                        std::vector<int>&        kv_accepted_lengths);

    void runEagleMultiTokenAdvance(const std::vector<int>&  h_token_ids,
                                   const std::vector<bool>& h_finished_slots,
                                   int                      batch_size,
                                   int                      eagle_tokens_per_seq,
                                   const std::vector<int>&  eagle_accepted_lens,
                                   const std::vector<int>&  eagle_accepted_tokens,
                                   GenerationState&         g);

    void runEagleKVRewind(const std::vector<int>& kv_draft_lengths,
                          const std::vector<int>& kv_accepted_lengths,
                          int                     batch_size,
                          const GenerationState&  g);

    // Disable multi-token speculative decoding for a given slot and latch the
    // corresponding request into single-token mode for the remainder of its
    // lifetime. This is the only place that should mutate
    // `eagle_disable_multitoken_slot_` so that all fallback paths share
    // consistent semantics and logging.
    void disableEagleMultitokenForSlot(int slot, const char* reason);

private:
    const EngineParam param_;

    const std::shared_ptr<Gateway> gateway_;

    const int      max_batch_size_;
    const int      max_forward_token_num_;
    const int      max_context_token_num_;
    const int      num_tokens_per_iter_;
    const int      max_prefill_iters_;
    const int      device_id_;
    const int      dp_rank_;
    const int      tp_size_;
    const int      tp_rank_;
    const DataType data_type_;
    const bool     debug_;

    // Refs into `Context<T>`
    cudaStream_t const stream_{};

    int session_len_;  // May be truncated in ctor

    std::shared_ptr<Context>         context_;
    std::unique_ptr<LlamaV2>         model_;
    std::unique_ptr<SequenceManager> sequence_manager_;

    // Per-sequence planned draft token budget for the current EAGLE step.
    // Length is at most `max_batch_size_` and is populated in the EAGLE
    // planning branch of `Forward`. For now all entries share the same
    // value, but the layout supports per-sequence variability.
    std::vector<int> eagle_planned_tokens_per_seq_;

    // Per-slot kill-switch for multi-token EAGLE. When a hard invariant is
    // violated for a given request, the corresponding slot is marked here and
    // multi-token advance/rewind is disabled for the remainder of that
    // request's lifetime.
    std::vector<uint8_t> eagle_disable_multitoken_slot_;

    // Last observed generation step for EAGLE-related operations. Updated
    // from the decode loop so that helper utilities (like the per-slot
    // kill-switch) can emit meaningful logs without threading GenerationState
    // through every call site.
    int eagle_last_step_{-1};

    // First EOS id per request for EAGLE acceptance (device buffer).
    Buffer_<int>   eagle_end_ids_;
    Buffer_<float> eagle_posterior_thresholds_;  // [max_batch_size_]
    Buffer_<float> eagle_posterior_alphas_;      // [max_batch_size_]
    Buffer_<float> eagle_temperatures_;          // [max_batch_size_]

    Communicators& comm_;

    Allocator symm_alloc_;

    ///////////////////////////////////////////////////////////////////
    // k/v cache block buffers
    Buffer_<int>       cu_block_counts_;
    // Device-side per-block KV data pointers for the current batch.
    Buffer_<uintptr_t> block_ptrs_;
    // Device-side per-block FP4 scale pointers (only populated when FP4
    // KV cache is active; otherwise empty).
    Buffer_<uintptr_t> scale_block_ptrs_;

    // EAGLE KV rewind integration (Engineer B scope)
    int           kv_block_size_{};          // tokens per KV block
    int           kv_max_blocks_per_seq_{};  // max blocks per sequence
    Buffer_<int>  eagle_kv_rewind_lengths_;  // [max_batch_size_]
    Buffer_<int>  eagle_kv_batch_slots_;     // [max_batch_size_]
    Buffer_<int>  eagle_kv_block_tables_;    // [max_batch_size_, kv_max_blocks_per_seq_]
    void**        eagle_kv_cache_blocks_{nullptr};  // [num_layers, kv_max_blocks_per_seq_]

    ////////////////////////////////////////////////////////////////////
    // context decoding temp buffers
    Tensor symm_hidden_states_buf_;
    Tensor symm_logits_buf_;

    // context parallel
    Tensor_<float> symm_partial_ML_;

    Tensor decoder_output_buf_;

    Tensor_<float> sampling_logits_;

    Buffer_<int> input_ids_buf_;

    // lengths
    Buffer_<int> input_length_buf_;    // input + cache missed length
    Buffer_<int> context_length_buf_;  // history length + input_length
    Buffer_<int> init_context_length_;

    Buffer_<int> sequence_lengths_;  // current sequence length
    Buffer_<int> init_ctx_lens_;
    Buffer_<int> lora_mask_buf_;  // lora

    Buffer_<float>    sampled_logprobs_;
    Buffer_<uint32_t> sampled_indexes_;
    Buffer_<uint32_t> sampled_nums_;
    Buffer_<float>    h_sampled_logprobs_;
    Buffer_<uint32_t> h_sampled_indexes_;
    Buffer_<uint32_t> h_sampled_nums_;

    Buffer_<float> rope_theta_;

    // used by dynamic decoder
    Buffer_<int>  token_ids_buf_;  // all token IDs in [S, B], indexed using `step`
    Buffer_<bool> finished_buf_;
    Buffer_<int>  seq_limit_len_;

    // pinned buffers
    Buffer_<int> h_output_ids_;
    Buffer_<int> h_input_length_buf_;
    Buffer_<int> h_seq_limit_len_;

    // Host-side per-block KV pointers and optional FP4 scale pointers.
    Buffer_<int>       h_cu_block_counts_;
    Buffer_<uintptr_t> h_block_ptrs_;
    Buffer_<uintptr_t> h_scale_block_ptrs_;

    Buffer_<uint64_t> h_random_seed_;
    Buffer_<uint64_t> d_random_seed_;

    Tensor_<uint8_t> h_curand_state_;  // [n, sizeof(curandState_t)]
    Tensor_<uint8_t> d_curand_state_;

    std::array<BatchState, 3> states_{};

    BatchState* state_{};
    BatchState* back_{};
    BatchState* incoming_{};

    // hard limits for persistent buffers
    static constexpr int kMaxStopBadWordsLen = 32;
    static constexpr int kMaxEndIdsSize      = 32;

    std::thread internal_thread_;

    bool            enable_metrics_;
    ScheduleMetrics schedule_metrics_;
    std::mutex      metrics_mutex_;
};

using Engine = LlamaBatch;

}  // namespace turbomind

