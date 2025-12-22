// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <memory>
#include <memory_resource>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>
#include <limits>

#include <cuda_runtime.h>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/comm/host_comm.h"

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/buffer.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/tensor.h"

#include "src/turbomind/macro.h"

#include "src/turbomind/engine/gateway.h"
#include "src/turbomind/engine/request.h"

#include "src/turbomind/kernels/decoding_kernels.h"
#include "src/turbomind/kernels/gemm/tuner/params.h"
#include "src/turbomind/kernels/sampling_topk_kernels.h"

#include "src/turbomind/models/llama/BlockManager.h"
#include "src/turbomind/models/llama/LlamaBatch.h"
#include "src/turbomind/models/llama/LlamaV2.h"
#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/llama/copy.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"

#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/metrics.h"
#include "src/turbomind/utils/constant.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/debug_utils.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/eagle_debug.h"
#include "lmdeploy/turbomind/kernels/speculative_decoding/kv_rewind_helper.h"

namespace turbomind {

void PrintDecodeTokens(
    const int* token_ids, int max_seq_len, int batch_sizse, cudaStream_t stream, const std::string& msg)
{
    // tokens in [S, B] layout
    std::vector<int> tokens(max_seq_len * batch_sizse);
    check_cuda_error(cudaMemcpyAsync(tokens.data(), token_ids, sizeof(int) * tokens.size(), cudaMemcpyDefault, stream));
    check_cuda_error(cudaStreamSynchronize(stream));

    printf("[%s] ", msg.c_str());
    for (int j = 0; j < max_seq_len; ++j) {
        printf("%5d ", j);
    }
    printf("\n");
    for (int i = 0; i < batch_sizse; ++i) {
        printf("[%s] ", msg.c_str());
        for (int j = 0; j < max_seq_len; ++j) {
            // std::cout << sb_tokens[j * batch_size + i] << " ";
            printf("%5d ", tokens[j * batch_sizse + i]);
        }
        printf("\n");
    }
}
void ClearState(BatchState& s)
{
    std::fill_n(s.requests.begin(), s.size, nullptr);
    std::fill_n(s.sequences.begin(), s.size, nullptr);
    std::fill_n(s.errors.begin(), s.size, 0);
    s.size = s.active_size = 0;
}

void DropEmbeddings(const Sequence& seq)
{
    int    seq_len = seq.tokens.size();
    int    num_emb = seq.input_embeddings.size();
    size_t sz      = num_emb;
    for (; sz >= 1; sz--) {
        if (seq.input_embedding_ranges[sz - 1].second <= seq_len) {
            break;
        }
    }
    // should we keep part of embedding?
    seq.input_embeddings.resize(sz);
    seq.input_embedding_ranges.resize(sz);
}

void LlamaBatch::DisableInvalidRequests(Requests& infer_reqs, Requests& kill_reqs)
{
    NvtxScope _("disable invalid");

    std::pmr::monotonic_buffer_resource    mbr;
    std::pmr::unordered_map<uint64_t, int> occur(&mbr);

    auto count = [&occur](const auto& reqs) {
        for (const auto& r : reqs) {
            ++occur[r->id];
        }
    };

    auto validate = [&](auto& reqs, const char* type) {
        for (const auto& r : reqs) {
            if (occur[r->id] > 1) {
                TM_LOG_ERROR("Skip conflicting %s request for ID %lu", type, r->id);
                r->ec = Request::kConflict;
            }
            if (param_.enable_prefix_caching) {
                if (r->session.step != 0) {
                    // Prefix caching is incompatible with interactive mode
                    TM_LOG_ERROR("Skip inconsistent %s request for ID %lu step %d", type, r->id, r->session.step);
                    r->ec = Request::kInconsistency;
                }
                else if (r->gen_cfg.output_logits == GenerationConfig::kAll
                         || r->gen_cfg.output_last_hidden_state == GenerationConfig::kAll) {
                    // Prefix caching is incompatible with outputting all tokens' logits or last_hidden_state
                    TM_LOG_ERROR("Skip inconsistent %s request for ID %lu. It cannot output logits or "
                                 "last_hidden_states for all tokens",
                                 type,
                                 r->id);
                    r->ec = Request::kInconsistency;
                }
            }
        }
    };

    // Current batch
    for (int i = 0; i < state_->size; ++i) {
        if (state_->requests[i]) {
            ++occur[state_->requests[i]->id];
        }
    }

    count(kill_reqs);
    count(infer_reqs);

    validate(kill_reqs, "kill");
    validate(infer_reqs, "infer");

    // New requests that never get a chance to start
    for (auto& r : infer_reqs) {
        if (r && r->cancel_flag.load(std::memory_order_acquire) == -1) {
            r->ec = Request::kCancel;
        }
    }
}

void LlamaBatch::FindCanceledIndices(std::vector<int>& indices)
{
    for (int i = 0; i < state_->size; ++i) {  // current batch
        const auto& r = state_->requests[i];
        if (r && r->cancel_flag.load(std::memory_order_acquire) == -1) {
            indices.push_back(i);
        }
    }
}

void LlamaBatch::ProcessCancelRequests(std::vector<int>& indices, std::vector<Signal>& signals)
{
    int count = 0;

    for (const auto& i : indices) {
        if (auto& r = state_->requests[i]) {
            ++count;
            signals.push_back(Interrupt(i, true));
            // Interrupt should reset r
            FT_CHECK(!r);
        }
    }

    if (count) {
        // Still need this sync after `Interrupt`?
        check_cuda_error(cudaStreamSynchronize(stream_));
    }
}

void LlamaBatch::ProcessKillRequests(const Requests& kill_reqs, std::vector<Signal>& signals)
{
    for (auto& r : kill_reqs) {
        if (r) {
            int ec = r->ec;
            if (!ec) {
                if (!sequence_manager_->Erase(r->id)) {
                    ec = Request::kInvalid;
                }
            }
            signals.push_back([=] {
                if (r->end_cb) {
                    r->end_cb(ec);
                }
            });
        }
    }
}

void LlamaBatch::ProcessInferRequests(const Requests& reqs, std::vector<Signal>& signals)
{
    NvtxScope scope("infer_request");
    auto&     state = *incoming_;

    FT_CHECK(state.size == 0);
    FT_CHECK(state.active_size == 0);

    std::vector<int> existing_idx;

    int idx = 0;
    for (const auto& r : reqs) {

        if (tp_rank_ == 0) {
            TM_LOG_INFO("[ProcessInferRequests] Request for %llu received.", r->id);
        }

        if (r->ec) {
            signals.push_back([r] { UpdateState(*r, r->ec, 0); });
            continue;
        }

        const int input_length = r->inputs.at("input_ids").shape(0);

        if (input_length > session_len_) {
            signals.push_back([r] { UpdateState(*r, Request::kTooLong, 0); });
            continue;
        }

        auto ptr = r->session.start_flag ? sequence_manager_->Create(r->id) : sequence_manager_->Get(r->id);
        if (!ptr) {
            signals.push_back([r] { UpdateState(*r, Request::kInvalid, 0); });
            continue;
        }

        const int step = [&] {
            int s = r->session.step;
            if (s < 0) {
                s = ptr->tokens.size();
            }
            else if (s > ptr->tokens.size()) {
                if (tp_rank_ == 0) {
                    TM_LOG_WARNING("[ProcessInferRequests] Skipping invalid step (%d) setting for ID %lu", s, ptr->id);
                }
                s = ptr->tokens.size();
            }
            return s;
        }();

        if (step + input_length > session_len_) {
            signals.push_back([r] { UpdateState(*r, Request::kTooLong, 0); });
            continue;
        }

        FT_CHECK(!state.requests[idx]);

        state.requests[idx]  = r;
        state.sequences[idx] = ptr;

        auto& seq = *state.sequences[idx];

        if (!param_.enable_prefix_caching && step < seq.tokens.size()) {
            // resize sequence tokens to match step
            seq.tokens.resize(step);
            seq.cache_len = std::min(seq.cache_len, step);
            DropEmbeddings(seq);
        }

        const int* input_ids = r->inputs.at("input_ids").data<int>();

        {
            // `output_ids` contains all token ids of the sequences
            const auto output_ids_base = state.output_ids.data() + session_len_ * idx;
            auto       d_output_ids    = output_ids_base;
            auto       h_output_ids    = r->output_ids.data();
            // copy history tokens
            if (!seq.tokens.empty()) {
                d_output_ids = core::Copy(seq.tokens.data(), seq.tokens.size(), d_output_ids);
                h_output_ids = std::copy_n(seq.tokens.data(), seq.tokens.size(), h_output_ids);
            }

            // copy input tokens
            if (input_length) {
                d_output_ids = core::Copy(input_ids, input_length, d_output_ids);
                h_output_ids = std::copy_n(input_ids, input_length, h_output_ids);
            }

            // total context length (history + input)
            state.h_prompt_length[idx]  = d_output_ids - output_ids_base;
            state.h_context_length[idx] = d_output_ids - output_ids_base;
            state.h_finished[idx]       = false;
        }

        // copy input tokens to prompt for prefix matching
        if (input_length && r->session.start_flag && !r->inputs.contains("input_embedding_ranges")) {
            // TODO: truncate prompt to enable prefix caching for VLM
            seq.prompt.resize(input_length);
            std::copy_n(input_ids, input_length, seq.prompt.data());
        }

        const int elem_size = byte_size(data_type_);

        // copy input embeddings
        if (r->inputs.contains("input_embedding_ranges")) {
            const auto& range_tensor = r->inputs.at("input_embedding_ranges");
            const auto& emb_tensor   = r->inputs.at("input_embeddings");
            const int*  ranges       = range_tensor.data<int>();

            auto check_embeddings = [&](int& num_valid_embeddings) {
                if (range_tensor.ndim() != 3 || range_tensor.shape(2) % 2 != 0) {
                    return false;
                }
                int embedding_count  = range_tensor.shape(1);
                int embedding_length = 0;
                int pre_end          = -1;

                for (size_t i = 0; i < embedding_count; i++) {
                    int begin = ranges[i * 2];
                    int end   = ranges[i * 2 + 1];
                    embedding_length += (end - begin);
                    if (begin < 0 || end < 0) {
                        break;
                    }
                    if (begin >= end || end > input_length || begin < pre_end
                        || embedding_length * model_->hidden_units_ * elem_size > emb_tensor.shape(1)) {
                        return false;
                    }
                    pre_end              = end;
                    num_valid_embeddings = i + 1;
                }
                return true;
            };

            int num_valid_embeddings = 0;
            if (!check_embeddings(num_valid_embeddings)) {
                TM_LOG_WARNING("[ImageFeature] Skip invalid input embeddings, id = %ld, input_length = %d",
                               (long)seq.id,
                               input_length);
            }
            else {
                const std::byte* emb_tensor_ptr = (const std::byte*)emb_tensor.raw_data();
                for (size_t i = 0; i < num_valid_embeddings; i++) {
                    int    begin = ranges[i * 2];
                    int    end   = ranges[i * 2 + 1];
                    size_t count = (end - begin) * model_->hidden_units_ * elem_size;
                    seq.input_embeddings.emplace_back(emb_tensor_ptr, emb_tensor_ptr + count);
                    seq.input_embedding_ranges.emplace_back(begin + seq.tokens.size(), end + seq.tokens.size());
                    emb_tensor_ptr += count;
                }
            }
        }

        // copy mrope input meta
        if (model_->attn_param_.rope.type == RopeType::kMrope) {
            TM_CHECK(r->session.start_flag) << "Mrope doesn't support interactive chat";
            if (r->inputs.count("mrope_position_ids")) {
                if (turbomind::isEagleDebugEnabled()
                    && turbomind::isEnvVarEnabled("LMDEPLOY_EAGLE_COPY_DEBUG")) {
                    TM_LOG_WARNING(
                        "[LlamaBatch][EAGLE][CopySITE:mrope_pos_ids] idx=%d a=%p b=%p n=%d",
                        idx,
                        r->inputs.at("mrope_position_ids").data<int>(),
                        state.mrope.position_ids.data() + idx * state.mrope.position_ids.shape(1),
                        input_length * 3);
                    TM_LOG_WARNING(
                        "[LlamaBatch][EAGLE][CopySITE:mrope_pos_delta] idx=%d a=%p b=%p n=%d",
                        idx,
                        r->inputs.at("mrope_position_delta").data<int>(),
                        state.mrope.position_delta.data() + idx,
                        1);
                    TM_LOG_WARNING(
                        "[LlamaBatch][EAGLE][CopySITE:mrope_length] idx=%d a=%p b=%p n=%d",
                        idx,
                        r->inputs.at("mrope_length").data<int>(),
                        state.mrope.length.data() + idx,
                        1);
                }
                core::Copy(r->inputs.at("mrope_position_ids").data<int>(),
                           input_length * 3,
                           state.mrope.position_ids.data() + idx * state.mrope.position_ids.shape(1));
                core::Copy(
                    r->inputs.at("mrope_position_delta").data<int>(), 1, state.mrope.position_delta.data() + idx);
                core::Copy(r->inputs.at("mrope_length").data<int>(), 1, state.mrope.length.data() + idx);
            }
            else {
                check_cuda_error(cudaMemsetAsync(state.mrope.length.data() + idx, 0, sizeof(int), stream_));
            }
        }

        const int max_new_tokens = state.requests[idx]->gen_cfg.max_new_tokens;
        state.seq_len_limit[idx] = state.h_context_length[idx] + max_new_tokens;
        // `length_criterion` sets finish flag when step >= seq_limit_len, however when step == seq_limit_len
        // the actual sequence length is seq_limit_len + 1, hence seq_limit_len must truncated to session_len - 1
        if (state.seq_len_limit[idx] >= session_len_) {
            state.seq_len_limit[idx] = session_len_ - 1;
            if (tp_rank_ == 0) {
                const int trunc_output_len = state.seq_len_limit[idx] - state.h_context_length[idx];
                TM_LOG_WARNING(
                    "[ProcessInferRequests] [%ld] total sequence length (%d + %d) exceeds `session_len` (%d), `max_new_tokens` is truncated to %d",
                    (long)seq.id,
                    state.h_context_length[idx],
                    max_new_tokens,
                    (int)session_len_,
                    trunc_output_len);
            }
        }

        // compute rope scaling factor
        if (r->session.start_flag) {
            seq.rope_theta = model_->attn_param_.rope.base;
            if (model_->attn_param_.rope.type == RopeType::kDynamic) {
                auto scaling_factor = model_->attn_param_.rope.factor;
                if (scaling_factor >= 1.f) {  // infer by current context length
                    auto max_seq_len = state.h_context_length[idx];
                    auto max_pos_emb = model_->attn_param_.rope.max_position_embeddings;
                    if (max_seq_len > max_pos_emb) {
                        scaling_factor = scaling_factor * max_seq_len / max_pos_emb - (scaling_factor - 1);
                        float rope_dim = model_->attn_param_.rope.dim;
                        seq.rope_theta *= powf(scaling_factor, rope_dim / (rope_dim - 2.f));
                        TM_LOG_INFO("[ProcessInferRequests] %ld rope_scaling_factor: %f, rope_theta = %f",
                                    (long)seq.id,
                                    scaling_factor,
                                    seq.rope_theta);
                    }
                }
            }
        }
        state.h_rope_theta[idx] = seq.rope_theta;

        if (r->session.start_flag) {
            // prepare to initialize random state for new sequence
            h_random_seed_[idx] = r->gen_cfg.random_seed;
        }
        else {
            // Recover device states if not a new sequence
            ((curandState_t*)h_curand_state_.data())[existing_idx.size()] = *(curandState_t*)seq.random_state.data();
            existing_idx.push_back(idx);
        }

        // increment pointer
        idx++;
    }

    state.size = idx;

    // when there are new sequences
    if (state.size != existing_idx.size()) {
        // copy random seeds to device
        Copy(h_random_seed_, state.size, d_random_seed_);
        // initialize random states
        invokeCurandBatchInitialize(
            (curandState_t*)state.curand_state.data(), state.size, d_random_seed_.data(), stream_);
        sync_check_cuda_error();
    }

    if (!existing_idx.empty()) {
        // copy existing curand states to device
        core::Copy((curandState_t*)h_curand_state_.data(), existing_idx.size(), (curandState_t*)h_curand_state_.data());
        // insert the states to their correct positions in the batch
        IndexedCopy({},
                    existing_idx,
                    std::tuple{(curandState_t*)d_curand_state_.data(), (curandState_t*)state.curand_state.data(), 1});
    }
}

int LlamaBatch::AdjustMaxInputCount(GenerationState&                    g,
                                    const std::vector<const Sequence*>& sequences,
                                    const std::vector<int>&             context_length)
{
    int input_count = 0;
    for (int i = 0; i < sequences.size(); ++i) {
        input_count += context_length[i] - sequences[i]->cache_len;
    }
    const int batch_size = sequences.size();
    input_count -= batch_size;

    // min tokens per iter for satisfying max prefill iters constraint
    input_count = (input_count + max_prefill_iters_ - 1) / max_prefill_iters_;

    if (g.min_input_count.empty()) {
        g.min_input_count.resize(max_prefill_iters_);
    }
    g.min_input_count.pop_front();
    g.min_input_count.push_back(input_count);
    /// TODO: sub-optimal when there are inactive sequences due to memory constraint
    for (auto& x : g.min_input_count) {
        x = std::max(x, input_count);
    }

    // Enlarge to satisfy `max_prefill_iters_`
    input_count = std::max(g.min_input_count.front() + batch_size, num_tokens_per_iter_);
    // Clamp to conform memory constraint
    input_count = std::min(input_count, max_forward_token_num_);

    return input_count;
}

void LlamaBatch::Initialize(GenerationState& g)
{
    NvtxScope                                scope("initialize");
    std::vector<const Sequence*>             sequences;
    std::vector<Sequence::Status>            status;
    std::vector<uint64_t>                    priorities;
    std::vector<int>                         context_lengths;
    std::vector<std::pair<BatchState*, int>> coords;

    // count the holes introduced by finished requests in from previous iteration or stop requests from
    // current iteration
    int holes{};
    int active_holes{};
    for (int i = 0; i < state_->size; ++i) {
        if (!state_->requests[i]) {
            ++holes;
            if (i < state_->active_size) {
                ++active_holes;
            }
        }
    }

    auto process = [&](BatchState* state) {
        for (int i = 0; i < state->size; ++i) {
            if (auto& r = state->requests[i]) {
                if (state == incoming_) {
                    // Alloc stable slot for new request
                    if (free_eagle_slots_.empty()) {
                        TM_LOG_ERROR("[LlamaBatch] No free Eagle slots available. Dropping request.");
                        // Handle error? LlamaBatch doesn't easily support drop here.
                        // Assuming max_batch_size matches pool size.
                        state->eagle_slots[i] = -1; // Invalid
                    } else {
                        state->eagle_slots[i] = free_eagle_slots_.back();
                        free_eagle_slots_.pop_back();
                    }
                }
                sequences.push_back(state->sequences[i]);
                status.push_back(state->sequences[i]->status);
                priorities.push_back(r->unique_id);
                context_lengths.push_back(state->h_context_length[i]);
                coords.emplace_back(state, i);
            }
        }
    };

    process(state_);
    process(incoming_);

    auto adjust = [this, &g](const Sequences& sequences, const std::vector<int>& context_length) -> int {
        return AdjustMaxInputCount(g, sequences, context_length);
    };

    // TM_LOG_INFO("max_input_count %d", max_input_count);
    auto outcome = sequence_manager_->Materialize(sequences, context_lengths, priorities, 1, adjust);

    if (outcome.allocation || outcome.swap_in || outcome.swap_out) {
        dbg(outcome);
    }

    bool exchange = outcome.swap_in + outcome.swap_out > 0;

    std::vector<int> idxs(sequences.size());
    std::iota(idxs.begin(), idxs.end(), 0);

    if (exchange || holes || incoming_->size) {
        // put active ones first
        auto active_end = std::stable_partition(idxs.begin(), idxs.end(), [&](int idx) {
            return sequences[idx]->status == Sequence::kActive;  // current status
        });

        // all blocks are not enough to hold a single sequence
        if (!sequences.empty()) {
            FT_CHECK_WITH_INFO(active_end != idxs.begin(), "No enough blocks.");
        }

        // move the partial seq to the back
        auto partial_beg = std::stable_partition(idxs.begin(), active_end, [&](int i) {
            return sequences[i]->cache_len + sequences[i]->input_length == context_lengths[i];
        });
        FT_CHECK(active_end - partial_beg <= 1);

        auto swapin_beg = std::stable_partition(idxs.begin(), partial_beg, [&](int i) {
            return status[i] == Sequence::kActive;  // past status
        });

        // sort swap-ins according to input length
        if (swapin_beg != partial_beg) {
            std::stable_sort(swapin_beg, partial_beg, [&](int i, int j) {
                return sequences[i]->input_length < sequences[j]->input_length;
            });
        }

        // Free slots for inactive/dropped sequences
        for (size_t i = 0; i < sequences.size(); ++i) {
            if (sequences[i]->status != Sequence::kActive) {
                auto [src_state, src_idx] = coords[i];
                // Check if we allocated a slot (might be -1 if failed or not init?)
                if (static_cast<size_t>(src_idx) < src_state->eagle_slots.size()) {
                    int slot = src_state->eagle_slots[src_idx];
                    if (slot >= 0) {
                        free_eagle_slots_.push_back(slot);
                        src_state->eagle_slots[src_idx] = -1;
                    }
                }
            }
        }

        // Copy sequence states to back buffer
        FT_CHECK(back_->size == 0 && back_->active_size == 0);
        std::vector<std::tuple<BatchState*, BatchState*, int, int>> cpys;
        for (const auto& i : idxs) {
            auto& s = *sequences[i];
            if (s.status == Sequence::kActive) {
                ++back_->active_size;
            }
            cpys.emplace_back(coords[i].first, back_, coords[i].second, back_->size++);
        }
        CopyState(cpys);
        // Swap the buffers
        std::swap(state_, back_);

        ClearState(*back_);
        ClearState(*incoming_);
    }

    FT_CHECK(state_->size <= max_batch_size_);

    /// Update block ptrs when there were
    //  1. swap-in or swap-out
    //  2. holes in the active buffer
    //  3. new allocations (for existing active sequences)
    if (exchange || active_holes || outcome.allocation) {
        // Prepare intermediate buffers
        h_cu_block_counts_[0] = 0;

        auto block_ptrs = h_block_ptrs_.data();

        const int batch_size = state_->active_size;

        for (int i = 0; i < batch_size; ++i) {
            const auto& seq = *state_->sequences[i];

            // cumulative num of blocks
            h_cu_block_counts_[i + 1] = h_cu_block_counts_[i] + seq.blocks.size();

            block_ptrs = std::transform(seq.blocks.cbegin(), seq.blocks.cend(), block_ptrs, [&](int block_id) {
                return reinterpret_cast<uintptr_t>(sequence_manager_->GetBlockPtr(block_id));
            });
        }

        static_assert(sizeof(uintptr_t) == sizeof(void*));

        Copy(h_cu_block_counts_, batch_size + 1, cu_block_counts_);
        Copy(h_block_ptrs_, h_cu_block_counts_[batch_size], block_ptrs_);
    }

    const int batch_size = state_->active_size;

    // check if the last sequence is partial
    int partial     = 0;
    int partial_len = -1;
    if (state_->active_size) {
        const int i = state_->active_size - 1;
        partial = state_->sequences[i]->cache_len + state_->sequences[i]->input_length != state_->h_context_length[i];
        if (partial) {
            // backup full context length of partial
            partial_len = state_->h_context_length[i];
            // replace with partial context length
            state_->h_context_length[i] = state_->sequences[i]->cache_len + state_->sequences[i]->input_length;
        }
    }

    const int max_context_len =
        *std::max_element(state_->h_context_length.data(), state_->h_context_length.data() + batch_size);

    std::vector<uint64_t> unique_ids(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        unique_ids[i] = state_->requests[i]->unique_id;
    }

    // Real-time context length that will change during generation
    Copy_(state_->h_context_length, batch_size, context_length_buf_);
    Copy_(state_->h_finished, batch_size, finished_buf_);
    Copy_(state_->h_rope_theta, batch_size, rope_theta_);

    bool skip_init_sampling = std::equal(g.unique_ids.begin(),  //
                                         g.unique_ids.end() - g.partial,
                                         unique_ids.begin(),
                                         unique_ids.end() - partial);

    g.partial                = partial;
    g.partial_context_legnth = partial_len;
    g.unique_ids             = std::move(unique_ids);
    g.finished_count         = 0;
    g.skip_init_sampling     = skip_init_sampling;

    // TM_LOG_ERROR("[Initialize] batch size: %d, active size: %d", state_->size, state_->active_size);

    if (!skip_init_sampling) {
        g.max_init_ctx_len = max_context_len;
        g.step             = max_context_len;
    }
}

void LlamaBatch::CopyState(const std::vector<std::tuple<BatchState*, BatchState*, int, int>>& desc)
{
    if (desc.empty()) {
        return;
    }

    std::vector<int> idxs(desc.size());
    std::iota(idxs.begin(), idxs.end(), 0);

    std::sort(idxs.begin(), idxs.end(), [&](int i, int j) { return desc[i] < desc[j]; });

    auto get_signature = [&](int i) -> std::pair<BatchState*, BatchState*> {
        return std::make_pair(std::get<0>(desc[idxs[i]]), std::get<1>(desc[idxs[i]]));
    };

    std::vector<int> offsets;
    auto             current = get_signature(0);
    offsets.push_back(0);
    for (int i = 0; i < idxs.size(); ++i) {
        if (auto signature = get_signature(i); signature != current) {
            current = signature;
            offsets.push_back(i);
        }
    }
    offsets.push_back(idxs.size());

    for (int bi = 1; bi < offsets.size(); ++bi) {
        int beg = offsets[bi - 1];
        int end = offsets[bi];

        if (beg == end) {
            continue;
        }

        auto [s, d] = get_signature(beg);

        std::vector<int> s_idx;
        std::vector<int> d_idx;
        for (int i = beg; i < end; ++i) {
            s_idx.push_back(std::get<2>(desc[idxs[i]]));
            d_idx.push_back(std::get<3>(desc[idxs[i]]));
            // Copy eagle_slots
            d->eagle_slots[d_idx.back()] = s->eagle_slots[s_idx.back()];
        }

        IndexedCopy(s_idx,
                    d_idx,
                    std::tuple{s->output_ids.data(), d->output_ids.data(), session_len_},
                    std::tuple{(curandState_t*)s->curand_state.data(), (curandState_t*)d->curand_state.data(), 1});

        if (model_->attn_param_.rope.type == RopeType::kMrope) {
            IndexedCopy(s_idx,
                        d_idx,
                        std::tuple{s->mrope.position_ids.data(),
                                   d->mrope.position_ids.data(),
                                   (int)s->mrope.position_ids.shape(1)},
                        std::tuple{s->mrope.position_delta.data(), d->mrope.position_delta.data(), 1},
                        std::tuple{s->mrope.length.data(), d->mrope.length.data(), 1});
        }
    }

    for (const auto& [s, d, si, di] : desc) {
        d->h_prompt_length[di]  = s->h_prompt_length[si];
        d->h_context_length[di] = s->h_context_length[si];
        d->h_finished[di]       = s->h_finished[si];
        d->h_rope_theta[di]     = s->h_rope_theta[si];
        d->seq_len_limit[di]    = s->seq_len_limit[si];
        d->sequences[di]        = s->sequences[si];
        d->requests[di]         = s->requests[si];
    }
}

void LlamaBatch::AllocateBuffer(ssize_t batch_size, ssize_t session_len, int cache_block_seq_len)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    const ssize_t batchxbeam = batch_size;

    const ssize_t hidden_units      = model_->hidden_units_;
    const ssize_t vocab_size        = model_->vocab_size_padded_;
    const ssize_t head_dim          = model_->size_per_head_;
    const ssize_t local_kv_head_num = model_->local_kv_head_num_;
    // +1 padding, BlockIterator does not use predicate
    const ssize_t max_batch_block_count =
        batch_size * ((session_len + cache_block_seq_len - 1) / cache_block_seq_len) + 1;

    input_ids_buf_ = {max_forward_token_num_, kDEVICE};

    decoder_output_buf_ = {{batchxbeam, hidden_units}, data_type_, kDEVICE};

    input_length_buf_    = {batchxbeam, kDEVICE};
    context_length_buf_  = {batchxbeam, kDEVICE};
    init_context_length_ = {batchxbeam, kDEVICE};

    sequence_lengths_ = {batchxbeam, kDEVICE};

    cu_block_counts_ = {batch_size + 1, kDEVICE};
    block_ptrs_      = {max_batch_block_count, kDEVICE};

    sampled_logprobs_ = {batchxbeam * kMaxLogProb, kDEVICE};
    sampled_indexes_  = {batchxbeam * kMaxLogProb, kDEVICE};
    sampled_nums_     = {batchxbeam, kDEVICE};

    token_ids_buf_ = {ssize_t(session_len * 2 * batchxbeam), kDEVICE};

    sampling_logits_ = {{(ssize_t)max_batch_size_, (ssize_t)model_->vocab_size_padded_}, kDEVICE};

    finished_buf_  = {(int)batchxbeam, kDEVICE};
    seq_limit_len_ = {batch_size, kDEVICE};

    rope_theta_ = {batch_size, kDEVICE};

    h_random_seed_ = {batch_size, kCPUpinned};
    Clear(h_random_seed_);

    d_random_seed_ = {batch_size, kDEVICE};
    Clear(d_random_seed_);

    h_curand_state_ = {{batch_size, sizeof(curandState_t)}, kCPUpinned};
    Clear(h_curand_state_.buffer());

    d_curand_state_ = {{batch_size, sizeof(curandState_t)}, kDEVICE};
    Clear(d_curand_state_.buffer());

    for (auto& s : states_) {
        s.output_ids = {{batch_size, session_len_}, kDEVICE};
        Clear(s.output_ids.buffer());

        s.curand_state = {{batch_size, sizeof(curandState_t)}, kDEVICE};
        Clear(s.curand_state.buffer());

        if (model_->attn_param_.rope.type == RopeType::kMrope) {
            s.mrope.position_ids   = {{batch_size, session_len_ * 3}, kDEVICE};
            s.mrope.position_delta = {batch_size, kDEVICE};
            s.mrope.length         = {batch_size, kDEVICE};
            Clear(s.mrope.position_delta);
            Clear(s.mrope.length);
        }
    }

    h_input_length_buf_ = {batch_size, kCPUpinned};
    h_cu_block_counts_  = {batch_size + 1, kCPUpinned};
    h_block_ptrs_       = {(ssize_t)max_batch_block_count, kCPUpinned};

    for (auto& s : states_) {
        s.h_prompt_length  = {batch_size, kCPUpinned};
        s.h_context_length = {batch_size, kCPUpinned};
        s.h_finished       = {batch_size * 2, kCPUpinned};
        s.h_finished       = {batch_size * 2, kCPUpinned};
        s.h_rope_theta     = {batch_size, kCPUpinned};
        s.eagle_slots.resize(batch_size);
    }

    h_seq_limit_len_ = {batch_size, kCPUpinned};
    std::fill_n(h_seq_limit_len_.data(), batch_size, 0);

    h_output_ids_ = {batch_size * session_len_, kCPUpinned};

    h_sampled_logprobs_ = {batch_size * kMaxLogProb, kCPUpinned};
    h_sampled_indexes_  = {batch_size * kMaxLogProb, kCPUpinned};
    h_sampled_nums_     = {batch_size, kCPUpinned};

    // EAGLE KV rewind persistent buffers. These are only used when multi-token
    // speculative decoding is enabled and the experimental flag is set.
    eagle_kv_rewind_lengths_ = {max_batch_size_, kDEVICE};
    eagle_kv_batch_slots_    = {max_batch_size_, kDEVICE};
    if (kv_max_blocks_per_seq_ > 0) {
        eagle_kv_block_tables_ = {ssize_t(max_batch_size_ * kv_max_blocks_per_seq_), kDEVICE};
    }

    // Per-request EOS id for EAGLE acceptance (first eos id or -1).
    // These live on device memory; initialize them via cudaMemcpyAsync
    // rather than std::fill_n on raw device pointers.
    eagle_end_ids_             = {max_batch_size_, kDEVICE};
    eagle_posterior_thresholds_ = {max_batch_size_, kDEVICE};
    eagle_posterior_alphas_     = {max_batch_size_, kDEVICE};
    eagle_temperatures_         = {max_batch_size_, kDEVICE};

    // Multi-token per-slot kill switch, initially all zero (multi-token
    // enabled). Length is fixed to `max_batch_size_`.
    eagle_disable_multitoken_slot_.assign(max_batch_size_, 0);

    {
        // Host-side staging buffers for device initialization.
        std::vector<int>   h_end_ids(max_batch_size_, -1);
        std::vector<float> h_thresholds(max_batch_size_, 1.0f);
        std::vector<float> h_alphas(max_batch_size_, 1.0f);
        std::vector<float> h_temps(max_batch_size_, 1.0f);

        check_cuda_error(cudaMemcpyAsync(eagle_end_ids_.raw_data(),
                                         h_end_ids.data(),
                                         sizeof(int) * max_batch_size_,
                                         cudaMemcpyHostToDevice,
                                         stream_));
        check_cuda_error(cudaMemcpyAsync(eagle_posterior_thresholds_.raw_data(),
                                         h_thresholds.data(),
                                         sizeof(float) * max_batch_size_,
                                         cudaMemcpyHostToDevice,
                                         stream_));
        check_cuda_error(cudaMemcpyAsync(eagle_posterior_alphas_.raw_data(),
                                         h_alphas.data(),
                                         sizeof(float) * max_batch_size_,
                                         cudaMemcpyHostToDevice,
                                         stream_));
        check_cuda_error(cudaMemcpyAsync(eagle_temperatures_.raw_data(),
                                         h_temps.data(),
                                         sizeof(float) * max_batch_size_,
                                         cudaMemcpyHostToDevice,
                                         stream_));
    }
}

void LlamaBatch::AllocSymmBuffers()
{
    const ssize_t hidden_units      = model_->hidden_units_;
    const ssize_t vocab_size_padded = model_->vocab_size_padded_;

    // Native comm fuses allreduce & rmsnorm in token granularity
    TM_CHECK(max_forward_token_num_ % tp_size_ == 0) << max_forward_token_num_ << " vs " << tp_size_;

    symm_hidden_states_buf_ = {{max_forward_token_num_ * param_.attn_dp_size, hidden_units}, data_type_, symm_alloc_};
    symm_logits_buf_        = {{max_batch_size_, vocab_size_padded}, data_type_, symm_alloc_};

    // for context parallel, we use symm_alloc_ and both prefill and decode stage have reduce process
    // w/o context parallel, we use common alloc and only decode stage has reduce process
    // perhaps it would be more appropriate to put this buffer in the unified_attention_layer.
    Allocator     alloc          = param_.attn_cp_size > 1 ? symm_alloc_ : core::Context::device_alloc();
    const ssize_t attn_ws_tokens = param_.attn_cp_size > 1 ?
                                       UnifiedAttentionLayer::kMaxWorkspaceTokens + max_forward_token_num_ :
                                       UnifiedAttentionLayer::kMaxWorkspaceTokens;
    symm_partial_ML_             = {{param_.attn_cp_size, attn_ws_tokens, (int)model_->local_head_num_, 2}, alloc};
}

void LlamaBatch::FreeSymmBuffers()
{
    symm_hidden_states_buf_ = {};
    symm_logits_buf_        = {};

    symm_partial_ML_ = {};
}

LlamaBatch::~LlamaBatch()
{
    TM_LOG_DEBUG("~LlamaBatch()");

    if (internal_thread_.joinable()) {
        internal_thread_.join();
    }

    // The dtor maybe called from unknown thread, set device id before CUDA calls
    cudaSetDevice(device_id_);
    cudaStreamSynchronize(stream_);

    model_.reset();
    FreeBufferAndKVCache();
    context_.reset();  // This destroy all objects in context except for `stream`
}

LlamaBatch::LlamaBatch(DataType                 data_type,
                       const EngineParam&       param,
                       std::unique_ptr<LlamaV2> model,  // ! This is moved
                       std::shared_ptr<Context> ctx,
                       std::shared_ptr<Gateway> gateway,
                       int                      device_id,
                       int                      dp_rank):
    param_(param),
    gateway_(gateway),
    max_batch_size_(param.max_batch_size),
    max_forward_token_num_(param.max_forward_token_num),
    max_context_token_num_(param.max_context_token_num),
    num_tokens_per_iter_(param.num_tokens_per_iter),
    max_prefill_iters_(param.max_prefill_iters),
    device_id_(device_id),
    dp_rank_(dp_rank),
    tp_size_(model->tp_size_),
    tp_rank_(model->tp_rank_),
    data_type_(data_type),
    debug_(isDebug()),
    stream_(ctx->stream),
    context_(std::move(ctx)),
    model_(std::move(model)),
    comm_(context_->comm),
    session_len_(param.session_len)
{

    symm_alloc_ = core::SimpleAllocator::Create([this](ssize_t size) { return SymmAlloc(size, true); },
                                                [this](void* p, ssize_t size) { return SymmFree(p, size, true); },
                                                kDEVICE);

    // Init free slot pool
    for (int i = 0; i < max_batch_size_; ++i) {
        free_eagle_slots_.push_back(max_batch_size_ - 1 - i);
    }

    InitializeBufferAndKVCache();

    // Wait for allocations
    check_cuda_error(cudaStreamSynchronize(stream_));

    UpdateMetrics();
}

void LlamaBatch::disableEagleMultitokenForSlot(int slot, const char* reason)
{
    if (slot < 0 || slot >= static_cast<int>(eagle_disable_multitoken_slot_.size())) {
        return;
    }

    if (eagle_disable_multitoken_slot_[slot]) {
        return;
    }

    eagle_disable_multitoken_slot_[slot] = 1;

    if (tp_rank_ == 0) {
        TM_LOG_WARNING(
            "[LlamaBatch][EAGLE][fallback] step=%d seq=%d: %s; disabling multi-token for this slot",
            eagle_last_step_,
            slot,
            reason ? reason : "no reason provided");
    }
}

bool LlamaBatch::isEagleMultiTokenStepEnabled(const GenerationState& g) const
{
    if (!param_.enable_speculative_decoding) {
        return false;
    }

    // Multi-token speculative steps currently assume a single-GPU
    // target model without tensor / data / pipeline parallel splits
    // along the attention stack. When any of these parallel modes are
    // active we conservatively disable multi-token behaviour and fall
    // back to single-token EAGLE semantics for this engine.
    if (tp_size_ != 1) {
        static bool logged_tp_warning = false;
        if (!logged_tp_warning && tp_rank_ == 0) {
            logged_tp_warning = true;
            TM_LOG_WARNING(
                "[LlamaBatch][EAGLE][fallback] Multi-token speculative decoding is only supported for tp_size==1; "
                "multi-token behaviour will be disabled for this engine run.");
        }
        return false;
    }
    if (param_.attn_dp_size > 1 || param_.outer_dp_size > 1 || param_.attn_cp_size > 1) {
        static bool logged_dp_pp_warning = false;
        if (!logged_dp_pp_warning && tp_rank_ == 0) {
            logged_dp_pp_warning = true;
            TM_LOG_WARNING(
                "[LlamaBatch][EAGLE][fallback] Multi-token speculative decoding is only supported for "
                "attn_dp_size==1, outer_dp_size==1, attn_cp_size==1; "
                "multi-token behaviour will be disabled for this engine run.");
        }
        return false;
    }
    if (g.partial != 0) {
        return false;
    }
    // Require the engine to have a multi-token speculative budget; when the
    // per-step engine limit is <= 1 we treat EAGLE as single-token only.
    if (model_->eagleMaxEngineTokensPerStep() <= 1) {
        return false;
    }
    return true;
}

bool LlamaBatch::isEagleMultiTokenSlotEnabled(int slot) const
{
    if (slot < 0 || slot >= static_cast<int>(eagle_disable_multitoken_slot_.size())) {
        return true;
    }

    // Per-slot kill switch driven by runtime invariants (mismatch, EOS in
    // extra tokens, geometry issues, etc.). Once a slot is killed it never
    // re-enters multi-token behaviour for the remainder of the request.
    return eagle_disable_multitoken_slot_[slot] == 0;
}

void LlamaBatch::collectEagleStepHostState(const GenerationState& g,
                                           int                    batch_size,
                                           std::vector<int>&      h_token_ids,
                                           std::vector<bool>&     h_finished_slots)
{
    if (batch_size <= 0) {
        h_token_ids.clear();
        h_finished_slots.clear();
        return;
    }

    if (static_cast<int>(h_token_ids.size()) < batch_size) {
        h_token_ids.resize(batch_size);
        h_finished_slots.resize(batch_size);
    }

    // When g.step == 0 there is no "previous" base token row to copy.
    // In that case, treat the token ids as unknown for this step and
    // skip the device-to-host copy to avoid reading out-of-bounds
    // from token_ids_buf_.
    if (g.step > 0) {
        if (isEagleDebugEnabled() && tp_rank_ == 0
            && turbomind::isEnvVarEnabled("LMDEPLOY_EAGLE_COPY_DEBUG")) {
            TM_LOG_WARNING(
                "[LlamaBatch][EAGLE][CopySITE:collect_step_tokens] step=%d batch=%d src_row=%d a=%p b=%p n=%d",
                g.step,
                batch_size,
                g.step - 1,
                token_ids_buf_.data() + (g.step - 1) * batch_size,
                h_token_ids.data(),
                batch_size);
        }
        core::Copy(token_ids_buf_.data() + (g.step - 1) * batch_size, batch_size, h_token_ids.data());
    }
    else {
        std::fill_n(h_token_ids.data(), batch_size, -1);
    }
    // finished_buf_ is a device buffer of bools; std::vector<bool> has a
    // specialized representation, so we cannot copy into it directly via
    // core::Copy. Use a temporary byte buffer and then map to bools.
    static std::vector<uint8_t> h_finished_tmp;
    if (static_cast<int>(h_finished_tmp.size()) < batch_size) {
        h_finished_tmp.resize(batch_size);
    }
    check_cuda_error(cudaMemcpyAsync(h_finished_tmp.data(),
                                     finished_buf_.data(),
                                     static_cast<size_t>(batch_size) * sizeof(bool),
                                     cudaMemcpyDeviceToHost,
                                     stream_));
    check_cuda_error(cudaStreamSynchronize(stream_));
    for (int i = 0; i < batch_size; ++i) {
        h_finished_slots[i] = (h_finished_tmp[i] != 0);
    }
}

void LlamaBatch::updateEagleMetricsAndKVLengths(const GenerationState&   g,
                                                const std::vector<int>&  h_token_ids,
                                                const std::vector<bool>& h_finished_slots,
                                                int                      batch_size,
                                                int                      max_path_len,
                                                int                      eagle_tokens_per_seq,
                                                const std::vector<int>&  eagle_accepted_lens,
                                                const std::vector<int>&  eagle_accepted_tokens,
                                                std::vector<int>&        kv_draft_lengths,
                                                std::vector<int>&        kv_accepted_lengths)
{
    if (!param_.enable_metrics || eagle_accepted_lens.empty() || eagle_tokens_per_seq <= 0 || batch_size <= 0) {
        return;
    }

    const bool target_tree_active =
        model_ && model_->isTargetTreeDecodeActiveStep();

    int step_draft_total    = 0;
    int step_accepted_total = 0;

    if (static_cast<int>(kv_draft_lengths.size()) < batch_size) {
        kv_draft_lengths.resize(batch_size, 0);
        kv_accepted_lengths.resize(batch_size, 0);
    }

    for (int i = 0; i < batch_size; ++i) {
        auto& req = state_->requests[i];
        if (!req || !req->metrics) {
            continue;
        }

        const bool multi_token_slot_enabled = isEagleMultiTokenSlotEnabled(i);
        const bool finished                 = (i < static_cast<int>(h_finished_slots.size())) ? h_finished_slots[i]
                                                                                            : false;

        // Raw acceptance from EAGLE (prior to any alignment with
        // DynamicDecode). We keep this for metrics so that aggregate
        // acceptance can be inspected even when we conservatively
        // clamp or reject tokens at the integration boundary.
        int raw_accepted_len   = 0;
        int raw_accepted_token = -1;

        // Effective acceptance used for KV length and multi-token
        // advancement decisions. This may be stricter than the raw
        // acceptance (e.g. when DynamicDecode chooses a different
        // first token).
        int accepted_len   = 0;
        int accepted_token = -1;

        if (i < static_cast<int>(eagle_accepted_lens.size())) {
            raw_accepted_len = eagle_accepted_lens[i];
            accepted_len     = raw_accepted_len;
            if (accepted_len > 0 && max_path_len > 0
                && i * max_path_len < static_cast<int>(eagle_accepted_tokens.size())) {
                raw_accepted_token = eagle_accepted_tokens[i * max_path_len];
                accepted_token     = raw_accepted_token;
            }
        }

        // If DynamicDecode has already marked this slot finished for the
        // current step, treat any multi-token acceptance as a strong invariant
        // violation and permanently disable multi-token for this request. For
        // metrics/KV purposes, clamp to a single-token acceptance.
        if (finished && accepted_len > 1) {
            disableEagleMultitokenForSlot(i, "finished slot but EAGLE reported accepted_len > 1");
            accepted_len = 1;
        }

        if (accepted_len > 0 && accepted_token != h_token_ids[i]) {
            // Keep raw acceptance for metrics, but log the divergence
            // between the first accepted token and DynamicDecode's
            // committed token. This can happen when sampling layers
            // (e.g. penalties, guidance) adjust logits between the
            // point where `target_tokens` were captured and the final
            // decode decision.
            TM_LOG_WARNING(
                "[LlamaBatch][EAGLE] step=%d, seq=%d, accepted_token=%d mismatches "
                "dynamicDecode token=%d; keeping raw acceptance for metrics",
                g.step,
                i,
                accepted_token,
                h_token_ids[i]);
        }

        int planned_tokens_per_seq = eagle_tokens_per_seq;
        if (!eagle_planned_tokens_per_seq_.empty()
            && i < static_cast<int>(eagle_planned_tokens_per_seq_.size())) {
            planned_tokens_per_seq = eagle_planned_tokens_per_seq_[i];
        }

        if (!multi_token_slot_enabled) {
            planned_tokens_per_seq = 1;
            if (accepted_len > 1) {
                accepted_len = 1;
            }
        }

        const int kv_draft_len    = planned_tokens_per_seq;
        const int kv_accepted_len = std::max(1, std::min(accepted_len, kv_draft_len));
        const int rewind_len      = std::max(0, kv_draft_len - kv_accepted_len);

        kv_draft_lengths[i]    = kv_draft_len;
        kv_accepted_lengths[i] = kv_accepted_len;

        step_draft_total += planned_tokens_per_seq;
        step_accepted_total += raw_accepted_len;

        req->metrics->eagle_total_draft_tokens += planned_tokens_per_seq;
        req->metrics->eagle_total_accepted_tokens += raw_accepted_len;
        req->metrics->eagle_steps += 1;

        // When target-tree decode is enabled for this engine, also track
        // tree-aware metrics so benchmarks can distinguish how many tokens
        // flowed through the tree path versus the regular speculative path.
        if (target_tree_active) {
            req->metrics->eagle_tree_draft_tokens += planned_tokens_per_seq;
            req->metrics->eagle_tree_target_tokens += planned_tokens_per_seq;
            req->metrics->eagle_tree_accepted_tokens += raw_accepted_len;
        }

        if (rewind_len > 0) {
            req->metrics->eagle_total_rewound_tokens += rewind_len;
            req->metrics->eagle_rewind_steps += 1;
        }

        TM_LOG_INFO(
            "[LlamaBatch][EAGLE] step=%d, seq=%d, dynamic_token=%d, raw_accepted_len=%d, "
            "effective_accepted_len=%d, accepted_token=%d, tokens_per_seq=%d, rewind_len=%d",
            g.step,
            i,
            h_token_ids[i],
            raw_accepted_len,
            accepted_len,
            accepted_token,
            planned_tokens_per_seq,
            rewind_len);
    }

    if (param_.eagle_metrics_debug && tp_rank_ == 0) {
        TM_LOG_INFO(
            "[LlamaBatch][EAGLE_METRICS] step=%d, batch=%d, draft_tokens=%d, accepted_tokens=%d",
            g.step,
            batch_size,
            step_draft_total,
            step_accepted_total);
    }
}

void LlamaBatch::runEagleMultiTokenAdvance(const std::vector<int>&  h_token_ids,
                                           const std::vector<bool>& h_finished_slots,
                                           int                      batch_size,
                                           int                      eagle_tokens_per_seq,
                                           const std::vector<int>&  eagle_accepted_lens,
                                           const std::vector<int>&  eagle_accepted_tokens,
                                           GenerationState&         g)
{
    TM_LOG_INFO("[LlamaBatch][EAGLE] step=%d, verified target tokens after DynamicDecode", g.step);

    advanceSequencesByEagleAcceptance(h_token_ids,
                                      h_finished_slots,
                                      batch_size,
                                      eagle_tokens_per_seq,
                                      eagle_accepted_lens,
                                      eagle_accepted_tokens,
                                      g);

    runEagleKVCacheCompaction(eagle_accepted_lens, batch_size, g);
}

void LlamaBatch::runEagleKVCacheCompaction(const std::vector<int>& kv_accepted_lengths,
                                           int                     batch_size,
                                           const GenerationState&  g)
{
    if (!isEagleMultiTokenStepEnabled(g)) {
        return;
    }

    bool any_compaction = false;
    for (int i = 0; i < batch_size && i < static_cast<int>(kv_accepted_lengths.size()); ++i) {
        if (kv_accepted_lengths[i] > 0) {
            any_compaction = true;
            break;
        }
    }
    if (!any_compaction) {
        return;
    }

    // Prepare Block Tables (Physical IDs from SequenceManager)
    const int max_batch_size = max_batch_size_;
    static std::vector<int> h_block_tables_storage;
    const int total_block_entries = max_batch_size * kv_max_blocks_per_seq_;
    if (h_block_tables_storage.size() < static_cast<size_t>(total_block_entries)) {
        h_block_tables_storage.resize(total_block_entries);
    }
    std::fill_n(h_block_tables_storage.data(), total_block_entries, -1);
    
    for (int slot = 0; slot < state_->active_size; ++slot) {
        const auto* seq = state_->sequences[slot];
        if (!seq) continue;
        const int count = std::min<int>(seq->blocks.size(), kv_max_blocks_per_seq_);
        for (int j = 0; j < count; ++j) {
            h_block_tables_storage[slot * kv_max_blocks_per_seq_ + j] = seq->blocks[j];
        }
    }

    if (!eagle_kv_block_tables_) {
        return;
    }

    // Copy block tables to device
    core::Copy(h_block_tables_storage.data(), h_block_tables_storage.size(), eagle_kv_block_tables_.data());

    // Get destination block pointers from model's buffer
    void** dst_block_base_ptrs = (void**)model_->getKvBlockPtrsBuffer().raw_data();
    
    // Call Model Compaction (currently stubbed)
    model_->compactKVCache(batch_size,
                           eagle_kv_block_tables_.data(),
                           dst_block_base_ptrs,
                           kv_max_blocks_per_seq_,
                           kv_block_size_,
                           stream_);
}

void LlamaBatch::runEagleKVRewind(const std::vector<int>& kv_draft_lengths,
                                  const std::vector<int>& kv_accepted_lengths,
                                  int                     batch_size,
                                  const GenerationState&  g)
{
    if (!isEagleMultiTokenStepEnabled(g) || kv_block_size_ <= 0 || kv_max_blocks_per_seq_ <= 0
        || kv_draft_lengths.empty()) {
        return;
    }

    // Fast path: if no sequence has any rejected tokens (i.e. draft_len ==
    // accepted_len for all slots), there is nothing to rewind and we can
    // skip all KV-related work for this step.
    bool any_rewind = false;
    for (int i = 0; i < batch_size && i < static_cast<int>(kv_draft_lengths.size())
                    && i < static_cast<int>(kv_accepted_lengths.size());
         ++i) {
        if (kv_draft_lengths[i] > kv_accepted_lengths[i]) {
            any_rewind = true;
            break;
        }
    }
    if (!any_rewind) {
        return;
    }

    using namespace turbomind::kernels::speculative_decoding;

    const int max_batch_size = max_batch_size_;
    const int num_layers     = model_->layer_num_;

    EagleKVRewindConfig cfg{};
    cfg.block_size         = static_cast<SizeType>(kv_block_size_);
    cfg.max_batch_size     = static_cast<SizeType>(max_batch_size);
    cfg.max_blocks_per_seq = static_cast<SizeType>(kv_max_blocks_per_seq_);
    cfg.num_layers         = static_cast<SizeType>(num_layers);

    static std::vector<SizeType> draft_lengths_storage;
    static std::vector<SizeType> accepted_lengths_storage;
    static std::vector<SizeType> batch_slots_storage;

    const int bsz = batch_size;

    if (draft_lengths_storage.size() < static_cast<size_t>(max_batch_size)) {
        draft_lengths_storage.resize(max_batch_size);
        accepted_lengths_storage.resize(max_batch_size);
    }
    if (batch_slots_storage.size() < static_cast<size_t>(bsz)) {
        batch_slots_storage.resize(bsz);
    }

    std::fill_n(draft_lengths_storage.data(), max_batch_size, SizeType{0});
    std::fill_n(accepted_lengths_storage.data(), max_batch_size, SizeType{0});

    for (int i = 0; i < bsz; ++i) {
        draft_lengths_storage[i]    = static_cast<SizeType>(kv_draft_lengths[i]);
        accepted_lengths_storage[i] = static_cast<SizeType>(kv_accepted_lengths[i]);
        batch_slots_storage[i]      = static_cast<SizeType>(i);  // local index == slot
    }

    // Build a logical block table from SequenceManager's host-side
    // block IDs for each active slot. Unused entries are filled
    // with -1 and may be cleared further by the rewind kernel.
    static std::vector<int> h_block_tables_storage;
    const int               total_block_entries = max_batch_size * kv_max_blocks_per_seq_;
    if (h_block_tables_storage.size() < static_cast<size_t>(total_block_entries)) {
        h_block_tables_storage.resize(total_block_entries);
    }
    std::fill_n(h_block_tables_storage.data(), total_block_entries, -1);
    for (int slot = 0; slot < state_->active_size; ++slot) {
        const auto* seq = state_->sequences[slot];
        if (!seq) {
            continue;
        }
        const int count = std::min<int>(seq->blocks.size(), kv_max_blocks_per_seq_);
        for (int j = 0; j < count; ++j) {
            h_block_tables_storage[slot * kv_max_blocks_per_seq_ + j] = seq->blocks[j];
        }
    }

    if (!eagle_kv_block_tables_ || !eagle_kv_batch_slots_ || !eagle_kv_rewind_lengths_) {
        return;
    }

    core::Copy(h_block_tables_storage.data(), h_block_tables_storage.size(), eagle_kv_block_tables_.data());
    core::Copy(batch_slots_storage.data(), batch_slots_storage.size(), eagle_kv_batch_slots_.data());

    // Optional: build a flat view over BlockManager's KV blocks so the
    // rewind kernel can null-out pointers for rewound blocks.
    if (!eagle_kv_cache_blocks_) {
        const size_t total_entries =
            static_cast<size_t>(num_layers) * static_cast<size_t>(kv_max_blocks_per_seq_);
        check_cuda_error(cudaMalloc(&eagle_kv_cache_blocks_, total_entries * sizeof(void*)));
    }
    static std::vector<void*> h_kv_cache_blocks;
    const int total_blocks = std::min(kv_max_blocks_per_seq_, sequence_manager_->total_count());
    const size_t total_entries =
        static_cast<size_t>(num_layers) * static_cast<size_t>(kv_max_blocks_per_seq_);
    if (h_kv_cache_blocks.size() < total_entries) {
        h_kv_cache_blocks.resize(total_entries);
    }
    std::fill_n(h_kv_cache_blocks.data(), total_entries, nullptr);
    for (int block_id = 0; block_id < total_blocks; ++block_id) {
        void* ptr = sequence_manager_->GetBlockPtr(block_id);
        for (int layer = 0; layer < num_layers; ++layer) {
            const size_t idx =
                static_cast<size_t>(layer) * static_cast<size_t>(kv_max_blocks_per_seq_)
                + static_cast<size_t>(block_id);
            h_kv_cache_blocks[idx] = ptr;
        }
    }
    check_cuda_error(cudaMemcpyAsync(eagle_kv_cache_blocks_,
                                     h_kv_cache_blocks.data(),
                                     total_entries * sizeof(void*),
                                     cudaMemcpyHostToDevice,
                                     stream_));

    computeAndInvokeKVCacheRewind(cfg,
                                  draft_lengths_storage.data(),
                                  accepted_lengths_storage.data(),
                                  batch_slots_storage.data(),
                                  static_cast<SizeType>(bsz),
                                  reinterpret_cast<SizeType*>(eagle_kv_rewind_lengths_.data()),
                                  reinterpret_cast<SizeType const*>(eagle_kv_batch_slots_.data()),
                                  reinterpret_cast<SizeType*>(eagle_kv_block_tables_.data()),
                                  eagle_kv_cache_blocks_,
                                  stream_);

    // Mirror rewind lengths into SequenceManager's host-side
    // sequences: trim tail blocks, unlock KV blocks in the
    // BlockManager, and reduce cache_len.
    static std::vector<SizeType> h_rewind_lengths_storage;
    if (h_rewind_lengths_storage.size() < static_cast<size_t>(max_batch_size)) {
        h_rewind_lengths_storage.resize(max_batch_size);
    }
    if (isEagleDebugEnabled() && tp_rank_ == 0
        && turbomind::isEnvVarEnabled("LMDEPLOY_EAGLE_COPY_DEBUG")) {
        TM_LOG_WARNING(
            "[LlamaBatch][EAGLE][CopySITE:kv_rewind_lengths] step=%d max_batch=%d a=%p b=%p n=%d",
            g.step,
            max_batch_size,
            eagle_kv_rewind_lengths_.data(),
            h_rewind_lengths_storage.data(),
            max_batch_size);
    }
    core::Copy(eagle_kv_rewind_lengths_.data(), max_batch_size, h_rewind_lengths_storage.data());
    check_cuda_error(cudaStreamSynchronize(stream_));

    const int active_size = state_->active_size;
    for (int slot = 0; slot < active_size; ++slot) {
        const SizeType rewind_tokens = h_rewind_lengths_storage[slot];
        if (rewind_tokens <= 0) {
            continue;
        }
        auto* seq = const_cast<Sequence*>(state_->sequences[slot]);
        if (!seq) {
            continue;
        }

        const int blocks_to_free =
            (rewind_tokens + kv_block_size_ - 1) / kv_block_size_;

        if (blocks_to_free > 0 && !seq->blocks.empty()) {
            const int old_size = static_cast<int>(seq->blocks.size());
            const int new_size =
                std::max(0, old_size - blocks_to_free);
            if (new_size < old_size) {
                BlockIds tail_blocks(seq->blocks.begin() + new_size, seq->blocks.end());
                seq->blocks.resize(new_size);
                // Unlock KV blocks for the rewound tail so they
                // can be reused by the BlockManager.
                sequence_manager_->UnlockBlocks(tail_blocks);

                if (seq->cache_len > 0) {
                    const int tokens_to_drop =
                        std::min<int>(rewind_tokens, seq->cache_len);
                    seq->cache_len = std::max(0, seq->cache_len - tokens_to_drop);
                }
            }
        }
    }
}

void LlamaBatch::FreeBuffer()
{
    input_ids_buf_       = {};
    decoder_output_buf_  = {};
    input_length_buf_    = {};
    context_length_buf_  = {};
    init_context_length_ = {};
    sequence_lengths_    = {};
    cu_block_counts_     = {};
    block_ptrs_          = {};
    sampled_logprobs_    = {};
    sampled_indexes_     = {};
    sampled_nums_        = {};
    token_ids_buf_       = {};
    sampling_logits_     = {};
    finished_buf_        = {};
    seq_limit_len_       = {};
    rope_theta_          = {};
    d_random_seed_       = {};
    d_curand_state_      = {};
}

void LlamaBatch::InitializeSampling(const GenerationState& g)
{
    NvtxScope _("InitSampling");

    const int batch_size = state_->active_size - g.partial;

    if (batch_size == 0) {
        return;
    }

    // Context length at initialization, will stay constant until re-initialziation
    Copy(context_length_buf_, batch_size, init_context_length_);

    Copy(context_length_buf_, batch_size, sequence_lengths_);
    // `sequence_lengths_` will be increased by dynamic decode
    // note that in decoder and in output "sequence length" has different semantic
    // - in decoder it means length of sequence that has kv cache already computed
    // - in output it means length of all tokens (the last generated token does not have k/v cache computed yet)
    invokePlusScalar(sequence_lengths_.data(), -1, batch_size, stream_);
    sync_check_cuda_error();

    Clear(token_ids_buf_.slice(0, batch_size * session_len_));
    invokeTranspose2D(token_ids_buf_.data(), state_->output_ids.data(), batch_size, session_len_, stream_);
    sync_check_cuda_error();

    // token_ids_buf_[s, b]
    // ABCDe            ABCDe     e
    // ABCDEFGHIJk      ABCDEFGHIJk
    // ABCDEFGHi    ->  ABCDEFGHi i
    // ABCDEFGh         ABCDEFGh  h
    // ABCd             ABCd      d
    invokePadLastTokenIds(token_ids_buf_.data(), init_context_length_.data(), g.max_init_ctx_len, batch_size, stream_);
    sync_check_cuda_error();

    // seq_limit_len_, will be compared to `step` instead of `sequence_length`, so padding len should be accounted for
    for (int i = 0; i < batch_size; ++i) {
        h_seq_limit_len_[i] = state_->seq_len_limit[i] + (g.max_init_ctx_len - state_->h_context_length[i]);
    }
    Copy(h_seq_limit_len_, batch_size, seq_limit_len_);

    std::vector<const Request*> rs;
    rs.reserve(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        rs.push_back(state_->requests[i].get());
    }

    model_->dynamic_decode_->Setup(rs, {{"prompt_length", {state_->h_prompt_length, {batch_size}}}});

    sync_check_cuda_error();
}

void LlamaBatch::ComputeAndOutputLogits(const Tensor& hidden_states, int first, int last)
{
    auto enable = [&] {
        for (int i = first; i < last; ++i) {
            if (state_->requests[i]->gen_cfg.output_logits == GenerationConfig::kAll) {
                const auto& s = *state_->sequences[i];
                // Skip when the seq is filling missed cache only
                if (s.cache_len + h_input_length_buf_[i] > s.tokens.size()) {
                    return true;
                }
            }
        }
        return false;
    }();

    if (!enable) {
        return;
    }

    const int vocab_size_padded = model_->vocab_size_padded_;
    const int token_num         = hidden_states.shape(0);

    if (symm_logits_buf_.shape(0) < token_num) {
        if (tp_size_ > 1) {
            check_cuda_error(cudaStreamSynchronize(stream_));
            comm_.h_tp_group->Sync();
        }
        symm_logits_buf_ = {{token_num, vocab_size_padded}, data_type_, symm_alloc_};
        if (tp_size_ > 1) {
            check_cuda_error(cudaStreamSynchronize(stream_));
            comm_.h_tp_group->Sync();
        }
    }

    auto logits = model_->postDecodeEmbedding(hidden_states, symm_logits_buf_.buffer());

    if (tp_rank_ == 0) {
        OutputLogits(logits, first, last, GenerationConfig::kAll);
    }
}

void LlamaBatch::OutputLogits(const Tensor& logits, int first, int last, GenerationConfig::OutType out_type)
{
    const auto& src_buf   = logits.buffer();
    const auto  elem_size = byte_size(logits.dtype(), 1);
    // when `is_all` is false, logits only contains last token of the sequences
    const bool is_all = out_type == GenerationConfig::kAll;

    int base = 0;

    for (int i = first; i < last; ++i) {

        const int input_len = h_input_length_buf_[i];  // input length for this iter

        if (state_->requests[i]->gen_cfg.output_logits == out_type) {

            auto& dst_buf = state_->requests[i]->outputs.at("logits").buffer();

            const int cache_len   = state_->sequences[i]->cache_len;
            const int history_len = state_->sequences[i]->tokens.size();

            // ----------H------I-------P-----------
            //      C        C      C         C

            // offset to the last token prompt
            const int offset = is_all ? 0 : state_->requests[i]->inputs.at("input_ids").shape(0) - 1;

            int diff = (history_len + offset) - cache_len;

            const int valid_len = input_len - std::max(0, (history_len + offset) - cache_len);

            // TM_LOG_ERROR("%d %d   %d %d  %d  %d %d",
            //              history_len,
            //              offset,
            //              cache_len,
            //              input_len,
            //              valid_len,
            //              std::max(0, diff),
            //              std::max(0, -diff));

            if (valid_len <= 0) {
                continue;
            }

            int src_base = base;

            if (is_all) {
                // Skip invalid tokens caused by cache miss
                src_base += std::max(0, (history_len + offset) - cache_len);
            }
            // Skip previous chunks
            int dst_base = std::max(0, cache_len - (history_len + offset));

            check_cuda_error(cudaMemcpy2DAsync(dst_buf.raw_data(dst_base * model_->vocab_size_),
                                               elem_size * model_->vocab_size_,
                                               src_buf.raw_data(src_base * model_->vocab_size_padded_),
                                               elem_size * model_->vocab_size_padded_,
                                               elem_size * model_->vocab_size_,
                                               valid_len,
                                               cudaMemcpyDefault,
                                               stream_));
        }

        base += is_all ? input_len : 1;
    }
}

void LlamaBatch::OutputLastHiddenState(const Tensor& hidden_states, int first, int last)
{
    const auto& src_buf   = hidden_states.buffer();
    const auto  data_type = src_buf.dtype();
    int         base      = 0;

    for (int i = first; i < last; ++i) {
        const int input_len = h_input_length_buf_[i];  // input lenght for this iter

        if (auto out_type = state_->requests[i]->gen_cfg.output_last_hidden_state) {

            const bool is_all = out_type == GenerationConfig::kAll;

            auto& dst_buf = state_->requests[i]->outputs.at("last_hidden_state").buffer();

            const int cache_len   = state_->sequences[i]->cache_len;
            const int history_len = state_->sequences[i]->tokens.size();

            // offset to the last prompt token
            const int offset = is_all ? 0 : state_->requests[i]->inputs.at("input_ids").shape(0) - 1;

            const int valid_len = input_len - std::max(0, (history_len + offset) - cache_len);

            // TM_LOG_ERROR("%d %d %d %d %d", history_len, offset, cache_len, input_len, valid_len);

            if (valid_len > 0) {
                // Skip invalid tokens caused by cache miss
                int src_base = std::max(0, (history_len + offset) - cache_len) + base;
                // Skip previous chunks
                int dst_base = std::max(0, cache_len - (history_len + offset));

                if (turbomind::isEagleDebugEnabled()) {
                    TM_LOG_WARNING(
                        "[LlamaBatch][EAGLE][CopySITE:last_hidden_state] src_base=%d dst_base=%d valid_len=%d a=%p "
                        "b=%p bytes=%zu",
                        src_base,
                        dst_base,
                        valid_len,
                        src_buf.raw_data(src_base * model_->hidden_units_),
                        dst_buf.raw_data(dst_base * model_->hidden_units_),
                        byte_size(data_type, valid_len * model_->hidden_units_));
                }
                core::Copy(src_buf.raw_data(src_base * model_->hidden_units_),
                           byte_size(data_type, valid_len * model_->hidden_units_),
                           dst_buf.raw_data(dst_base * model_->hidden_units_));
            }
        }

        // hidden_states += input_len * model_->hidden_units_;
        base += input_len;
    }
}

void LlamaBatch::Finish(GenerationState& g, std::vector<Signal>& signals)
{
    NvtxScope scope("Finish");
    const int batch_size = state_->active_size;

    signals.reserve(batch_size);

    if (batch_size - g.partial) {
        FT_CHECK(g.step >= 0);

        // [s,b] -> [b,s] and skip padding in [context_len, max_context_len)
        invokeGatherOutput(state_->output_ids.data(),
                           token_ids_buf_.data(),
                           init_context_length_.data(),
                           g.max_init_ctx_len,
                           g.step,
                           session_len_,
                           batch_size - g.partial,
                           stream_);
        sync_check_cuda_error();
    }

    Copy(token_ids_buf_.slice((g.step - 1) * (batch_size - g.partial), -1), batch_size - g.partial, h_output_ids_);
    Copy(finished_buf_, batch_size, state_->h_finished);
    Copy(sequence_lengths_, batch_size, state_->h_context_length);

    bool output_logprobs = false;
    for (int i = 0; i < batch_size - g.partial; ++i) {
        if (state_->requests[i]->gen_cfg.output_logprobs) {
            output_logprobs = true;
            break;
        }
    }
    if (output_logprobs) {
        Copy(sampled_logprobs_, batch_size * kMaxLogProb, h_sampled_logprobs_);
        Copy(sampled_indexes_, batch_size * kMaxLogProb, h_sampled_indexes_);
        Copy(sampled_nums_, batch_size, h_sampled_nums_);
    }

    check_cuda_error(cudaStreamSynchronize(stream_));

    // invariant: context_length = sequence_length + 1, so that h_context_length include all (including the one just
    // generated) tokens
    for (int i = 0; i < batch_size; ++i) {
        ++state_->h_context_length[i];
    }

    // ! Only rank-0 writes to output
    if (tp_rank_ == 0 && output_logprobs) {
        NvtxScope scope("logprobs");
        float*    sampled_logprobs_ptr = h_sampled_logprobs_.data();
        uint32_t* sampled_indexes_ptr  = h_sampled_indexes_.data();
        uint32_t* sampled_nums_ptr     = h_sampled_nums_.data();
        for (int i = 0; i < batch_size - g.partial; ++i) {
            if (state_->requests[i] && state_->requests[i]->gen_cfg.output_logprobs) {
                auto logprob_vals    = state_->requests[i]->outputs.at("logprob_vals").data<float>();
                auto logprob_indexes = state_->requests[i]->outputs.at("logprob_indexes").data<int32_t>();
                auto logprob_nums    = state_->requests[i]->outputs.at("logprob_nums").data<int32_t>();

                int offset = state_->h_context_length[i] - state_->h_prompt_length[i] - 1;
                std::copy(sampled_logprobs_ptr,
                          sampled_logprobs_ptr + *sampled_nums_ptr,
                          logprob_vals + offset * kMaxLogProb);
                std::copy(sampled_indexes_ptr,
                          sampled_indexes_ptr + *sampled_nums_ptr,
                          logprob_indexes + offset * kMaxLogProb);
                *(logprob_nums + offset) = *sampled_nums_ptr;
            }
            sampled_logprobs_ptr += kMaxLogProb;
            sampled_indexes_ptr += kMaxLogProb;
            sampled_nums_ptr++;
        }
    }

    // ! Only rank-0 writes to output
    if (tp_rank_ == 0) {
        NvtxScope scope("output_ids");
        for (int i = 0; i < batch_size - g.partial; ++i) {
            if (auto& r = state_->requests[i]) {
                auto      output_ids  = r->output_ids.data();
                auto      output_len  = r->sequence_length.data();
                const int count       = state_->h_context_length[i];
                output_ids[count - 1] = h_output_ids_[i];
                *output_len           = count;
            }
        }
    }

    // Cache computed blocks to block trie
    sequence_manager_->CachePrompt(state_->sequences, batch_size);

    if (debug_ && tp_rank_ == 0) {
        for (int i = 0; i < batch_size; ++i) {
            // ss << (i ? ", " : "") << "(" << state_->h_context_length[i] << "," << state_->h_finished[i] << ")";
            std::vector<int> tokens(state_->h_context_length[i]);
            if (turbomind::isEagleDebugEnabled()) {
                TM_LOG_WARNING(
                    "[LlamaBatch][EAGLE][CopySITE:dump_tokens_i] i=%d session_len=%d len=%zu a=%p b=%p",
                    i,
                    session_len_,
                    tokens.size(),
                    state_->output_ids.data() + i * session_len_,
                    tokens.data());
            }
            core::Copy(state_->output_ids.data() + i * session_len_, tokens.size(), tokens.data());
            cudaStreamSynchronize(stream_);
            std::stringstream ss;
            for (const auto& t : tokens) {
                ss << " " << t;
            }
            TM_LOG_INFO("[Finish] slot %d, tokens [%s]", i, ss.str().c_str());
        }
    }

    {
        NvtxScope _("count and sync");
        bool      need_sync = false;
        for (int i = 0; i < batch_size - g.partial; ++i) {
            if (state_->h_finished[i]) {
                ++g.finished_count;
                if (!state_->requests[i]->session.end_flag) {
                    need_sync = true;
                }
            }
        }
        if (need_sync) {
            // Release updates on request output buffers to all ranks (`Interrupt` will use it)
            comm_.h_tp_group->Sync();
        }
    }

    {
        NvtxScope _("stream_and_completion_signal");
        for (int i = 0; i < batch_size - g.partial; ++i) {
            auto& r = state_->requests[i];
            if (state_->h_finished[i]) {
                // Interrupt finished sequences and move the request handle into the signal closure
                signals.push_back(Interrupt(i));
                // Interrupt should reset r
                FT_CHECK(!r);
            }
            else if (r->stream_output && tp_rank_ == 0) {
                const auto seq_len = *r->sequence_length.data();
                // Create signals by copying the request handles for non-finished streaming requests
                signals.push_back([this, r, seq_len] {  //
                    UpdateState(*r, Request::kOk, seq_len);
                });
            }
        }
    }

    if (g.finished_count) {
        // synchronize for interrupted sequences
        check_cuda_error(cudaStreamSynchronize(stream_));
    }

    if (g.partial) {
        const int i = batch_size - 1;
        // recover full context length of partial
        state_->h_context_length[i] = g.partial_context_legnth;
    }

    // Update the schedule metrics since some requests might finish the inference
    UpdateMetrics();
}

auto LlamaBatch::Interrupt(int index, bool force_stop) -> Signal
{
    if (tp_rank_ == 0) {
        TM_LOG_INFO("[Interrupt] slot %d, request %llu, stop %d", index, state_->requests[index]->id, force_stop);
    }

    if (debug_ && tp_rank_ == 0) {
        std::vector<int> tokens(state_->h_context_length[index]);
        if (turbomind::isEagleDebugEnabled()) {
            TM_LOG_WARNING(
                "[LlamaBatch][EAGLE][CopySITE:dump_tokens_index] index=%d session_len=%d len=%zu a=%p b=%p",
                index,
                session_len_,
                tokens.size(),
                state_->output_ids.data() + index * session_len_,
                tokens.data());
        }
        core::Copy(state_->output_ids.data() + index * session_len_, tokens.size(), tokens.data());
        cudaStreamSynchronize(stream_);
        std::stringstream ss;
        for (const auto& t : tokens) {
            ss << " " << t;
        }
        TM_LOG_INFO("[Interrupt] slot %d, tokens [%s]", index, ss.str().c_str());
    }

    auto& seq = *state_->sequences[index];
    {
        // output_ids is updated & synced in `Finish`
        const auto output_ids = state_->requests[index]->output_ids.data();
        const auto output_len = state_->h_context_length[index];
        // Update token IDs to perform `CacheGeneration` later
        seq.tokens.resize(output_len);
        std::copy_n(output_ids, output_len, seq.tokens.data());
    }

    if (state_->requests[index]->session.end_flag) {
        // Cache the generated tokens of the sequence
        sequence_manager_->CacheGeneration(seq);
        FT_CHECK(sequence_manager_->Erase(state_->requests[index]->id));
    }
    else {
        // Prefix caching is incompatible with interactive mode, so we don't perform `CacheGeneration` here
        // Save random state in host memory
        seq.random_state.resize(sizeof(curandState_t));
        // This async copy must be synchronized by the caller
        if (turbomind::isEagleDebugEnabled()) {
            TM_LOG_WARNING(
                "[LlamaBatch][EAGLE][CopySITE:curand_state_seq] index=%d a=%p b=%p n=%d",
                index,
                (curandState_t*)state_->curand_state.data() + index,
                (curandState_t*)seq.random_state.data(),
                1);
        }
        core::Copy((curandState_t*)state_->curand_state.data() + index, 1, (curandState_t*)seq.random_state.data());

        // Set unlock flag for corresponding blocks, will be unlocked in the next `Materialize()`
        sequence_manager_->UpdateAndSetUnlock(seq);
    }

    state_->sequences[index] = nullptr;

    auto ec = std::exchange(state_->errors[index], Request::kOk);

    const auto len = *state_->requests[index]->sequence_length.data();
    // move the request handle into the signal
    return [this, len, force_stop, r = std::move(state_->requests[index])] {  //
        UpdateState(*r, force_stop ? Request::kCancel : Request::kFinish, len);
    };
}

namespace {

struct RequestData {
    std::vector<std::shared_ptr<Request>> infer;  // incoming inference request
    std::vector<std::shared_ptr<Request>> kill;   // incoming kill request

    std::vector<int> cancel;  // canceled indices in current batch
    bool             abort;
};

}  // namespace

void LlamaBatch::InternalThreadEntry()
{
    // TM_LOG_INFO("[InternalThreadEntry] %d", (int)rank_);
    check_cuda_error(cudaSetDevice(device_id_));

    core::ContextGuard guard{context_->core_stream, context_->allocator};

    // Initialize `AnomalyHandler`
    AnomalyHandler::instance().Init(tp_rank_, model_->vocab_size_padded_, 0, max_batch_size_, stream_);

    GenerationState g{};

    while (1) {

        std::shared_ptr<RequestData> req;

        if (tp_rank_ == 0) {
            req = std::make_shared<RequestData>();
            {
                NvtxScope  _("pop");
                const int  free_slot_count = max_batch_size_ - state_->size + g.finished_count;
                const bool is_empty        = (free_slot_count == max_batch_size_);
                // Block if batch is empty AND no silbings are ready
                gateway_->pop(req->infer, req->kill, free_slot_count, is_empty, req->abort, dp_rank_);
            }
            // Mark reqs to the same session_id as invalid and also interactive-mode reqs when
            // prefix caching is enabled(which are dangerous to the engine)
            DisableInvalidRequests(req->infer, req->kill);
            FindCanceledIndices(req->cancel);
        }

        if (state_->size == g.finished_count) {
            // Batch is empty, use blocking sync to avoid spinning
            comm_.h_tp_group->Sync(true);
        }

        NvtxScope scope("mainloop");

        // 1. Wait while rank-0 is dequeueing
        // 2. Broadcast `ec` from rank-0
        Broadcast(comm_.h_tp_group, req, 0);

        if (req->abort) {
            TM_LOG_INFO("[InternalThreadEntry] stop requested.");
            break;
        }

        std::vector<Signal> signals;

        ProcessKillRequests(req->kill, signals);

        // Shared `priority` field will be assigned by rank-0
        ProcessInferRequests(req->infer, signals);

        // 1. Wait while shared `requests` is being used
        // 2. Broadcast modifcations from rank-0
        // comm_.h_comm->Sync(comm_.h_comm_tp_group);

        ProcessCancelRequests(req->cancel, signals);

        if (tp_rank_ == 0) {
            gateway_->notify(std::move(signals));
        }

        Initialize(g);

        // update the schedule metrics and request metrics in every forward iter
        UpdateMetrics();

        const int n_active = AllReduce(comm_.h_dp_group, state_->active_size, comm::RedOp::kSum);

        if (n_active) {
            //
            Forward(g);

            Finish(g, signals);

            if (g.finished_count) {
                // Finished requests and corresponding output tensors will be released when notified
                // wait for all ranks to ensure no rank (except for output thread) will access related
                // resources
                comm_.h_tp_group->Sync();
            }

            if (tp_rank_ == 0) {
                gateway_->notify(std::move(signals));
            }
        }
    }

    // barrier synchronization inside
    DestroyCommunicators();
}

void LlamaBatch::Start()
{
    TM_LOG_INFO("LlamaBatch<T>::Start()");
    internal_thread_ = std::thread([this] {
        try {
            InternalThreadEntry();
        }
        catch (const std::exception& e) {
            TM_LOG_ERROR("[Engine] %s", e.what());
            std::abort();
        }
    });
}

bool LlamaBatch::Forward(GenerationState& g)
{
    NvtxScope _("Forward");

    FT_CHECK(max_context_token_num_ >= max_batch_size_);

    const int batch_size = state_->active_size;

    const bool eagle_enabled =
        param_.enable_speculative_decoding
        && (param_.spec_method == "eagle" || param_.spec_method == "eagle3");

    // Per-step EAGLE acceptance bookkeeping. We keep these on the host so
    // that, after DynamicDecode runs, we can reason about how many tokens
    // were accepted per sequence and how that compares to the tokens that
    // actually advanced the decode state. Clear per-step state up front so
    // no stale entries from previous steps influence this iteration.
    std::vector<int> eagle_accepted_lens;
    std::vector<int> eagle_accepted_tokens;
    int              eagle_tokens_per_seq = 0;

    eagle_planned_tokens_per_seq_.clear();

    // Helper for multi-token EAGLE: given the active decode batch size,
    // compute how many draft tokens per sequence we would like to generate
    // in this engine step, respecting both the global engine token budget
    // and the per-sequence speculative token cap.
    auto compute_eagle_draft_tokens_per_seq = [&](int decode_batch_size) -> int {
        if (!eagle_enabled || decode_batch_size <= 0) {
            return 0;
        }
        const int max_engine_tokens = model_->eagleMaxEngineTokensPerStep();
        if (max_engine_tokens <= 0) {
            return 0;
        }

        // When the engine-side budget resolves to 1, treat this as
        // single-token EAGLE regardless of batch size.
        if (max_engine_tokens <= 1) {
            return 1;
        }

        const int per_seq_cap = param_.spec_max_decoding_draft_tokens;
        if (per_seq_cap <= 0) {
            return 1;
        }

        // Hard upper bound from engine budget: do not exceed the per-step
        // engine token allowance when spreading draft tokens across the
        // active decode batch.
        const int max_by_engine = max_engine_tokens / decode_batch_size;
        if (max_by_engine <= 0) {
            return 1;
        }

        const int tokens_per_seq = std::max(1, std::min(per_seq_cap, max_by_engine));
        return tokens_per_seq;
    };

    if (tp_rank_ == 0 && eagle_enabled) {
        const int max_engine_tokens = model_->eagleMaxEngineTokensPerStep();
        TM_LOG_INFO(
            "[LlamaBatch][EAGLE] step=%d, method=%s, "
            "config_num_speculative_tokens=%d, "
            "spec_max_decoding_tokens=%d, "
            "engine_max_tokens_per_step=%d, "
            "max_non_leaf_nodes=%d",
            g.step,
            param_.spec_method.c_str(),
            param_.spec_max_decoding_draft_tokens,
            param_.spec_max_decoding_tokens,
            max_engine_tokens,
            param_.spec_max_non_leaf_nodes);
    }

    const int active_size = state_->active_size;

    constexpr int kLogInterval = 10;
    if (tp_rank_ == 0 && (g.step - 1) % kLogInterval == 0) {
        TM_LOG_INFO("------------------------- step = %d -------------------------", g.step - 1);
    }

    int               pf_offset = -1;
    std::vector<int*> input_d_ptrs(active_size);

    for (int i = 0; i < active_size; ++i) {
        const auto& seq = *state_->sequences[i];
        // const int   missing = state_->h_context_length[i] - seq.cache_len;
        FT_CHECK(seq.input_length >= 1);
        h_input_length_buf_[i] = seq.input_length;
        input_d_ptrs[i]        = state_->output_ids.data() + i * session_len_ + seq.cache_len;
        if (seq.input_length > 1 && pf_offset < 0) {
            pf_offset = i;
        }
    }
    if (pf_offset < 0) {
        pf_offset = active_size;
    }

    // These buffers are only accessed when there are prefill workloads
    if (pf_offset != active_size) {
        Copy(state_->h_context_length, active_size, context_length_buf_);
        Copy(h_input_length_buf_, active_size, input_length_buf_);
    }

    // Find mini-batch offsets: input length > 1 ? prefill() : decode()
    // Constraints on mini-batches
    //   sum(Q) <= `max_forward_token_num` && sum(K) <= `max_context_token_num`
    std::vector<int> offsets{0};
    // initialize first mini-batch with decode tokens
    int sum_q = pf_offset;
    int sum_k = 0;  // only for prefill
    for (int i = pf_offset; i < active_size; ++i) {
        FT_CHECK(h_input_length_buf_[i] <= max_forward_token_num_);
        const int q = sum_q + h_input_length_buf_[i];
        const int k = sum_k + state_->h_context_length[i];
        if (q <= max_forward_token_num_ && k <= max_context_token_num_) {
            sum_q = q;
            sum_k = k;
        }
        else {
            offsets.push_back(i);
            sum_q = h_input_length_buf_[i];
            sum_k = state_->h_context_length[i];
        }
    }
    offsets.push_back(active_size);

    // Synchronize mini batch count with sync DP ranks
    int n_batches = AllReduce(comm_.h_dp_group, (int)offsets.size(), comm::RedOp::kMax);

    // Populate empty batches
    while (offsets.size() < n_batches) {
        offsets.push_back(offsets.back());
    }

    // forward on mini-batches
    for (int p = 0; p < (int)offsets.size() - 1; ++p) {
        const int first           = offsets[p];
        const int last            = offsets[p + 1];
        const int mini_batch_size = last - first;
        int*      input_ids       = input_ids_buf_.data();

        BatchedCopy batched_copy;
        int         sum_k = 0;
        for (int i = first; i < last; ++i) {
            input_ids = batched_copy.Add(input_d_ptrs[i], h_input_length_buf_[i], input_ids);
            if (h_input_length_buf_[i] > 1) {
                sum_k += state_->h_context_length[i];
            }
        }
        int sum_q = input_ids - input_ids_buf_.data();

        batched_copy.Submit(stream_);

        const int dc_batch_size = p ? 0 : pf_offset;
        const int pf_batch_size = mini_batch_size - dc_batch_size;

        if (tp_rank_ == 0) {
            if (pf_batch_size) {
                const auto max_q =
                    *std::max_element(h_input_length_buf_.data() + first, h_input_length_buf_.data() + last);
                const auto max_k =
                    *std::max_element(state_->h_context_length.data() + first, state_->h_context_length.data() + last);
                TM_LOG_INFO("[Forward] [%d, %d), dc=%d, pf=%d, sum_q=%d, sum_k=%d, max_q=%d, max_k=%d",
                            first,
                            last,
                            dc_batch_size,
                            pf_batch_size,
                            sum_q,
                            sum_k,
                            max_q,
                            max_k);
            }
            if (eagle_enabled) {
                const int max_engine_tokens = model_->eagleMaxEngineTokensPerStep();
                TM_LOG_INFO(
                    "[LlamaBatch][EAGLE] candidate engine step: step=%d, "
                    "range=[%d,%d), dc=%d, pf=%d, engine_tokens=%d, ctx_tokens=%d, engine_budget=%d",
                    g.step,
                    first,
                    last,
                    dc_batch_size,
                    pf_batch_size,
                    sum_q,
                    sum_k,
                    max_engine_tokens);
            }
        }

        MropeRope mrope;
        if (model_->attn_param_.rope.type == RopeType::kMrope) {
            mrope.stride         = state_->mrope.position_ids.shape(1);
            mrope.position_ids   = state_->mrope.position_ids.slice(first, mini_batch_size);
            mrope.position_delta = state_->mrope.position_delta.slice(first, mini_batch_size);
            mrope.length         = state_->mrope.length.slice(first, mini_batch_size);
        }

        // Synchronize batch token num with sync DP ranks
        auto local_token_nums = AllGather(comm_.h_dp_group, sum_q);
        auto global_token_num = std::accumulate(local_token_nums.begin(), local_token_nums.end(), 0);

        auto hidden_states = symm_hidden_states_buf_.slice(0, global_token_num);

        model_->Forward(input_ids_buf_.slice(0, sum_q),  // temp
                        hidden_states,                   // temp
                        decoder_output_buf_.slice(first, mini_batch_size),
                        block_ptrs_,
                        cu_block_counts_.slice(first, mini_batch_size + 1),
                        h_input_length_buf_.slice(first, mini_batch_size),
                        state_->h_context_length.slice(first, mini_batch_size),
                        rope_theta_.slice(first, mini_batch_size),
                        &mrope,
                        symm_partial_ML_,
                        finished_buf_.slice(first, mini_batch_size),
                        Buffer(local_token_nums.data(), local_token_nums.size(), kCPU),
                        lora_mask_buf_,
                        dc_batch_size,
                        pf_batch_size,
                        state_->sequences.data() + first);

        ComputeAndOutputLogits(hidden_states, first, last);
        OutputLastHiddenState(hidden_states, first, last);
    }

    if (const auto bsz = active_size - g.partial; bsz > 0) {

        auto decoder_features = decoder_output_buf_.slice(0, bsz);
        auto logits           = model_->postDecodeEmbedding(decoder_features, symm_logits_buf_.buffer());

        // AnomalyHandler::instance().FixLogits(logits.data<nv_bfloat16>(), bsz, 1);

        OutputLogits(logits, 0, bsz, GenerationConfig::kGeneration);

        auto sampling_logits = sampling_logits_.slice(0, bsz);
        if (sampling_logits.dtype() != kFloat32) {
            TM_LOG_WARNING(
                "[LlamaBatch][EAGLE][fallback] sampling_logits_ dtype=%s is not FP32; "
                "treating this step as single-token decode while preserving EAGLE mode.",
                to_string(sampling_logits.dtype()));
            // In this rare case we skip the speculative path for this step
            // but keep EAGLE enabled for the engine so subsequent steps can
            // still use speculative decoding once logits numerics are valid.
        }
        else {
            invokeCastFloat2D(logits, sampling_logits, stream_);
            sync_check_cuda_error();
        }

        TM_CHECK_GE(g.step, 0);

        if (!g.skip_init_sampling) {
            InitializeSampling(g);
        }

        // Detect a strictly greedy-decoding batch so that EAGLE's view
        // of target tokens can be aligned with DynamicDecode's commit
        // token in offline validation scenarios. Behaviour for non-greedy
        // configs remains unchanged.
        auto is_greedy_batch = [&]() -> bool {
            for (int i = 0; i < bsz; ++i) {
                const auto& cfg = state_->requests[i]->gen_cfg;
                if (cfg.temperature != 0.0f) {
                    return false;
                }
                if (cfg.top_k > 1) {
                    return false;
                }
                if (cfg.top_p < 1.0f) {
                    return false;
                }
            }
            return true;
        };

        bool output_logprobs = [&] {
            for (int i = 0; i < bsz; ++i) {
                if (state_->requests[i]->gen_cfg.output_logprobs) {
                    return true;
                }
            }
            return false;
        }();

        // Per-step EAGLE planning: decide how many speculative tokens per
        // sequence we would like to attempt this step and clamp per-slot
        // budgets by remaining max_new_tokens and per-slot kill switches.
        if (eagle_enabled && tp_rank_ == 0) {
            NvtxScope eagle_scope("eagle_draft_step");

            const int draft_batch_size      = bsz;
            const int draft_tokens_per_seq  = compute_eagle_draft_tokens_per_seq(draft_batch_size);
            eagle_last_step_                = g.step;

            TM_LOG_INFO(
                "[LlamaBatch][EAGLE] step=%d, decode_batch=%d, planned_draft_tokens_per_seq=%d",
                g.step,
                draft_batch_size,
                draft_tokens_per_seq);

            if (draft_tokens_per_seq <= 0) {
                TM_LOG_WARNING("[LlamaBatch][EAGLE] step=%d, decode_batch=%d, tokens_per_seq=0; "
                               "skipping speculative draft planning for this step",
                               g.step,
                               draft_batch_size);
            }
            else {
                const int tokens_per_seq = std::max(1, draft_tokens_per_seq);
                eagle_planned_tokens_per_seq_.assign(draft_batch_size, tokens_per_seq);
                for (int i = 0; i < draft_batch_size; ++i) {
                    int planned = tokens_per_seq;

                    auto&          req = state_->requests[i];
                    const Sequence* seq = state_->sequences[i];
                    if (req && seq) {
                        const int max_new_tokens = req->gen_cfg.max_new_tokens;
                        if (max_new_tokens > 0) {
                            int prompt_len = 0;
                            auto it        = req->inputs.find("input_ids");
                            if (it != req->inputs.end()) {
                                prompt_len = it->second.shape(0);
                            }
                            const int total_tokens =
                                static_cast<int>(seq->tokens.size());
                            const int generated_tokens =
                                std::max(0, total_tokens - prompt_len);
                            int remaining = max_new_tokens - generated_tokens;
                            if (remaining <= 1) {
                                // This request is essentially at the end of
                                // its max_new_tokens budget. Disable
                                // multi-token for this slot and fall back to
                                // single-token semantics.
                                disableEagleMultitokenForSlot(
                                    i, "remaining max_new_tokens <= 1");
                                planned = 1;
                            }
                            else {
                                planned = std::min(planned, remaining);
                            }
                        }
                    }

                    if (!isEagleMultiTokenSlotEnabled(i)) {
                        planned = 1;
                    }

                    eagle_planned_tokens_per_seq_[i] = std::max(1, planned);
                }
            }
        }

        // stop-words & bad-words require the matched tokens to be contiguous, so item size > 1 is not supported
        if (eagle_enabled && model_->isEagleEnabled()) {
            // Populate d_batch_slots for EagleOrchestrator
            // Copy current active slots to device buffer
            std::vector<int> current_slots;
            current_slots.reserve(batch_size);
            for (int i = 0; i < batch_size; ++i) {
                current_slots.push_back(state_->eagle_slots[i]);
            }
            if (!current_slots.empty()) {
                core::Copy(current_slots.data(), current_slots.size(), eagle_kv_batch_slots_.data());
            }

            LlamaV2::SpecContext spec_ctx{};
            spec_ctx.d_batch_slots = eagle_kv_batch_slots_.data();
            spec_ctx.max_decoding_tokens_step = param_.spec_max_decoding_tokens;
            spec_ctx.sequences                = state_->sequences.data();
            spec_ctx.d_sequence_lengths       = sequence_lengths_.data();
            spec_ctx.planned_tokens_per_seq   =
                eagle_planned_tokens_per_seq_.empty() ? nullptr : eagle_planned_tokens_per_seq_.data();
            spec_ctx.enable_eagle             = true;
            spec_ctx.enable_eagle_target_tree = param_.enable_eagle_target_tree && model_->isTargetTreeDecodeEnabled();
            // Populate per-slot end_ids (first EOS id per request or -1).
            {
                std::vector<int> host_end_ids(max_batch_size_, -1);
                for (int i = 0; i < batch_size; ++i) {
                    const auto* req = state_->requests[i].get();
                    if (req && !req->gen_cfg.eos_ids.empty()) {
                        host_end_ids[i] = req->gen_cfg.eos_ids.front();
                    }
                }
                if (isEagleDebugEnabled() && tp_rank_ == 0
                    && turbomind::isEnvVarEnabled("LMDEPLOY_EAGLE_COPY_DEBUG")) {
                    TM_LOG_WARNING(
                        "[LlamaBatch][EAGLE][CopySITE:eagle_end_ids_step] step=%d batch=%d a=%p b=%p n=%d",
                        g.step,
                        batch_size,
                        host_end_ids.data(),
                        eagle_end_ids_.data(),
                        batch_size);
                }
                core::Copy(host_end_ids.data(), batch_size, eagle_end_ids_.data());
                spec_ctx.d_end_ids = eagle_end_ids_.data();
            }
            // Posterior/typical gating knobs (per TRT).
            {
                std::vector<float> host_thr(max_batch_size_, 1.0f);
                std::vector<float> host_alpha(batch_size, 1.0f);
                std::vector<float> host_temp(batch_size, 1.0f);
                for (int i = 0; i < batch_size; ++i) {
                    const auto* req = state_->requests[i].get();
                    if (!req) {
                        continue;
                    }
                    host_temp[i] = req->gen_cfg.temperature;
                    // If posterior params are absent, default to 1.0 (no masking).
                    // if (!req->gen_cfg.posterior_thresholds.empty()) {
                    //     host_thr[i] = req->gen_cfg.posterior_thresholds.front();
                    // }
                    // if (!req->gen_cfg.posterior_alphas.empty()) {
                    //     host_alpha[i] = req->gen_cfg.posterior_alphas.front();
                    // }
                }
                if (isEagleDebugEnabled() && tp_rank_ == 0) {
                    TM_LOG_WARNING(
                        "[LlamaBatch][EAGLE][CopySITE:eagle_posterior_thr_step] step=%d batch=%d a=%p b=%p n=%d",
                        g.step,
                        batch_size,
                        host_thr.data(),
                        eagle_posterior_thresholds_.data(),
                        batch_size);
                    TM_LOG_WARNING(
                        "[LlamaBatch][EAGLE][CopySITE:eagle_posterior_alpha_step] step=%d batch=%d a=%p b=%p n=%d",
                        g.step,
                        batch_size,
                        host_alpha.data(),
                        eagle_posterior_alphas_.data(),
                        batch_size);
                    TM_LOG_WARNING(
                        "[LlamaBatch][EAGLE][CopySITE:eagle_temperature_step] step=%d batch=%d a=%p b=%p n=%d",
                        g.step,
                        batch_size,
                        host_temp.data(),
                        eagle_temperatures_.data(),
                        batch_size);
                }
                core::Copy(host_thr.data(), batch_size, eagle_posterior_thresholds_.data());
                core::Copy(host_alpha.data(), batch_size, eagle_posterior_alphas_.data());
                core::Copy(host_temp.data(), batch_size, eagle_temperatures_.data());
                spec_ctx.d_posterior_thresholds = eagle_posterior_thresholds_.data();
                spec_ctx.d_posterior_alphas     = eagle_posterior_alphas_.data();
                spec_ctx.d_temperatures         = eagle_temperatures_.data();
            }

            model_->dynamicDecodeWithSpecMulti(g,
                                               token_ids_buf_,
                                               finished_buf_,
                                               sequence_lengths_,
                                               state_->curand_state,
                                               decoder_features,
                                               sampling_logits,  // <- batch size indicator
                                               seq_limit_len_,
                                               init_context_length_,
                                               state_->h_context_length,
                                               state_->h_prompt_length,
                                               output_logprobs ? sampled_logprobs_ : Buffer{},  // <- indicator
                                               sampled_indexes_,
                                               sampled_nums_,
                                               g.max_init_ctx_len,
                                               spec_ctx);
        }
        else {
            model_->dynamicDecode(token_ids_buf_,
                                  finished_buf_,
                                  sequence_lengths_,
                                  state_->curand_state,
                                  sampling_logits,  // <- batch size indicator
                                  seq_limit_len_,
                                  init_context_length_,
                                  state_->h_context_length,
                                  state_->h_prompt_length,
                                  output_logprobs ? sampled_logprobs_ : Buffer{},  // <- indicator
                                  sampled_indexes_,
                                  sampled_nums_,
                                  g.step,
                                  g.max_init_ctx_len);
        }

        // Integrate EAGLE acceptance into decode state at the metrics level
        // and, when enabled, perform experimental multi-token advancement and
        // KV cache rewind.
        if (eagle_enabled && tp_rank_ == 0) {
            static std::vector<int>  h_token_ids_storage;
            static std::vector<bool> h_finished_slots_storage;
            static std::vector<int>  kv_draft_lengths;
            static std::vector<int>  kv_accepted_lengths;

            // Fetch per-step acceptance summary from the fused EAGLE path.
            model_->getEagleAcceptanceForStep(eagle_accepted_lens, eagle_accepted_tokens, eagle_tokens_per_seq);

            if (eagle_tokens_per_seq > 0 && !eagle_accepted_lens.empty()) {
                collectEagleStepHostState(g, bsz, h_token_ids_storage, h_finished_slots_storage);

                const int max_path_len = param_.spec_max_draft_path_len;
                updateEagleMetricsAndKVLengths(g,
                                               h_token_ids_storage,
                                               h_finished_slots_storage,
                                               bsz,
                                               max_path_len,
                                               eagle_tokens_per_seq,
                                               eagle_accepted_lens,
                                               eagle_accepted_tokens,
                                               kv_draft_lengths,
                                               kv_accepted_lengths);
                if (isEagleMultiTokenStepEnabled(g)) {
                    const int step_max_extra = model_->eagle_step_max_extra_;
                    if (step_max_extra > 0) {
                        g.step += step_max_extra;
                    }
                    runEagleKVRewind(kv_draft_lengths, kv_accepted_lengths, bsz, g);
                }
            }
        }
    }

    std::fill(h_input_length_buf_.data(), h_input_length_buf_.data() + active_size, 0);

    // `SequenceManager` needs real-time value of cache length
    for (int i = 0; i < active_size; ++i) {
        FT_CHECK((bool)state_->requests[i]);
        FT_CHECK(state_->sequences[i]);
        state_->sequences[i]->cache_len += state_->sequences[i]->input_length;
    }

    AnomalyHandler::instance().Summarize([&](const int* is_anomaly, int batch_size) {
        for (int i = 0; i < batch_size; ++i) {
            if (is_anomaly[i]) {
                TM_LOG_WARNING("[Forward] Abnormal logits detected for request (%s)",
                               std::to_string(state_->sequences[i]->id).c_str());
                state_->errors[i] = Request::kFail;
            }
        }
    });
    AnomalyHandler::instance().Reset();

    if (debug_ && tp_rank_ == 0) {
        std::vector<int> curr(active_size);
        if (isEagleDebugEnabled()) {
            TM_LOG_WARNING(
                "[LlamaBatch][EAGLE][CopySITE:debug_step_tokens] step=%d active_size=%d src_row=%d a=%p b=%p n=%d",
                g.step,
                active_size,
                g.step,
                token_ids_buf_.data() + g.step * active_size,
                curr.data(),
                active_size);
        }
        core::Copy(token_ids_buf_.data() + g.step * active_size, active_size, curr.data());
        cudaStreamSynchronize(stream_);
        std::stringstream scurr;
        for (int k = 0; k < curr.size(); ++k) {
            scurr << std::setw(10) << curr[k];
        }
        TM_LOG_INFO("[Forward] step = %d, [%s]", g.step - 1, scurr.str().c_str());
    }

    ////////////////////////////////////////////////
    /// ! increase the counters
    g.step += 1;

    return true;
}

void LlamaBatch::advanceSequencesByEagleAcceptance(const std::vector<int>&  dynamic_tokens,
                                                   const std::vector<bool>& finished_slots,
                                                   int                      batch_size,
                                                   int                      eagle_tokens_per_seq,
                                                   const std::vector<int>&  eagle_accepted_lens,
                                                   const std::vector<int>&  eagle_accepted_tokens,
                                                   GenerationState&         g)
{
    // Multi-token advancement is currently experimental and must preserve
    // baseline behaviour when disabled. Guard aggressively and fall back to
    // single-token semantics on any invariant violation.
    if (!isEagleMultiTokenStepEnabled(g)) {
        return;
    }

    if (eagle_tokens_per_seq <= 1 || eagle_accepted_lens.empty()) {
        return;
    }

    const int max_path_len = param_.spec_max_draft_path_len;
    if (max_path_len <= 0) {
        return;
    }

    if (batch_size <= 0) {
        return;
    }

    const int active_size = state_->active_size;
    if (batch_size > active_size) {
        TM_LOG_WARNING("[LlamaBatch][EAGLE][fallback] batch_size=%d exceeds active_size=%d; skipping multi-token advance",
                       batch_size,
                       active_size);
        return;
    }

    if (dynamic_tokens.size() < static_cast<size_t>(batch_size)) {
        TM_LOG_WARNING(
            "[LlamaBatch][EAGLE][fallback] dynamic_tokens.size()=%zu < batch_size=%d; skipping multi-token advance",
            dynamic_tokens.size(),
            batch_size);
        return;
    }

    const int base_step = g.step - 1;
    const int max_steps = session_len_ * 2;

    if (base_step < 0 || base_step >= max_steps) {
        TM_LOG_WARNING("[LlamaBatch][EAGLE][fallback] Invalid base_step=%d (max_steps=%d); skipping multi-token advance",
                       base_step,
                       max_steps);
        return;
    }

    int              max_extra     = 0;
    std::vector<int> extra_per_seq(batch_size, 0);
    std::vector<int> eos_extra_index(batch_size, -1);

    for (int i = 0; i < batch_size; ++i) {
        // Skip sequences that DynamicDecode has already marked finished on
        // this step; their terminal token (e.g. EOS or max_new_tokens) must
        // remain the last token in the sequence.
        if (i < static_cast<int>(finished_slots.size()) && finished_slots[i]) {
            continue;
        }

        // Per-request kill switch: once a hard invariant is violated for a
        // slot, multi-token advance is permanently disabled for that slot.
        if (!isEagleMultiTokenSlotEnabled(i)) {
            continue;
        }

        const int len = (i < static_cast<int>(eagle_accepted_lens.size())) ? eagle_accepted_lens[i] : 0;

        // `accepted_len == 0` means the draft was fully rejected. This is
        // valid but offers no multi-token benefit, so we simply keep the
        // single-token behaviour for this sequence.
        if (len <= 1) {
            continue;
        }

        const int max_path_len = param_.spec_max_draft_path_len;
        if (len > max_path_len) {
            disableEagleMultitokenForSlot(i, "accepted_len exceeds max_path_len");
            continue;
        }

        // Per-sequence planned draft budget for this step. Default to the
        // global scalar when the per-sequence vector is empty or out of
        // range, but drive all per-slot bounds from the vector when
        // available.
        int planned_tokens_per_seq = eagle_tokens_per_seq;
        if (!eagle_planned_tokens_per_seq_.empty()
            && i < static_cast<int>(eagle_planned_tokens_per_seq_.size())) {
            planned_tokens_per_seq = eagle_planned_tokens_per_seq_[i];
        }
        if (planned_tokens_per_seq <= 0) {
            planned_tokens_per_seq = 1;
        }

        if (len > planned_tokens_per_seq) {
            disableEagleMultitokenForSlot(i, "accepted_len exceeds planned_tokens_per_seq");
            continue;
        }

        const int token_offset = i * max_path_len;
        if (token_offset < 0 || token_offset + len > static_cast<int>(eagle_accepted_tokens.size())) {
            disableEagleMultitokenForSlot(i, "accepted_len/token_offset out of range for accepted_tokens");
            continue;
        }

        const int* seq_tokens = eagle_accepted_tokens.data() + token_offset;
        const int  committed  = dynamic_tokens[i];

        // For TurboMind's current EAGLE integration we do not require the
        // first accepted token to exactly match DynamicDecode's committed
        // token. Sampling layers (e.g. repetition penalties, guidance) can
        // legitimately cause divergence between the greedy top-1 from the
        // raw logits (used to build target_tokens) and the final decode
        // decision. We still treat `committed` as the first token in the
        // sequence and only append *extra* accepted tokens (positions
        // 1..len-1) below.

        // If an accepted extra token matches EOS, clamp extras so that we
        // commit tokens up to and including that EOS and mark the sequence
        // finished after multi-token advance.
        if (i < state_->active_size && state_->requests[i]) {
            const auto& eos_ids = state_->requests[i]->gen_cfg.eos_ids;
            if (!eos_ids.empty()) {
                for (int t = 1; t < len; ++t) {
                    const int tok = seq_tokens[t];
                    if (std::find(eos_ids.begin(), eos_ids.end(), tok) != eos_ids.end()) {
                        eos_extra_index[i] = t;
                        break;
                    }
                }
            }
        }

        // Commit-time safety net: ensure we do not append more extra tokens
        // for this request than its remaining max_new_tokens budget allows,
        // even if planning or acceptance overestimated what is safe.
        int max_extra_allowed = std::numeric_limits<int>::max();
        if (i < state_->active_size) {
            auto&          req = state_->requests[i];
            const Sequence* seq = state_->sequences[i];
            if (req && seq) {
                const int max_new_tokens = req->gen_cfg.max_new_tokens;
                if (max_new_tokens > 0) {
                    int prompt_len = 0;
                    auto it        = req->inputs.find("input_ids");
                    if (it != req->inputs.end()) {
                        prompt_len = it->second.shape(0);
                    }
                    const int total_tokens =
                        static_cast<int>(seq->tokens.size());
                    const int generated_tokens =
                        std::max(0, total_tokens - prompt_len);
                    int remaining = max_new_tokens - generated_tokens;
                    int local_max_extra = remaining - 1;  // one token is the committed DD token
                    if (local_max_extra <= 0) {
                        disableEagleMultitokenForSlot(
                            i,
                            "no remaining max_new_tokens budget for extra tokens at commit time");
                        continue;
                    }
                    max_extra_allowed = local_max_extra;
                }
            }
        }

        int effective_len = len;
        if (eos_extra_index[i] >= 1) {
            effective_len = eos_extra_index[i] + 1;
        }

        int extra = effective_len - 1;
        if (extra <= 0) {
            continue;
        }

        if (max_extra_allowed != std::numeric_limits<int>::max()) {
            extra = std::min(extra, max_extra_allowed);
            if (extra <= 0) {
                continue;
            }
        }

        if (base_step + extra >= max_steps) {
            extra = std::max(0, max_steps - base_step - 1);
            if (extra <= 0) {
                TM_LOG_WARNING(
                    "[LlamaBatch][EAGLE][fallback] step=%d seq=%d extra tokens would exceed buffer (max_steps=%d); "
                    "skipping",
                    g.step,
                    i,
                    max_steps);
                continue;
            }
        }

        extra_per_seq[i] = extra;
        max_extra        = std::max(max_extra, extra);
    }

    if (max_extra <= 0) {
        return;
    }

    if (isEagleDebugEnabled() && tp_rank_ == 0) {
        for (int i = 0; i < batch_size; ++i) {
            const int extra = extra_per_seq[i];
            if (extra <= 0) {
                continue;
            }
            const int len_i =
                (i < static_cast<int>(eagle_accepted_lens.size())) ? eagle_accepted_lens[i] : 0;
            // BOUNDS CHECK: Prevent buffer overflow when accessing eagle_accepted_tokens
            const int token_offset = i * max_path_len;
            if (token_offset + len_i > static_cast<int>(eagle_accepted_tokens.size())) {
                TM_LOG_WARNING(
                    "[LlamaBatch][EAGLE] step=%d, seq=%d, token_offset=%d + len=%d > tokens.size()=%zu; skipping debug",
                    g.step, i, token_offset, len_i, eagle_accepted_tokens.size());
                continue;
            }
            const int* seq_tokens = eagle_accepted_tokens.data() + token_offset;
            std::ostringstream accepted_ss;
            for (int e = 0; e < len_i; ++e) {
                if (e) {
                    accepted_ss << ",";
                }
                accepted_ss << seq_tokens[e];
            }
            TM_LOG_INFO(
                "[LlamaBatch][EAGLE] step=%d, seq=%d, committing %d extra accepted tokens (base_step=%d), "
                "accepted_tokens=[%s]",
                g.step,
                i,
                extra,
                base_step,
                accepted_ss.str().c_str());
        }
    }

    // Write extra accepted tokens directly into token_ids_buf_ on device,
    // without copying the existing slice back to host. Each extra token for
    // sequence i is placed at row (base_step + 1 + e) and column i.
    for (int i = 0; i < batch_size; ++i) {
        int extra = extra_per_seq[i];
        if (extra <= 0) {
            continue;
        }

        const int token_offset = i * max_path_len;
        // BOUNDS CHECK: Prevent buffer overflow - critical for heap safety
        // The access pattern is seq_tokens[1 + e] where e goes from 0 to extra-1
        // So we need token_offset + 1 + (extra - 1) = token_offset + extra to be valid
        if (token_offset + extra > static_cast<int>(eagle_accepted_tokens.size())) {
            TM_LOG_WARNING(
                "[LlamaBatch][EAGLE] step=%d, seq=%d, token_offset=%d + extra=%d > tokens.size()=%zu; "
                "SKIPPING to prevent heap corruption",
                g.step, i, token_offset, extra, eagle_accepted_tokens.size());
            continue;
        }
        const int* seq_tokens  = eagle_accepted_tokens.data() + token_offset;

        for (int e = 0; e < extra; ++e) {
            const int row_index = base_step + 1 + e;
            const int col_index = i;
            const int value     = seq_tokens[1 + e];
            int*       dst      = token_ids_buf_.data() + row_index * active_size + col_index;
            check_cuda_error(cudaMemcpyAsync(dst, &value, sizeof(int), cudaMemcpyHostToDevice, stream_));
        }
    }

    // Update sequence_lengths_ so that DynamicDecode sees the extra accepted
    // tokens as part of the sequence history on the next step. Only the
    // first `batch_size` entries are relevant for this mini-batch.
    std::vector<int> h_seq_lengths(batch_size);
    if (isEagleDebugEnabled() && tp_rank_ == 0) {
        TM_LOG_WARNING(
            "[LlamaBatch][EAGLE][CopySITE:seq_lengths_d2h] step=%d batch=%d a=%p b=%p n=%d",
            g.step,
            batch_size,
            sequence_lengths_.data(),
            h_seq_lengths.data(),
            batch_size);
    }
    core::Copy(sequence_lengths_.data(), batch_size, h_seq_lengths.data());
    check_cuda_error(cudaStreamSynchronize(stream_));

    for (int i = 0; i < batch_size; ++i) {
        const int extra = extra_per_seq[i];
        if (extra > 0) {
            h_seq_lengths[i] += extra;
        }
    }

    if (isEagleDebugEnabled() && tp_rank_ == 0) {
        TM_LOG_WARNING(
            "[LlamaBatch][EAGLE][CopySITE:seq_lengths_h2d] step=%d batch=%d a=%p b=%p n=%d",
            g.step,
            batch_size,
            h_seq_lengths.data(),
            sequence_lengths_.data(),
            batch_size);
    }
    core::Copy(h_seq_lengths.data(), batch_size, sequence_lengths_.data());
    check_cuda_error(cudaStreamSynchronize(stream_));

    // If an EOS token was committed as part of the extra accepted tokens
    // for this sequence, mark it finished in the device-side finished_buf_
    // so subsequent decode steps and Finish() see the terminal state.
    for (int i = 0; i < batch_size; ++i) {
        const int eos_idx = eos_extra_index[i];
        const int extra   = extra_per_seq[i];
        if (eos_idx >= 1 && extra >= eos_idx) {
            const bool done = true;
            check_cuda_error(cudaMemcpyAsync(
                finished_buf_.data() + i, &done, sizeof(bool), cudaMemcpyHostToDevice, stream_));
        }
    }

    // Advance the global generation step by the maximum number of extra
    // accepted tokens so that subsequent steps and gatherOutput see the
    // extended token range.
    g.step += max_extra;

    if (debug_ && tp_rank_ == 0) {
        // Sanity-check that sequence lengths for sequences that actually
        // received extra tokens do not outrun the global time index.
        for (int i = 0; i < batch_size; ++i) {
            const int extra = extra_per_seq[i];
            if (extra <= 0) {
                continue;
            }
            const int seq_len = h_seq_lengths[i];
            if (seq_len > g.step) {
                disableEagleMultitokenForSlot(i, "sequence_length exceeds global step after multi-token advance");
            }
        }
    }
}

namespace {

template<class First, class Last>
std::string Join(First first, Last last, const std::string& delim)
{
    if (first == last) {
        return {};
    }
    std::ostringstream oss;
    oss << *first++;
    while (first != last) {
        oss << delim << *first++;
    }
    return oss.str();
}

struct TuningContext {
    LlamaLinear& linear_;
    cudaStream_t stream_;
    TuningContext(LlamaLinear& linear, cudaStream_t stream): linear_{linear}, stream_{stream}
    {
        isTuning() = true;
        linear_.set_measure(true);
    }
    ~TuningContext()
    {
        linear_.set_measure(false);
        isTuning() = false;
    }
};

}  // namespace

void LlamaBatch::Warmup()
{
    auto& linear = *context_->linear;
    if (auto str = std::getenv("TM_GEMM_IMPORT")) {
        std::ifstream ifs(str);
        const int     n_imported = linear.Import(ifs);
        if (tp_rank_ == 0) {
            TM_LOG_INFO("[Gemm2] %d records imported", n_imported);
        }
        return;
    }

    std::vector<int> bss = linear.GetTuningSeq();
    if (bss.empty()) {
        bss = gemm::GenerateTuningSequence(gemm::GetDefaultTuningGenerators());
    }

    // remove bs that is too large
    bss.erase(std::remove_if(bss.begin(), bss.end(), [&](auto x) { return x > max_forward_token_num_; }), bss.end());

    if (bss.empty() || bss.back() < max_forward_token_num_) {
        bss.push_back(max_forward_token_num_);
    }

    if (tp_rank_ == 0) {
        auto str = Join(bss.begin(), bss.end(), ", ");
        TM_LOG_INFO("[Gemm2] Tuning sequence: %s", str.c_str());
    }

    if (!bss.empty()) {
        const auto                         max_bs = *std::max_element(bss.begin(), bss.end());
        Buffer_<int>                       input_ids(max_bs, kCPU);
        Buffer_<int>                       input_ids_buf(max_bs, kDEVICE);
        std::mt19937                       g{};
        std::uniform_int_distribution<int> d{0, (int)model_->vocab_size_ - 1};
        for (auto& x : input_ids) {
            x = d(g);
        }
        Copy(input_ids, input_ids_buf);
        check_cuda_error(cudaStreamSynchronize(stream_));

        TuningContext context{linear, stream_};

        auto tick = std::chrono::steady_clock::now();

        /// NOTE: No explicit barrier can be used here as internal threads are waiting on it now
        for (auto token_num : bss) {
            if (tp_rank_ == 0) {
                TM_LOG_INFO("[Gemm2] %d", token_num);
            }

            int  input_length     = token_num;
            auto local_token_nums = AllGather(comm_.h_dp_group, token_num);

            const auto bsz = 1;

            // A single sequence containing `token_num` prefill tokens
            model_->Forward(input_ids_buf.slice(0, token_num),
                            symm_hidden_states_buf_.slice(0, token_num * param_.attn_dp_size),
                            decoder_output_buf_.slice(0, bsz),
                            block_ptrs_,
                            cu_block_counts_.slice(0, bsz + 1),
                            Buffer{&input_length, 1, kCPU},
                            Buffer{&input_length, 1, kCPU},
                            rope_theta_.slice(0, bsz),
                            nullptr,  // mrope
                            symm_partial_ML_,
                            finished_buf_.slice(0, bsz),
                            Buffer{local_token_nums.data(), (int)local_token_nums.size(), kCPU},
                            Buffer{},
                            0,
                            bsz,
                            nullptr);
        }

        auto tock = std::chrono::steady_clock::now();

        if (tp_rank_ == 0) {
            TM_LOG_INFO("[Gemm2] Tuning finished in %.2f seconds.",
                        std::chrono::duration<float, std::ratio<1, 1>>(tock - tick).count());
        }
    }

    // This will catch async errors during tuning
    check_cuda_error(cudaStreamSynchronize(stream_));

    // Only rank-0 exports the dispatch cache
    if (tp_rank_ == 0) {
        if (auto path = std::getenv("TM_GEMM_EXPORT")) {
            std::ofstream ofs(path);
            const auto    n_records = context_->linear->Export(ofs);
            TM_LOG_INFO("[Gemm2] %d records exported.", n_records);
        }
    }
}

void LlamaBatch::InitializeBufferAndKVCache()
{
    // initialize kvcache, BatchState and persist buffers
    core::ContextGuard guard{context_->core_stream, context_->allocator, Allocator{kCPUpinned}};

    const auto cache_block_seq_len = model_->attn_param_.cache_block_seq_len;

    const int dbits = byte_size(data_type_, 8);

    const auto quant_policy = model_->param_.quant_policy;
    // Map quant_policy to actual KV element bit width
    int elem_bits = dbits;  // default: unquantized
    if (quant_policy & QuantPolicy::kCacheKVInt8) {
        elem_bits = 8;
    } else if (quant_policy & QuantPolicy::kCacheKVInt4) {
        elem_bits = 4;
    } else if (quant_policy & QuantPolicy::kCacheKVFP4) {
        elem_bits = 4;  // FP4 is 4 bits per element
    }

    SequenceManager::BlockConfig block_config{
        (int)model_->size_per_head_,
        (int)model_->local_kv_head_num_,
        cache_block_seq_len,
        elem_bits == dbits ? 0 : dbits,
        elem_bits,
    };

    const auto get_free_size = [&] {  //
        size_t free{}, total{};
        check_cuda_error(cudaMemGetInfo(&free, &total));
        return AllReduce(model_->comm_->h_tp_group, free, comm::RedOp::kMin);
    };

    // Optionally over‑provision KV blocks when EAGLE speculative decoding
    // is enabled so there is headroom for draft tree tokens in addition to
    // committed tokens. The factor is conservative and kept close to 1.0
    // to avoid surprising memory usage spikes.
    double cache_block_budget = param_.cache_max_block_count;
    if (param_.enable_speculative_decoding
        && (param_.spec_method == "eagle" || param_.spec_method == "eagle3")) {
        const double max_decoding_tokens =
            std::max(0, param_.spec_max_decoding_tokens);
        const double max_path_len =
            std::max(0, param_.spec_max_draft_path_len);
        const double session_len = std::max(1, param_.session_len);

        // Rough upper bound on extra transient KV usage from EAGLE per
        // sequence relative to the session length.
        const double extra_tokens = max_decoding_tokens + max_path_len;
        double       eagle_factor = 1.0 + extra_tokens / session_len;
        // Keep factor within a reasonable envelope.
        eagle_factor = std::min(eagle_factor, 2.0);

        cache_block_budget *= eagle_factor;

        if (tp_rank_ == 0) {
            TM_LOG_INFO("[LlamaBatch][EAGLE] KV over‑provisioning enabled: "
                        "base_cache_max_block_count=%.3f, eagle_factor=%.3f, "
                        "effective_cache_block_budget=%.3f",
                        param_.cache_max_block_count,
                        eagle_factor,
                        cache_block_budget);
        }
    }

    sequence_manager_.reset(new SequenceManager{model_->layer_num_,
                                                block_config,
                                                cache_block_budget,
                                                param_.cache_chunk_size,
                                                param_.enable_prefix_caching,
                                                tp_rank_,
                                                param_.attn_cp_size,
                                                core::Context::alloc(kDEVICE),
                                                get_free_size});

    // Cache basic KV layout parameters for EAGLE KV rewind integration.
    kv_block_size_         = cache_block_seq_len;
    kv_max_blocks_per_seq_ = sequence_manager_->max_block_count();

    const size_t max_session_len = sequence_manager_->max_block_count() * cache_block_seq_len * param_.attn_cp_size;
    if (max_session_len < session_len_) {
        if (tp_rank_ == 0) {
            TM_LOG_WARNING("No enough blocks for `session_len` (%d), `session_len` truncated to %d.",
                           session_len_,
                           max_session_len);
        }
        session_len_ = max_session_len;
    }

    FT_CHECK(max_context_token_num_ >= session_len_);
    FT_CHECK(max_forward_token_num_ >= max_batch_size_);

    for (auto& s : states_) {
        s.requests.resize(max_batch_size_);
        s.sequences.resize(max_batch_size_);
        s.seq_len_limit.resize(max_batch_size_);
        s.errors.resize(max_batch_size_);
    }
    state_    = &states_[0];
    back_     = &states_[1];
    incoming_ = &states_[2];

    AllocSymmBuffers();


    AllocateBuffer(max_batch_size_, session_len_, model_->attn_param_.cache_block_seq_len);

    // Allow the model to observe the live SequenceManager so that
    // specialized decode paths (e.g. EAGLE target-tree decode) can
    // reuse prefix KV block pointers as read-only when building
    // scratch decode passes.
    model_->sequence_manager_ = sequence_manager_.get();
}

void LlamaBatch::FreeBufferAndKVCache()
{
    if (eagle_kv_cache_blocks_) {
        check_cuda_error(cudaFree(eagle_kv_cache_blocks_));
        eagle_kv_cache_blocks_ = nullptr;
    }

    sequence_manager_.reset();

    for (auto& s : states_) {
        s = {};
    }

    FreeSymmBuffers();
    FreeBuffer();

    cudaStreamSynchronize(stream_);
    context_->allocator->trim(0);
}

void* LlamaBatch::SymmAlloc(size_t size, bool register_)
{
    if (auto& comm = model_->comm_->d_comm) {
        auto ptr = comm->Allocate(size);
        if (register_) {
            comm->Register(ptr, size);
        }
        return ptr;
    }
    else {
        return context_->allocator->allocate(size);
    }
}

void LlamaBatch::SymmFree(void* ptr, size_t size, bool deregister)
{
    if (!ptr) {
        return;
    }
    if (auto& comm = comm_.d_comm) {
        if (deregister) {
            comm->Deregister(ptr);
        }
        comm->Free(ptr);
    }
    else {
        context_->allocator->deallocate(ptr, size);
    }
}

void LlamaBatch::DestroyCommunicators()
{
    cudaStreamSynchronize(stream_);
    comm_.h_comm->Sync();

    FreeSymmBuffers();
    comm_.h_comm->Sync();

    cudaStreamSynchronize(stream_);
    comm_.h_comm->Sync();
}

void LlamaBatch::UpdateMetrics()
{
    if (tp_rank_ == 0 && param_.enable_metrics) {
        /*
        // update schedule metrics
        int total_seqs, active_seqs, cached_seqs;
        std::tie(total_seqs, active_seqs, cached_seqs) = sequence_manager_->seq_stats();

        {
            const std::lock_guard<std::mutex> lock(metrics_mutex_);

            schedule_metrics_.total_seqs    = total_seqs;
            schedule_metrics_.active_seqs   = active_seqs;
            schedule_metrics_.waiting_seqs  = total_seqs - active_seqs;
            schedule_metrics_.total_blocks  = sequence_manager_->total_count();
            schedule_metrics_.active_blocks = sequence_manager_->active_count();
            schedule_metrics_.cached_blocks = sequence_manager_->cached_count();
            schedule_metrics_.free_blocks   = sequence_manager_->free_count();
        }

        // update request metrics
        for (int i = 0; i < state_->active_size; ++i) {
            if (!state_->requests[i]) {
                continue;
            }
            auto& metrics = state_->requests[i]->metrics;
            if (!metrics || metrics->scheduled_time != 0) {
                continue;
            }
            metrics->scheduled_time = turbomind::RequestMetrics::timestamp();
        }
        */
    }
}


}  // namespace turbomind
