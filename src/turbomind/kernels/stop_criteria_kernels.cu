/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/turbomind/kernels/reduce_kernel_utils.cuh"
#include "src/turbomind/kernels/stop_criteria_kernels.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

__global__ void stop_words_criterion(const int* output_ids,
                                     const int* parent_ids,
                                     const int* stop_words,
                                     bool*      finished,
                                     size_t     id_offset,
                                     size_t     stop_words_len,
                                     int        batch_size,
                                     int        beam_width,
                                     int        step)
{
    const int id        = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y / beam_width;
    const int beam_idx  = blockIdx.y % beam_width;

    const int* base_stop_words = stop_words + batch_idx * 2 * stop_words_len;
    const int* base_offsets    = base_stop_words + stop_words_len;

    if (id >= stop_words_len || base_offsets[id] < 0) {
        return;
    }

    const int item_end   = base_offsets[id];
    const int item_start = (id > 0) ? base_offsets[id - 1] : 0;
    const int item_size  = item_end - item_start;

    /* The single-token case unconditionally bans the token */
    bool should_stop = false;

    /* Enough previously generated tokens to look for a match */
    if (step + 1 >= item_size) {
        should_stop            = true;
        int        parent_id   = beam_idx;
        const bool gather_beam = beam_width > 1;

        for (int token_idx = item_size - 1; token_idx >= 0; token_idx--) {
            const int previous_token = output_ids[(step - (item_size - 1) + token_idx) * batch_size * beam_width
                                                  + id_offset + batch_idx * beam_width + parent_id];

            if (previous_token != base_stop_words[item_start + token_idx]) {
                should_stop = false;
                break;
            }
            if (gather_beam) {
                parent_id = parent_ids[(step - (item_size - 1) + token_idx) * beam_width * batch_size + id_offset
                                       + batch_idx * beam_width + parent_id];

                if (parent_id < 0 || parent_id >= beam_width) {
                    should_stop = false;
                    break;
                }
            }
        }
    }

    if (should_stop) {
        finished[batch_idx * beam_width + beam_idx] = true;
    }
}

void invokeStopWordsCriterion(const int*   output_ids,
                              const int*   parent_ids,
                              const int*   stop_words,
                              bool*        finished,
                              size_t       id_offset,
                              size_t       stop_words_len,
                              int          batch_size,
                              int          beam_width,
                              int          step,
                              cudaStream_t stream)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    // Check if we have sampled a word from the stop_words list. If so, stop the sequence.
    dim3 block, grid;
    block.x = min((unsigned long)((stop_words_len + 32 - 1) / 32) * 32, 256UL);
    grid.x  = (stop_words_len + block.x - 1) / block.x;
    grid.y  = batch_size * beam_width;

    stop_words_criterion<<<grid, block, 0, stream>>>(
        output_ids, parent_ids, stop_words, finished, id_offset, stop_words_len, batch_size, beam_width, step);
    sync_check_cuda_error();
}

__global__ void length_criterion(bool*      finished,  //
                                 const int* sequence_limit_length,
                                 int        batch_size,
                                 int        beam_width,
                                 int        step)
{
    for (int index = threadIdx.x; index < batch_size * beam_width; index += blockDim.x) {
        const int batch_idx = index / beam_width;
        finished[index] |= step >= sequence_limit_length[batch_idx];
    }
}

void invokeLengthCriterion(bool*        finished,  //
                           const int*   sequence_limit_length,
                           int          batch_size,
                           int          beam_width,
                           int          step,
                           cudaStream_t stream)
{
    // Check if we have attained the sequence length limit. If so, stop the sequence.
    // In addition, check if all sequences are stopped and return the result in should_stop
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    dim3 block(std::min(512, batch_size * beam_width));
    dim3 grid{1};

    length_criterion<<<grid, block, 0, stream>>>(finished, sequence_limit_length, batch_size, beam_width, step);
}

namespace {

struct TailParams {
    int*       output_ids;
    int*       sequence_length;
    bool*      finished;
    const int* sequence_limit_length;

    const int* forced_tokens;
    const int* forced_lengths;
    int*       committed_lengths;

    const int* eos_ids;
    const int* eos_counts;

    int batch_size;    // logical active slots
    int max_tail_len;
    int max_seq_len;
    int max_eos_per_slot;
    int stride_batch;  // physical stride for output_ids rows
    int step;
};

__global__ void apply_forced_tail_kernel(TailParams params)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params.batch_size) {
        return;
    }

    bool finished = params.finished ? params.finished[i] : false;
    int  seq_len  = params.sequence_length ? params.sequence_length[i] : 0;

    const int limit = params.sequence_limit_length ? params.sequence_limit_length[i] : 0;

    int tail_len = params.forced_lengths ? params.forced_lengths[i] : 0;
    if (finished || tail_len <= 0 || params.max_tail_len <= 0) {
        if (params.committed_lengths) {
            params.committed_lengths[i] = 0;
        }
        return;
    }

    if (tail_len > params.max_tail_len) {
        tail_len = params.max_tail_len;
    }

    int committed = 0;

    for (int t = 0; t < tail_len; ++t) {
        if (finished) {
            break;
        }

        const int pos = params.step + 1 + t;
        if (pos < 0 || pos >= params.max_seq_len) {
            break;
        }

        if (limit > 0 && pos >= limit) {
            finished = true;
            break;
        }

        const int value =
            params.forced_tokens ? params.forced_tokens[i * params.max_tail_len + t] : -1;

        if (params.output_ids) {
            const int stride = params.stride_batch > 0 ? params.stride_batch : params.batch_size;
            params.output_ids[pos * stride + i] = value;
        }

        ++seq_len;
        ++committed;

        if (limit > 0 && seq_len >= limit) {
            finished = true;
            break;
        }

        const int eos_count = params.eos_counts ? params.eos_counts[i] : 0;
        if (params.eos_ids && eos_count > 0 && params.max_eos_per_slot > 0) {
            const int base = i * params.max_eos_per_slot;
            for (int k = 0; k < eos_count && k < params.max_eos_per_slot; ++k) {
                const int eos = params.eos_ids[base + k];
                if (value == eos) {
                    finished = true;
                    break;
                }
            }
        }
    }

    if (params.sequence_length) {
        params.sequence_length[i] = seq_len;
    }
    if (params.finished) {
        params.finished[i] = finished;
    }
    if (params.committed_lengths) {
        params.committed_lengths[i] = committed;
    }
}

}  // namespace

void invokeApplyForcedTail(const int*   forced_tokens,
                           const int*   forced_lengths,
                           int*         output_ids,
                           int*         sequence_length,
                           bool*        finished,
                           const int*   sequence_limit_length,
                           const int*   eos_ids,
                           const int*   eos_counts,
                           int          max_eos_per_slot,
                           int*         committed_lengths,
                           int          batch_size,
                           int          max_tail_len,
                           int          max_seq_len,
                           int          stride_batch,
                           int          step,
                           cudaStream_t stream)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    if (!forced_tokens || !forced_lengths || batch_size <= 0 || max_tail_len <= 0 || max_seq_len <= 0) {
        return;
    }

    TailParams params{};
    params.output_ids            = output_ids;
    params.sequence_length       = sequence_length;
    params.finished              = finished;
    params.sequence_limit_length = sequence_limit_length;
    params.forced_tokens         = forced_tokens;
    params.forced_lengths        = forced_lengths;
    params.committed_lengths     = committed_lengths;
    params.eos_ids               = eos_ids;
    params.eos_counts            = eos_counts;
    params.batch_size            = batch_size;
    params.max_tail_len          = max_tail_len;
    params.max_seq_len           = max_seq_len;
    params.max_eos_per_slot      = max_eos_per_slot;
    params.stride_batch          = stride_batch;
    params.step                  = step;

    const int block = 256;
    const int grid  = (batch_size + block - 1) / block;

    apply_forced_tail_kernel<<<grid, block, 0, stream>>>(params);
    sync_check_cuda_error();
}

}  // namespace turbomind
