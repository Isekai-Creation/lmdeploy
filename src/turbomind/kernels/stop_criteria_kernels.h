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
#pragma once

#include <cuda_runtime.h>

namespace turbomind {

void invokeStopWordsCriterion(const int*   output_ids,
                              const int*   parent_ids,
                              const int*   stop_words,
                              bool*        finished,
                              size_t       id_offset,
                              size_t       stop_words_len,
                              int          batch_size,
                              int          beam_width,
                              int          step,
                              cudaStream_t stream);

void invokeLengthCriterion(bool*        finished,  //
                           const int*   sequence_limit_length,
                           int          batch_size,
                           int          beam_width,
                           int          step,
                           cudaStream_t stream);

// Apply forced tail tokens on device, updating output_ids, sequence_length,
// finished flags, and optional per-slot committed_lengths. This mirrors the
// tail loop in DynamicDecodeLayer::ForwardMultiStep but runs entirely on GPU.
void invokeApplyForcedTail(const int*   forced_tokens,         // [batch_size, max_tail_len]
                           const int*   forced_lengths,        // [batch_size]
                           int*         output_ids,            // [max_seq_len, batch_size] (flat)
                           int*         sequence_length,       // [batch_size]
                           bool*        finished,              // [batch_size]
                           const int*   sequence_limit_length, // [batch_size] or nullptr
                           const int*   eos_ids,               // [batch_size, max_eos_per_slot] or nullptr
                           const int*   eos_counts,            // [batch_size] or nullptr
                           int          max_eos_per_slot,
                           int*         committed_lengths,     // [batch_size] or nullptr
                           int          batch_size,
                           int          max_tail_len,
                           int          max_seq_len,
                           int          step,
                           cudaStream_t stream);

}  // namespace turbomind
