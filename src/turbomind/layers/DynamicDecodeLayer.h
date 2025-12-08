/*
 * Copyright (c) 2022-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <memory>
#include <vector>

#include "src/turbomind/engine/request.h"
#include "src/turbomind/layers/BaseDynamicDecodeLayer.h"

#include "src/turbomind/core/tensor.h"
#include "src/turbomind/layers/sampling_layers/utils.h"

namespace turbomind {

struct ForcedTailContext {
    const int* forced_tokens{nullptr};
    const int* forced_lengths{nullptr};
    // Per-slot number of tail tokens actually committed by the
    // decode layer. May be null, in which case the tail loop
    // will not report per-slot counts.
    int*       committed_lengths{nullptr};
    int        max_tail_len{0};
};

class DynamicDecodeLayer {
public:
    DynamicDecodeLayer(DataType              data_type,
                       int                   max_batch_size,
                       int                   vocab_size,
                       int                   vocab_size_padded,
                       cudaStream_t          stream,
                       const cudaDeviceProp* device_prop);

    ~DynamicDecodeLayer();

    void Setup(const std::vector<const Request*>& rs, const TensorMap& args);

    void Forward(TensorMap& args);

    void ForwardMultiStep(TensorMap& args, const ForcedTailContext* forced_ctx);

private:
    cudaStream_t                                          stream_{};
    std::vector<std::unique_ptr<BaseDynamicDecodeLayer>> layers_;

    // Requests for the current batch, captured in Setup so tail
    // handling can consult per-slot GenerationConfig (eos_ids, etc.).
    std::vector<const Request*> requests_;

    // Stop-words tensor for tail stop-criteria, mirroring the layout
    // used by StopCriteriaLayer: [batch, 2, kMaxStopBadWordsLen].
    Buffer_<int> stop_words_;
    Buffer_<int> stop_words_buf_;
    Tensor_<int> stop_words_ten_;
};

}  // namespace turbomind
