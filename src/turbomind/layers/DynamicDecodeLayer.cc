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

#include "src/turbomind/layers/DynamicDecodeLayer.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/layers/BaseDynamicDecodeLayer.h"
#include "src/turbomind/layers/sampling_layers/GuidedDecodeMaskLayer.h"
#include "src/turbomind/layers/sampling_layers/GuidedDecodeUpdateLayer.h"
#include "src/turbomind/layers/sampling_layers/LogitsProcessorLayer.h"
#include "src/turbomind/layers/sampling_layers/SamplingLayer.h"
#include "src/turbomind/layers/sampling_layers/StopCriteriaLayer.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

DynamicDecodeLayer::DynamicDecodeLayer(DataType              dtype,
                                       int                   max_batch_size,
                                       int                   vocab_size,
                                       int                   vocab_size_padded,
                                       cudaStream_t          stream,
                                       const cudaDeviceProp* device_prop)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    TM_CHECK(dtype == kFloat32);
    stream_ = stream;
    BaseDynamicDecodeLayer::BaseParam param{max_batch_size, vocab_size, vocab_size_padded, stream, device_prop};
    layers_.emplace_back(new LogitsProcessorLayer<float>{param});
    layers_.emplace_back(new GuidedDecodeMaskLayer<float>{param});
    layers_.emplace_back(new SamplingLayer<float>{param});
    layers_.emplace_back(new GuidedDecodeUpdateLayer<float>{param});
    layers_.emplace_back(new StopCriteriaLayer<float>{param});
}

DynamicDecodeLayer::~DynamicDecodeLayer() {}

void DynamicDecodeLayer::Setup(const std::vector<const Request*>& rs, const TensorMap& args)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    for (const auto& layer : layers_) {
        layer->Setup(rs, args);
    }
}

void DynamicDecodeLayer::Forward(TensorMap& args)
{
    /**
     * @brief
     * input_tensors:
     *   \param  logits [batch_size, beam_width, vocab_size_padded]
     *   \param  step [1] on cpu
     *   \param  max_input_length [1] on cpu
     *   \param  input_lengths [batch_size, beam_width], optional
     *   \param  sequence_limit_length [batch_size]
     *   \param  ite [1] on cpu
     *   \param  local_batch_size [1] on cpu
     *   \param  stop_words_list [batch_size, 2, stop_words_length], optional
     *   \param  runtime_top_k [batch_size] on cpu, optional, uint
     *   \param  runtime_top_p [batch_size] on cpu, optional, float
     *   \param  temperature [batch_size] on cpu, optional, float
     *   \param  repetition_penalty [batch_size] on cpu, optional, float
     *   \param  bad_words_list [batch_size, 2, bad_words_length], optional
     *
     * output_tensors:
     *   \param  output_ids [max_seq_len, batch_size, 1]
     *   \param  curand_state [local_batch_size]
     *   \param  finished [batch_size * beam_width], optional
     *   \param  sequence_length [batch_size * beam_width], optional
     *   \param  sampled_indexes [batch_size, 1, kMaxLogProb], optional
     *   \param  sampled_logprobs [batch_size, 1, kMaxLogProb], optional
     *   \param  sampled_nums [batch_size, 1], optional
     */

    for (const auto& layer : layers_) {
        layer->Forward(args);
    }
}

void DynamicDecodeLayer::ForwardMultiStep(TensorMap& args, const ForcedTailContext* forced_ctx)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    // Always run the existing single-step decode once for the base token.
    Forward(args);

    // If no forced tail is provided, we are done.
    if (!forced_ctx || !forced_ctx->forced_tokens || !forced_ctx->forced_lengths || forced_ctx->max_tail_len <= 0) {
        TM_LOG_DEBUG("%s stop (no tail)", __PRETTY_FUNCTION__);
        return;
    }

    Tensor logits = args.at("logits");
    const int batch_size = logits.shape(0);

    Tensor output_ids      = args.at("output_ids");
    Tensor sequence_length = args.at("sequence_length");
    Tensor finished        = args.at("finished");

    std::vector<int> h_seq_limit;
    if (Tensor* seq_lim = args.try_("sequence_limit_length")) {
        h_seq_limit.resize(batch_size);
        check_cuda_error(cudaMemcpyAsync(
            h_seq_limit.data(), seq_lim->data<int>(), batch_size * sizeof(int), cudaMemcpyDeviceToHost, stream_));
        check_cuda_error(cudaStreamSynchronize(stream_));
    }

    if (output_ids.ndim() < 2) {
        TM_LOG_WARNING(
            "%s: output_ids tensor has invalid ndim=%d; skipping tail tokens", __PRETTY_FUNCTION__, output_ids.ndim());
        TM_LOG_DEBUG("%s stop (invalid output_ids)", __PRETTY_FUNCTION__);
        return;
    }

    const int max_seq_len = output_ids.shape(0);
    const int step        = *args.at("step").data<int>();

    const int* forced_tokens  = forced_ctx->forced_tokens;
    const int* forced_lengths = forced_ctx->forced_lengths;
    const int  max_tail_len   = forced_ctx->max_tail_len;
    int*       committed      = forced_ctx->committed_lengths;

    // Host copies of sequence lengths and finished flags for per-slot updates.
    std::vector<int>  h_seq_len(batch_size);
    std::vector<bool> h_finished(batch_size);

    check_cuda_error(cudaMemcpyAsync(
        h_seq_len.data(), sequence_length.data<int>(), batch_size * sizeof(int), cudaMemcpyDeviceToHost, stream_));
    check_cuda_error(cudaMemcpyAsync(
        h_finished.data(), finished.data<bool>(), batch_size * sizeof(bool), cudaMemcpyDeviceToHost, stream_));
    check_cuda_error(cudaStreamSynchronize(stream_));

    int* d_output_ids = output_ids.data<int>();

    for (int i = 0; i < batch_size; ++i) {
        if (h_finished[i]) {
            continue;
        }

        int committed_i = 0;

        int len = forced_lengths[i];
        if (len <= 0) {
            continue;
        }
        if (len > max_tail_len) {
            len = max_tail_len;
        }

        const int token_offset = i * max_tail_len;

        for (int t = 0; t < len; ++t) {
            const int pos = step + 1 + t;
            if (pos < 0 || pos >= max_seq_len) {
                break;
            }

            if (!h_seq_limit.empty()) {
                const int limit = h_seq_limit[i];
                if (limit > 0 && pos >= limit) {
                    break;
                }
            }

            const int value = forced_tokens[token_offset + t];
            int*      dst   = d_output_ids + pos * batch_size + i;
            check_cuda_error(cudaMemcpyAsync(dst, &value, sizeof(int), cudaMemcpyHostToDevice, stream_));

            // Increment host-side sequence length view so subsequent tokens
            // for this slot are appended after the newly committed ones.
            ++h_seq_len[i];
            ++committed_i;
        }

        if (committed) {
            committed[i] = committed_i;
        }
    }

    check_cuda_error(cudaStreamSynchronize(stream_));

    // Write back updated sequence lengths for all batch slots.
    check_cuda_error(cudaMemcpyAsync(
        sequence_length.data<int>(), h_seq_len.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice, stream_));
    check_cuda_error(cudaStreamSynchronize(stream_));

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

}  // namespace turbomind
