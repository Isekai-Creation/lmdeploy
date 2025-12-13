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
#include "src/turbomind/kernels/stop_criteria_kernels.h"
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

    // Core decode stack (logits processing, guided decode, sampling, stop).
    layers_.emplace_back(new LogitsProcessorLayer<float>{param});
    layers_.emplace_back(new GuidedDecodeMaskLayer<float>{param});
    layers_.emplace_back(new SamplingLayer<float>{param});
    layers_.emplace_back(new GuidedDecodeUpdateLayer<float>{param});
    layers_.emplace_back(new StopCriteriaLayer<float>{param});

    // Stop-words buffers for tail stop-criteria. Layout matches
    // StopCriteriaLayer: [batch, 2, kMaxStopBadWordsLen].
    stop_words_     = {max_batch_size * 2 * kMaxStopBadWordsLen, kCPUpinned};
    stop_words_buf_ = {max_batch_size * 2 * kMaxStopBadWordsLen, kDEVICE};
}

DynamicDecodeLayer::~DynamicDecodeLayer() {}

void DynamicDecodeLayer::Setup(const std::vector<const Request*>& rs, const TensorMap& args)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    requests_ = rs;
    for (const auto& layer : layers_) {
        layer->Setup(rs, args);
    }

     // Initialize stop-words tensor for tail steps from GenerationConfig::stop_ids.
    stop_words_ten_ = {};
    init_stop_bad_words(&GenerationConfig::stop_ids,  //
                        "stop_words",
                        rs,
                        stop_words_.data(),
                        stop_words_buf_.data(),
                        stop_words_ten_);
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

    Tensor logits      = args.at("logits");
    const int batch_size = logits.shape(0);

    Tensor output_ids      = args.at("output_ids");
    Tensor sequence_length = args.at("sequence_length");
    Tensor finished        = args.at("finished");

    // Optional per-slot sequence limits, used only in the host tail path.
    // When GPU tail is enabled and successfully taken, we avoid building
    // this host-side buffer and rely on the device limit vector instead.
    std::vector<int> h_seq_limit;

    int max_seq_len = 0;
    if (output_ids.ndim() >= 2) {
        // Normal case: output_ids is [max_seq_len, batch_size, ...]
        max_seq_len = output_ids.shape(0);
    }
    else {
        // Fallback: output_ids is a flat buffer. Infer [max_seq_len, batch_size]
        // from the total size and batch_size, matching the indexing pattern
        //   idx = pos * batch_size + i
        const int total_elems = static_cast<int>(output_ids.size());
        if (batch_size <= 0 || total_elems <= 0 || (total_elems % batch_size) != 0) {
            TM_LOG_WARNING(
                "%s: output_ids tensor has ndim=%d and size=%d; cannot infer "
                "[max_seq_len, batch_size] layout; skipping tail tokens",
                __PRETTY_FUNCTION__,
                output_ids.ndim(),
                total_elems);
            TM_LOG_DEBUG("%s stop (invalid flat output_ids)", __PRETTY_FUNCTION__);
            return;
        }
        max_seq_len = total_elems / batch_size;
        TM_LOG_WARNING(
            "%s: output_ids tensor has ndim=%d; inferring max_seq_len=%d from "
            "flat buffer of size=%d and batch_size=%d",
            __PRETTY_FUNCTION__,
            output_ids.ndim(),
            max_seq_len,
            total_elems,
            batch_size);
    }

    const int step = *args.at("step").data<int>();

    const int* forced_tokens  = forced_ctx->forced_tokens;
    const int* forced_lengths = forced_ctx->forced_lengths;
    const int  max_tail_len   = forced_ctx->max_tail_len;
    int*       committed      = forced_ctx->committed_lengths;

    // Optional stop-words configuration (may be empty).
    const bool have_stop_words = static_cast<bool>(stop_words_ten_);
    const int  stop_words_len  = have_stop_words ? stop_words_ten_.shape(2) : 0;
    const int* h_stop_words    = have_stop_words ? stop_words_.data() : nullptr;

    const bool gpu_tail_enabled = std::getenv("TM_ENABLE_GPU_TAIL") && !have_stop_words && max_tail_len > 0;

    // Optional GPU tail path: enabled via TM_ENABLE_GPU_TAIL and only when
    // stop-words are not configured for this batch. When we take this path
    // we avoid building any host-side sequence-limit or finished buffers.
    if (gpu_tail_enabled) {
        constexpr int kMaxEosPerSlot = 4;

        std::vector<int> h_eos_ids(static_cast<size_t>(batch_size) * kMaxEosPerSlot, 0);
        std::vector<int> h_eos_counts(batch_size, 0);
        bool             eos_ok = true;

        for (int i = 0; i < batch_size; ++i) {
            int count = 0;
            if (i < static_cast<int>(requests_.size()) && requests_[i]) {
                const auto& eos_vec = requests_[i]->gen_cfg.eos_ids;
                if (static_cast<int>(eos_vec.size()) > kMaxEosPerSlot) {
                    eos_ok = false;
                    break;
                }
                count = static_cast<int>(eos_vec.size());
                for (int k = 0; k < count; ++k) {
                    h_eos_ids[static_cast<size_t>(i) * kMaxEosPerSlot + k] = eos_vec[k];
                }
            }
            h_eos_counts[i] = count;
        }

        if (eos_ok) {
            Buffer_<int> d_eos_ids(h_eos_ids.size(), kDEVICE);
            Buffer_<int> d_eos_counts(batch_size, kDEVICE);
            Buffer_<int> d_forced_tokens(static_cast<size_t>(batch_size) * max_tail_len, kDEVICE);
            Buffer_<int> d_forced_lengths(batch_size, kDEVICE);
            Buffer_<int> d_committed(batch_size, kDEVICE);

            // Copy EOS metadata and forced tail context to device.
            check_cuda_error(cudaMemcpyAsync(d_eos_ids.data(),
                                             h_eos_ids.data(),
                                             h_eos_ids.size() * sizeof(int),
                                             cudaMemcpyHostToDevice,
                                             stream_));
            check_cuda_error(cudaMemcpyAsync(d_eos_counts.data(),
                                             h_eos_counts.data(),
                                             h_eos_counts.size() * sizeof(int),
                                             cudaMemcpyHostToDevice,
                                             stream_));
            check_cuda_error(cudaMemcpyAsync(d_forced_tokens.data(),
                                             forced_tokens,
                                             static_cast<size_t>(batch_size) * max_tail_len * sizeof(int),
                                             cudaMemcpyHostToDevice,
                                             stream_));
            check_cuda_error(cudaMemcpyAsync(
                d_forced_lengths.data(), forced_lengths, batch_size * sizeof(int), cudaMemcpyHostToDevice, stream_));

            Tensor* seq_lim     = args.try_("sequence_limit_length");
            int*    d_seq_limit = seq_lim ? seq_lim->data<int>() : nullptr;

            invokeApplyForcedTail(d_forced_tokens.data(),
                                  d_forced_lengths.data(),
                                  output_ids.data<int>(),
                                  sequence_length.data<int>(),
                                  finished.buffer().data<bool>(),
                                  d_seq_limit,
                                  d_eos_ids.data(),
                                  d_eos_counts.data(),
                                  kMaxEosPerSlot,
                                  d_committed.data(),
                                  batch_size,
                                  max_tail_len,
                                  max_seq_len,
                                  step,
                                  stream_);

            if (committed) {
                std::vector<int> committed_host(batch_size, 0);
                check_cuda_error(cudaMemcpyAsync(committed_host.data(),
                                                 d_committed.data(),
                                                 batch_size * sizeof(int),
                                                 cudaMemcpyDeviceToHost,
                                                 stream_));
                check_cuda_error(cudaStreamSynchronize(stream_));
                std::copy(committed_host.begin(), committed_host.end(), committed);
            }

            TM_LOG_DEBUG("%s stop (GPU tail)", __PRETTY_FUNCTION__);
            return;
        }
    }

    // Build host-side sequence limits only when we are not taking the GPU
    // tail path. The host tail logic uses this to clamp per-slot lengths.
    if (Tensor* seq_lim_host = args.try_("sequence_limit_length")) {
        h_seq_limit.resize(batch_size);
        check_cuda_error(cudaMemcpyAsync(
            h_seq_limit.data(), seq_lim_host->data<int>(), batch_size * sizeof(int), cudaMemcpyDeviceToHost, stream_));
        check_cuda_error(cudaStreamSynchronize(stream_));
    }

    // Host copies of sequence lengths and finished flags for per-slot updates.
    std::vector<int>     h_seq_len(batch_size);
    std::vector<uint8_t> h_finished_bytes(batch_size);
    std::vector<bool>    h_finished(batch_size);

    cudaError_t err = cudaMemcpyAsync(
        h_seq_len.data(),
        sequence_length.data<int>(),
        batch_size * sizeof(int),
        cudaMemcpyDeviceToHost,
        stream_);
    check_cuda_error(err);

    err = cudaMemcpyAsync(
        h_finished_bytes.data(),
        finished.buffer().raw_data(),
        batch_size * sizeof(bool),
        cudaMemcpyDeviceToHost,
        stream_);
    check_cuda_error(err);

    err = cudaStreamSynchronize(stream_);
    check_cuda_error(err);

    // Decode host finished bytes into bool flags.
    for (int i = 0; i < batch_size; ++i) {
        h_finished[i] = h_finished_bytes[i] != 0;
    }

    int* d_output_ids = output_ids.data<int>();

    // Pre-compute per-slot tail lengths (clamped by max_tail_len) and
    // the global maximum tail depth so we can emulate sequential decode
    // steps: for each t, append one token across all slots and then run
    // stop-criteria on the updated sequences.
    std::vector<int> tail_len(batch_size, 0);
    int              max_len = 0;
    for (int i = 0; i < batch_size; ++i) {
        if (h_finished[i]) {
            tail_len[i] = 0;
            continue;
        }
        int len = forced_lengths[i];
        if (len <= 0) {
            tail_len[i] = 0;
            continue;
        }
        if (len > max_tail_len) {
            len = max_tail_len;
        }
        tail_len[i] = len;
        max_len     = std::max(max_len, len);
    }

    // Per-slot committed tail counts.
    std::vector<int> committed_host(batch_size, 0);

    // Temporary buffer for stop-words window when checking tails.
    int window_buf[kMaxStopBadWordsLen];

    for (int t = 0; t < max_len; ++t) {
        const int pos = step + 1 + t;
        if (pos < 0 || pos >= max_seq_len) {
            break;
        }

        // Append one tail token (at depth t) for all eligible slots.
        for (int i = 0; i < batch_size; ++i) {
            if (h_finished[i]) {
                continue;
            }
            if (t >= tail_len[i]) {
                continue;
            }

            if (!h_seq_limit.empty()) {
                const int limit = h_seq_limit[i];
                if (limit > 0 && pos >= limit) {
                    h_finished[i] = true;
                    continue;
                }
            }

            const int value = forced_tokens[i * max_tail_len + t];
            int*      dst   = d_output_ids + pos * batch_size + i;
            check_cuda_error(cudaMemcpyAsync(dst, &value, sizeof(int), cudaMemcpyHostToDevice, stream_));

            ++h_seq_len[i];
            ++committed_host[i];

            // If this commit consumes the remaining per-request length
            // budget, mark the slot finished and skip further tails.
            if (!h_seq_limit.empty()) {
                const int limit = h_seq_limit[i];
                if (limit > 0 && h_seq_len[i] >= limit) {
                    h_finished[i] = true;
                    continue;
                }
            }

            // EOS stop: if this tail token hits EOS, mark finished and
            // prevent further tails for this slot.
            if (i < static_cast<int>(requests_.size()) && requests_[i]) {
                const auto& gen_cfg = requests_[i]->gen_cfg;
                if (!gen_cfg.eos_ids.empty()) {
                    for (int eos : gen_cfg.eos_ids) {
                        if (value == eos) {
                            h_finished[i] = true;
                            break;
                        }
                    }
                }
            }
        }

        check_cuda_error(cudaStreamSynchronize(stream_));

        // Stop-words criterion: emulate the single-step behaviour for
        // the synthetic step `pos` by scanning the tail suffix for each
        // slot using the same token patterns as StopCriteriaLayer.
        if (have_stop_words && stop_words_len > 0) {
            for (int i = 0; i < batch_size; ++i) {
                if (h_finished[i]) {
                    continue;
                }

                const int* base_tokens  = h_stop_words + i * 2 * stop_words_len;
                const int* base_offsets = base_tokens + stop_words_len;

                for (int id = 0; id < stop_words_len; ++id) {
                    const int item_end = base_offsets[id];
                    if (item_end < 0) {
                        continue;
                    }
                    const int item_start = (id > 0) ? base_offsets[id - 1] : 0;
                    const int item_size  = item_end - item_start;
                    if (item_size <= 0 || item_size > kMaxStopBadWordsLen) {
                        continue;
                    }

                    if (pos + 1 < item_size) {
                        continue;
                    }
                    const int start_time = pos - (item_size - 1);
                    if (start_time < 0) {
                        continue;
                    }

                    // Gather the candidate window from device for this slot.
                    bool window_ok = true;
                    for (int k = 0; k < item_size; ++k) {
                        const int time_idx = start_time + k;
                        const int idx      = time_idx * batch_size + i;
                        check_cuda_error(cudaMemcpyAsync(&window_buf[k],
                                                         d_output_ids + idx,
                                                         sizeof(int),
                                                         cudaMemcpyDeviceToHost,
                                                         stream_));
                    }
                    check_cuda_error(cudaStreamSynchronize(stream_));

                    for (int k = 0; k < item_size; ++k) {
                        if (window_buf[k] != base_tokens[item_start + k]) {
                            window_ok = false;
                            break;
                        }
                    }

                    if (window_ok) {
                        h_finished[i] = true;
                        break;
                    }
                }
            }
        }
    }

    if (committed) {
        std::copy(committed_host.begin(), committed_host.end(), committed);
    }

    // Write back updated sequence lengths for all batch slots.
    check_cuda_error(cudaMemcpyAsync(
        sequence_length.data<int>(), h_seq_len.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice, stream_));
    check_cuda_error(cudaStreamSynchronize(stream_));

    // Propagate updated finished flags (including EOS-triggered tails) back
    // to device so subsequent decode steps see the correct termination state.
    for (int i = 0; i < batch_size; ++i) {
        h_finished_bytes[i] = static_cast<uint8_t>(h_finished[i] ? 1 : 0);
    }
    err = cudaMemcpyAsync(
        finished.buffer().raw_data(),
        h_finished_bytes.data(),
        batch_size * sizeof(bool),
        cudaMemcpyHostToDevice,
        stream_);
    check_cuda_error(err);
    err = cudaStreamSynchronize(stream_);
    check_cuda_error(err);

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

}  // namespace turbomind
