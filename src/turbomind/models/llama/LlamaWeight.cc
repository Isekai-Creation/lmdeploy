/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

// Modified from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/models/multi_gpu_gpt/ParallelGptWeight.cc

#include <cuda_runtime.h>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaWeight.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace {

using turbomind::Tensor;

size_t tensor_expected_bytes(const Tensor& tensor)
{
    if (!tensor) {
        return 0;
    }
    const ssize_t rows    = tensor.shape(0);
    const ssize_t cols    = tensor.shape(1);
    const ssize_t numel   = rows * cols;
    const auto    dtype   = tensor.dtype();
    const size_t  bytes   = numel > 0 ? static_cast<size_t>(turbomind::byte_size(dtype, numel)) : 0;
    return bytes;
}

void log_tensor_footprint(const char* tag, const Tensor& tensor)
{
    if (!tensor) {
        TM_LOG_WARNING("[LlamaWeight][%s] tensor empty", tag);
        return;
    }
    const ssize_t rows          = tensor.shape(0);
    const ssize_t cols          = tensor.shape(1);
    const size_t  expected      = tensor_expected_bytes(tensor);
    const size_t  actual        = static_cast<size_t>(tensor.byte_size());
    const auto    dtype_code    = static_cast<int>(tensor.dtype());
    const int     device_type   = static_cast<int>(tensor.device().type);
    const void*   ptr           = tensor.raw_data();
    TM_LOG_WARNING("[LlamaWeight][%s] rows=%zd cols=%zd dtype=%d expected_bytes=%zu actual_bytes=%zu device=%d ptr=%p",
                   tag,
                   rows,
                   cols,
                   dtype_code,
                   expected,
                   actual,
                   device_type,
                   ptr);
    if (expected > 0 && actual < expected) {
        TM_LOG_ERROR(
            "[LlamaWeight][%s] tensor under-allocated: expected_bytes=%zu actual_bytes=%zu rows=%zd cols=%zd dtype=%d",
            tag,
            expected,
            actual,
            rows,
            cols,
            dtype_code);
    }
}

void check_tensor_capacity(const char* tag, const Tensor& tensor)
{
    if (!tensor) {
        TM_LOG_ERROR("[LlamaWeight][%s] tensor is empty", tag);
        TM_CHECK(tensor);
        return;
    }
    const size_t expected = tensor_expected_bytes(tensor);
    const size_t actual   = static_cast<size_t>(tensor.byte_size());
    if (expected > 0) {
        TM_CHECK(actual >= expected)
            << "[LlamaWeight][" << tag << "] tensor under-allocated: expected_bytes=" << expected
            << " actual_bytes=" << actual;
    }
}

}  // namespace

namespace turbomind {

LlamaWeight::LlamaWeight(DataType           data_type,
                         const ModelParam&  model,
                         const EngineParam& engine_param,
                         const LoraParam&   lora_param,
                         const MoeParam&    moe_param):
    model_param_{model},
    engine_param_{engine_param},
    lora_param_{lora_param},
    moe_param_{moe_param},
    hidden_units_(model.hidden_units),
    inter_size_(model.inter_size),
    vocab_size_(model.vocab_size),
    vocab_size_padded_(model.vocab_size),
    embedding_size_(model.embedding_size),
    num_layer_(model.layer_num),
    data_type_{data_type},
    weight_type_{model.weight_type},
    tp_size_(engine_param.attn_tp_size * engine_param.attn_cp_size),
    tp_rank_(engine_param.attn_tp_rank * engine_param.attn_cp_size + engine_param.attn_cp_rank)
{
    if (vocab_size_padded_ % tp_size_ != 0) {
        vocab_size_padded_ = (vocab_size_ + tp_size_ - 1) / tp_size_ * tp_size_;
        TM_LOG_WARNING("pad vocab size from %d to %d", vocab_size_, vocab_size_padded_);
    }
    const int configured_embedding_size = embedding_size_ == 0 ? vocab_size_ : embedding_size_;
    if (configured_embedding_size != vocab_size_) {
        TM_LOG_WARNING(
            "[LlamaWeight] embedding_size (%d) differs from vocab_size (%d); using padded vocab for embeddings",
            configured_embedding_size,
            vocab_size_);
    }
    embedding_size_ = vocab_size_padded_;
    if (embedding_size_ % tp_size_ != 0) {
        embedding_size_ = (embedding_size_ + tp_size_ - 1) / tp_size_ * tp_size_;
        TM_LOG_WARNING("pad embed size from %d to %d", embedding_size_, embedding_size_);
    }
    FT_CHECK(hidden_units_ % tp_size_ == 0);
    TM_CHECK_EQ(vocab_size_padded_ % tp_size_, 0);
    TM_CHECK_EQ(hidden_units_ % tp_size_, 0);

    stream_ = core::Stream::create();
    alloca_ = core::Allocator{stream_, false};

    initialize();
}

LlamaWeight::~LlamaWeight()
{
    release();
}

bool LlamaWeight::is_initialized() const
{
    return initialized_;
}

void LlamaWeight::initialize()
{
    core::ContextGuard guard = context();

    TM_LOG_WARNING(
        "[LlamaWeight] init: vocab_size=%d vocab_size_padded=%d embedding_size=%d hidden_units=%d tp_size=%d",
        vocab_size_,
        vocab_size_padded_,
        embedding_size_,
        hidden_units_,
        tp_size_);

    const int    embedding_rows = vocab_size_padded_;
    const int    embedding_cols = hidden_units_ / tp_size_;
    const double bytes_per_val  = static_cast<double>(byte_size(data_type_, 256)) / 256.0;

    pre_decoder_embedding.emplace(embedding_rows, embedding_cols, data_type_, false, data_type_, 1);
    post_decoder_embedding.emplace(hidden_units_, vocab_size_padded_ / tp_size_, data_type_, false, data_type_, 1);

    TM_LOG_WARNING("[LlamaWeight] pre_decoder_embedding: rows=%d cols=%d dtype=%d",
                   embedding_rows,
                   embedding_cols,
                   static_cast<int>(data_type_));
    TM_LOG_WARNING("[LlamaWeight] post_decoder_embedding: rows=%d cols=%d dtype=%d",
                   hidden_units_,
                   vocab_size_padded_ / tp_size_,
                   static_cast<int>(data_type_));

    const auto expected_emb_elems = static_cast<size_t>(embedding_rows) * static_cast<size_t>(embedding_cols);
    const auto expected_emb_bytes = static_cast<size_t>(byte_size(data_type_, expected_emb_elems));
    auto       log_pre_embedding  = [&](const char* tag, const Tensor& tensor) {
        if (!tensor) {
            TM_LOG_ERROR("[LlamaWeight][%s] embedding tensor empty", tag);
            TM_CHECK(tensor);
            return;
        }
        const size_t actual_emb_bytes = static_cast<size_t>(tensor.byte_size());
        TM_LOG_WARNING(
            "[LlamaWeight][%s] rows=%zd cols=%zd dtype=%d expected_bytes=%zu actual_bytes=%zu bytes_per_val=%.4f",
            tag,
            tensor.shape(0),
            tensor.shape(1),
            static_cast<int>(tensor.dtype()),
            expected_emb_bytes,
            actual_emb_bytes,
            bytes_per_val);
        TM_CHECK_EQ(tensor.shape(0), embedding_rows);
        TM_CHECK_EQ(tensor.shape(1), embedding_cols);
        TM_CHECK(actual_emb_bytes >= expected_emb_bytes)
            << "[LlamaWeight][" << tag << "] embedding under-allocated: expected_bytes=" << expected_emb_bytes
            << " actual_bytes=" << actual_emb_bytes;
    };

    log_pre_embedding("pre_decoder_embedding_device_alloc", pre_decoder_embedding.weight);
    log_tensor_footprint("post_decoder_embedding_device_alloc", post_decoder_embedding.weight);

    register_module("tok_embeddings", pre_decoder_embedding, tp_rank_);
    register_module("output", post_decoder_embedding, tp_rank_);

    /// Lower VRAM pressure on consumer grade GPUs
    /// TODO: Support token embeds on pinned host memory
    pre_decoder_embedding.weight  = empty_like(pre_decoder_embedding.weight, kCPU);
    post_decoder_embedding.weight = empty_like(post_decoder_embedding.weight, kCPU);

    log_pre_embedding("pre_decoder_embedding_host_alloc", pre_decoder_embedding.weight);
    log_tensor_footprint("post_decoder_embedding_host_alloc", post_decoder_embedding.weight);

    decoder_layer_weights.reserve(num_layer_);
    for (int i = 0; i < num_layer_; ++i) {
        decoder_layer_weights.emplace_back(
            new LlamaDecoderLayerWeight(data_type_, i, model_param_, engine_param_, lora_param_, moe_param_));
        register_module("layers", *decoder_layer_weights.back(), i);
    }

    output_norm_weight = Tensor{{hidden_units_}, data_type_, kDEVICE};
    register_parameter("norm.weight", output_norm_weight);
    initialized_ = true;
}

void LlamaWeight::release()
{
    core::ContextGuard guard = context();

    pre_decoder_embedding  = {};
    post_decoder_embedding = {};
    output_norm_weight     = {};

    for (auto& p : decoder_layer_weights) {
        delete p;
    }

    decoder_layer_weights.clear();
    pinned_weights_.clear();

    // Wait for deallocations
    core::Context::stream().Sync();

    // release memory back to os
    core::Context::device_alloc()->trim(0);
    initialized_ = false;
}

void LlamaWeight::to_device(const core::Device& device)
{
    TM_CHECK(device.type == kCPU || device.type == kDEVICE);
    core::ContextGuard guard{stream_, alloca_, Allocator{kCPUpinned}};

    auto tensor_ptr_map = get_parameters();
    for (auto& [name, tensor_ptr] : tensor_ptr_map) {
        if (device.type == kCPU) {
            if (pinned_weights_.find(name) == pinned_weights_.end()) {
                pinned_weights_[name] = empty_like(*tensor_ptr, kCPUpinned);
                Copy(*tensor_ptr, pinned_weights_[name]);
            }
            *tensor_ptr = {};
        }
        else {
            TM_CHECK(pinned_weights_.find(name) != pinned_weights_.end());
            *tensor_ptr = empty_like(pinned_weights_[name], kDEVICE);
            Copy(pinned_weights_[name], *tensor_ptr);
        }
    }
    core::Context::stream().Sync();
    if (device.type == kCPU) {
        core::Context::device_alloc()->trim(0);
    }
}

core::ContextGuard LlamaWeight::context() const
{
    return core::ContextGuard{stream_, alloca_};
}

void LlamaWeight::prepare(const cudaDeviceProp& prop)
{
    core::ContextGuard guard = context();

    const int    embedding_rows   = vocab_size_padded_;
    const int    embedding_cols   = hidden_units_ / tp_size_;
    const auto   expected_elems   = static_cast<size_t>(embedding_rows) * static_cast<size_t>(embedding_cols);
    const auto   expected_bytes   = static_cast<size_t>(byte_size(data_type_, expected_elems));
    const double bytes_per_val    = static_cast<double>(byte_size(data_type_, 256)) / 256.0;
    auto         log_pre_embedding = [&](const char* tag, const Tensor& tensor) {
        if (!tensor) {
            TM_LOG_ERROR("[LlamaWeight][%s] embedding tensor empty", tag);
            TM_CHECK(tensor);
            return;
        }
        const size_t actual_bytes = static_cast<size_t>(tensor.byte_size());
        TM_LOG_WARNING(
            "[LlamaWeight][%s] rows=%zd cols=%zd dtype=%d expected_bytes=%zu actual_bytes=%zu bytes_per_val=%.4f",
            tag,
            tensor.shape(0),
            tensor.shape(1),
            static_cast<int>(tensor.dtype()),
            expected_bytes,
            actual_bytes,
            bytes_per_val);
        TM_CHECK_EQ(tensor.shape(0), embedding_rows);
        TM_CHECK_EQ(tensor.shape(1), embedding_cols);
        TM_CHECK(actual_bytes >= expected_bytes)
            << "[LlamaWeight][" << tag << "] embedding under-allocated: expected_bytes=" << expected_bytes
            << " actual_bytes=" << actual_bytes;
    };

    // Wait for the weights to be filled externally
    check_cuda_error(cudaDeviceSynchronize());

    auto stream = core::Context::stream().handle();

    for (auto& layer : decoder_layer_weights) {
        layer->prepare(prop, stream);
    }

    auto to_device = [](Tensor& x) {
        auto tmp = std::exchange(x, empty_like(x, kDEVICE));
        Copy(tmp, x);
        return tmp;
    };

    // Keep the host tensor until stream synchronization
    auto tmp_token_embeds = to_device(pre_decoder_embedding.weight);
    auto tmp_lm_head      = to_device(post_decoder_embedding.weight);

    log_pre_embedding("pre_decoder_embedding_device_upload", pre_decoder_embedding.weight);
    log_tensor_footprint("post_decoder_embedding_device_upload", post_decoder_embedding.weight);
    log_pre_embedding("pre_decoder_embedding_host_staging", tmp_token_embeds);
    log_tensor_footprint("post_decoder_embedding_host_staging", tmp_lm_head);
    check_tensor_capacity("post_decoder_embedding_device", post_decoder_embedding.weight);

    post_decoder_embedding.prepare();

    const auto emb_bytes     = static_cast<size_t>(pre_decoder_embedding.weight.byte_size());
    const double bytes_per_row =
        vocab_size_padded_ > 0 ? static_cast<double>(emb_bytes) / static_cast<double>(vocab_size_padded_) : 0.0;
    TM_LOG_WARNING("[EmbeddingSanity] vocab=%d hidden=%d dtype=%d alloc_bytes=%zu bytes_per_row=%.2f",
                   vocab_size_padded_,
                   hidden_units_,
                   static_cast<int>(data_type_),
                   emb_bytes,
                   bytes_per_row);

    // Block until processing is done
    check_cuda_error(cudaStreamSynchronize(stream));
}

}  // namespace turbomind
