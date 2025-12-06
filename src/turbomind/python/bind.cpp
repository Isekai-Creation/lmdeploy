// Copyright (c) OpenMMLab. All rights reserved.

#include <memory>
#include <sstream>
#include <stdexcept>
#include <cstdint>

#include <cuda_runtime.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <xgrammar/xgrammar.h>

#include "lmdeploy/turbomind/kernels/speculative_decoding/common.h"
#include "lmdeploy/turbomind/kernels/speculative_decoding/tree_accept_kernels.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/engine/model_request.h"
#include "src/turbomind/models/llama/EagleModule.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/python/dlpack.h"
#include "src/turbomind/triton_backend/llama/LlamaTritonModel.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/metrics.h"

namespace py = pybind11;
namespace ft = turbomind;
using namespace pybind11::literals;

using ft::core::Tensor;
namespace eagle_kernels = turbomind::kernels::speculative_decoding;

// prepare to bind container
using TensorMap = ft::core::TensorMap;
PYBIND11_MAKE_OPAQUE(TensorMap);
static const char kDlTensorCapsuleName[] = "dltensor";

DLDevice getDLDevice(const Tensor& tensor)
{
    int device_id = 0;
    if (tensor.device().type == ft::kDEVICE) {
        cudaPointerAttributes ptr_attr{};
        cudaPointerGetAttributes(&ptr_attr, tensor.raw_data());
        device_id = ptr_attr.device;
    }

    DLDevice device{kDLCPU, device_id};

    switch (tensor.device().type) {
        case ft::kCPU:
            device.device_type = DLDeviceType::kDLCPU;
            break;
        case ft::kCPUpinned:
            device.device_type = DLDeviceType::kDLCUDAHost;
            break;
        case ft::kDEVICE:
            device.device_type = DLDeviceType::kDLCUDA;
            break;
        default:
            break;
    }

    return device;
}

DLManagedTensor* TritonTensorToDLManagedTensor(Tensor& tensor)
{
    DLDevice   device = getDLDevice(tensor);
    DLDataType data_type{0, 0, 1};
    using ft::data_type_v;
    switch (tensor.dtype()) {
        case data_type_v<bool>:
            data_type.code = DLDataTypeCode::kDLBool;
            data_type.bits = 8;
            break;
        case data_type_v<uint8_t>:
            data_type.code = DLDataTypeCode::kDLUInt;
            data_type.bits = 8;
            break;
        case data_type_v<uint16_t>:
            data_type.code = DLDataTypeCode::kDLUInt;
            data_type.bits = 16;
            break;
        case data_type_v<uint32_t>:
            data_type.code = DLDataTypeCode::kDLUInt;
            data_type.bits = 32;
            break;
        case data_type_v<uint64_t>:
            data_type.code = DLDataTypeCode::kDLUInt;
            data_type.bits = 64;
            break;
        case data_type_v<int8_t>:
            data_type.code = DLDataTypeCode::kDLInt;
            data_type.bits = 8;
            break;
        case data_type_v<int16_t>:
            data_type.code = DLDataTypeCode::kDLInt;
            data_type.bits = 16;
            break;
        case data_type_v<int32_t>:
            data_type.code = DLDataTypeCode::kDLInt;
            data_type.bits = 32;
            break;
        case data_type_v<int64_t>:
            data_type.code = DLDataTypeCode::kDLInt;
            data_type.bits = 64;
            break;
        case data_type_v<turbomind::half_t>:
            data_type.code = DLDataTypeCode::kDLFloat;
            data_type.bits = 16;
            break;
        case data_type_v<float>:
            data_type.code = DLDataTypeCode::kDLFloat;
            data_type.bits = 32;
            break;
        case data_type_v<double>:
            data_type.code = DLDataTypeCode::kDLFloat;
            data_type.bits = 64;
            break;
        case data_type_v<turbomind::bfloat16_t>:
            data_type.code = DLDataTypeCode::kDLBfloat;
            data_type.bits = 16;
            break;
        default:
            break;
    }

    static_assert(sizeof(int64_t) == sizeof(tensor.shape(0)));

    Tensor*  ctx = new Tensor(tensor);
    DLTensor dl_tensor{const_cast<void*>(ctx->raw_data()),
                       device,
                       (int32_t)(ctx->ndim()),
                       data_type,
                       (int64_t*)ctx->shape().data(),
                       (int64_t*)(nullptr),
                       0};
    return new DLManagedTensor{dl_tensor, ctx, [](DLManagedTensor* dlmt) {  //
                                   delete (Tensor*)dlmt->manager_ctx;
                                   delete dlmt;
                               }};
}

ft::DeviceType getMemoryType(DLDevice device)
{
    switch (device.device_type) {
        case DLDeviceType::kDLCUDAHost:
            return ft::DeviceType::kCPUpinned;
        case DLDeviceType::kDLCUDA:
            return ft::DeviceType::kDEVICE;
        case DLDeviceType::kDLCPU:
        default:
            return ft::DeviceType::kCPU;
    }
}

ft::DataType getDataType(DLDataType data_type)
{
    using ft::data_type_v;
    switch (data_type.code) {
        case DLDataTypeCode::kDLUInt:
            switch (data_type.bits) {
                case 8:
                    return data_type_v<uint8_t>;
                case 16:
                    return data_type_v<uint16_t>;
                case 32:
                    return data_type_v<uint32_t>;
                case 64:
                    return data_type_v<uint64_t>;
                default:
                    return data_type_v<void>;
            }
            break;
        case DLDataTypeCode::kDLInt:
            switch (data_type.bits) {
                case 8:
                    return data_type_v<int8_t>;
                case 16:
                    return data_type_v<int16_t>;
                case 32:
                    return data_type_v<int32_t>;
                case 64:
                    return data_type_v<int64_t>;
                default:
                    return data_type_v<void>;
            }
            break;
        case DLDataTypeCode::kDLFloat:
            switch (data_type.bits) {
                case 16:
                    return data_type_v<turbomind::half_t>;
                case 32:
                    return data_type_v<float>;
                case 64:
                    return data_type_v<double>;
                default:
                    return data_type_v<void>;
            }
            break;
        case DLDataTypeCode::kDLBfloat:
            switch (data_type.bits) {
                case 16:
                    return data_type_v<turbomind::bfloat16_t>;
                default:
                    return data_type_v<void>;
            }
            break;
        case DLDataTypeCode::kDLBool:
            return data_type_v<bool>;
        default:
            return data_type_v<void>;
    }
}

std::shared_ptr<Tensor> DLManagedTensorToTritonTensor(DLManagedTensor* tensor)
{
    auto& dl_tensor = tensor->dl_tensor;
    auto  where     = getMemoryType(dl_tensor.device);
    auto  dtype     = getDataType(dl_tensor.dtype);
    assert(dl_tensor.ndim > 0);
    std::vector<ft::core::ssize_t> shape(dl_tensor.shape, dl_tensor.shape + dl_tensor.ndim);

    std::shared_ptr<void> ptr{dl_tensor.data, [tensor](void* p) {
                                  if (tensor->deleter) {
                                      tensor->deleter(tensor);
                                  }
                              }};
    return std::make_shared<Tensor>(ptr, std::move(shape), dtype, where);
}

static void safe_memcpy(void* dst, const void* src, size_t size)
{
    cudaPointerAttributes dat{};
    cudaPointerAttributes sat{};
    ft::check_cuda_error(cudaPointerGetAttributes(&dat, dst));
    ft::check_cuda_error(cudaPointerGetAttributes(&sat, src));
    try {
        if (dat.devicePointer && sat.devicePointer) {
            // Both can be accessed from current context
            ft::check_cuda_error(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
        }
        else if (dat.type == cudaMemoryTypeDevice && sat.type == cudaMemoryTypeDevice) {
            if (dat.device != sat.device) {
                // On different devices, try peer memcpy
                ft::check_cuda_error(cudaMemcpyPeer(dst, dat.device, src, sat.device, size));
            }
            else {
                // Same device, switch to the device first (this is unlikely)
                ft::CudaDeviceGuard guard(dat.device);
                ft::check_cuda_error(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
            }
        }
        else {
            // Unknown case, give it a try anyway
            ft::check_cuda_error(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
        }
    }
    catch (...) {
        int device_id{-1};
        cudaGetDevice(&device_id);
        TM_LOG_ERROR("cudaMemcpy failed: dst=(%d, %d, %p, %p), src=(%d, %d, %p, %p), size=%s, device=%d",
                     (int)dat.type,
                     dat.device,
                     dat.devicePointer,
                     dat.hostPointer,
                     (int)sat.type,
                     sat.device,
                     sat.devicePointer,
                     sat.hostPointer,
                     std::to_string(size).c_str(),
                     device_id);
        throw;
    }
}

namespace {

struct ScopedGIL {
    ScopedGIL(const ScopedGIL&) = delete;
    ScopedGIL& operator=(const ScopedGIL&) = delete;
    ScopedGIL(ScopedGIL&&)                 = delete;
    ScopedGIL& operator=(ScopedGIL&&) = delete;
    ScopedGIL()
    {
        state = PyGILState_Ensure();
    }
    ~ScopedGIL()
    {
        PyGILState_Release(state);
    }
    PyGILState_STATE state;
};

}  // namespace

PYBIND11_MODULE(_turbomind, m)
{
    // Lightweight bindings for EAGLE speculative decoding kernels used in tests.
    //
    // These helpers accept torch-like tensor objects (anything exposing
    // .shape, .device, and .data_ptr()) and launch the underlying CUDA
    // kernels on the current device.
    // EAGLE A21: prototype acceptance/pack bindings – see EAGLE_TODO.md (🧪, GPU/CI validation pending).
    m.def(
        "eagle_accept_draft_tokens",
        [](py::object                   output_ids,
           py::object                   draft_ids,
           py::object                   target_ids,
           py::object                   accepted_lengths,
           py::object                   sequence_lengths,
           py::object                   paths,
           py::object                   best_path_ids,
           py::object                   batch_slots_opt) {
            // Infer basic shapes from tensors.
            auto draft_shape  = draft_ids.attr("shape").cast<py::tuple>();
            auto output_shape = output_ids.attr("shape").cast<py::tuple>();
            auto paths_shape  = paths.attr("shape").cast<py::tuple>();

            if (draft_shape.size() != 2 || output_shape.size() != 2 || paths_shape.size() != 3) {
                throw std::invalid_argument("eagle_accept_draft_tokens: unexpected tensor ranks");
            }

            const auto batch_size       = draft_shape[0].cast<int64_t>();
            const auto max_draft_tokens = draft_shape[1].cast<int64_t>();
            const auto max_batch_size   = output_shape[0].cast<int64_t>();
            const auto max_seq_len      = output_shape[1].cast<int64_t>();
            const auto max_path_len     = paths_shape[2].cast<int64_t>();

            // Resolve device from output_ids tensor.
            auto device_obj = output_ids.attr("device");
            auto device_type
                = device_obj.attr("type").cast<std::string>();  // e.g. "cuda" or "cpu"

            if (device_type != "cuda") {
                throw std::runtime_error("eagle_accept_draft_tokens expects CUDA tensors");
            }

            int device_index = 0;
            if (!device_obj.attr("index").is_none()) {
                device_index = device_obj.attr("index").cast<int>();
            }
            ft::CudaDeviceGuard device_guard(device_index);

            auto get_int32_ptr = [](py::object const& tensor) -> eagle_kernels::SizeType* {
                auto ptr_obj = tensor.attr("data_ptr")();
                auto ptr_val = ptr_obj.cast<uintptr_t>();
                return reinterpret_cast<eagle_kernels::SizeType*>(ptr_val);
            };

            auto get_token_ptr = [](py::object const& tensor) -> eagle_kernels::TokenIdType* {
                auto ptr_obj = tensor.attr("data_ptr")();
                auto ptr_val = ptr_obj.cast<uintptr_t>();
                return reinterpret_cast<eagle_kernels::TokenIdType*>(ptr_val);
            };

            eagle_kernels::TokenIdType* output_ids_ptr       = get_token_ptr(output_ids);
            eagle_kernels::TokenIdType* draft_ids_ptr        = get_token_ptr(draft_ids);
            eagle_kernels::TokenIdType* target_ids_ptr       = get_token_ptr(target_ids);
            eagle_kernels::SizeType*    accepted_lengths_ptr = get_int32_ptr(accepted_lengths);
            eagle_kernels::SizeType*    sequence_lengths_ptr = get_int32_ptr(sequence_lengths);
            eagle_kernels::SizeType*    paths_ptr            = get_int32_ptr(paths);
            eagle_kernels::SizeType*    best_path_ids_ptr    = get_int32_ptr(best_path_ids);

            eagle_kernels::SizeType* batch_slots_ptr = nullptr;
            if (!batch_slots_opt.is_none()) {
                batch_slots_ptr = get_int32_ptr(batch_slots_opt);
            }

            cudaStream_t stream{};
            ft::check_cuda_error(cudaStreamCreate(&stream));

            eagle_kernels::launchAcceptDraftTokensKernel(
                output_ids_ptr,
                draft_ids_ptr,
                target_ids_ptr,
                accepted_lengths_ptr,
                sequence_lengths_ptr,
                paths_ptr,
                best_path_ids_ptr,
                batch_slots_ptr,
                static_cast<eagle_kernels::SizeType>(batch_size),
                static_cast<eagle_kernels::SizeType>(max_batch_size),
                static_cast<eagle_kernels::SizeType>(max_seq_len),
                static_cast<eagle_kernels::SizeType>(max_draft_tokens),
                static_cast<eagle_kernels::SizeType>(max_path_len),
                stream);

            ft::check_cuda_error(cudaStreamSynchronize(stream));
            ft::check_cuda_error(cudaStreamDestroy(stream));
        },
        "output_ids"_a,
        "draft_ids"_a,
        "target_ids"_a,
        "accepted_lengths"_a,
        "sequence_lengths"_a,
        "paths"_a,
        "best_path_ids"_a,
        "batch_slots"_a = py::none());

    // EAGLE A21/A22: prototype path-pack bindings – see EAGLE_TODO.md (🧪, GPU/CI validation pending).
    m.def(
        "eagle_pack_accepted_paths",
        [](py::object accepted_lengths_cumsum,
           py::object paths_offsets,
           py::object accepted_lengths,
           py::object best_path_ids,
           py::object paths,
           py::object batch_slots_opt) {
            auto lengths_shape = accepted_lengths.attr("shape").cast<py::tuple>();
            auto paths_shape   = paths.attr("shape").cast<py::tuple>();

            if (lengths_shape.size() != 1 || paths_shape.size() != 3) {
                throw std::invalid_argument("eagle_pack_accepted_paths: unexpected tensor ranks");
            }

            const auto batch_size     = lengths_shape[0].cast<int64_t>();
            const auto max_batch_size = paths_shape[0].cast<int64_t>();
            const auto num_paths      = paths_shape[1].cast<int64_t>();
            const auto max_path_len   = paths_shape[2].cast<int64_t>();

            auto device_obj = paths.attr("device");
            auto device_type
                = device_obj.attr("type").cast<std::string>();  // e.g. "cuda" or "cpu"

            if (device_type != "cuda") {
                throw std::runtime_error("eagle_pack_accepted_paths expects CUDA tensors");
            }

            int device_index = 0;
            if (!device_obj.attr("index").is_none()) {
                device_index = device_obj.attr("index").cast<int>();
            }
            ft::CudaDeviceGuard device_guard(device_index);

            auto get_int32_ptr = [](py::object const& tensor) -> eagle_kernels::SizeType* {
                auto ptr_obj = tensor.attr("data_ptr")();
                auto ptr_val = ptr_obj.cast<uintptr_t>();
                return reinterpret_cast<eagle_kernels::SizeType*>(ptr_val);
            };

            eagle_kernels::SizeType* accepted_lengths_cumsum_ptr = get_int32_ptr(accepted_lengths_cumsum);
            eagle_kernels::SizeType* paths_offsets_ptr           = get_int32_ptr(paths_offsets);
            eagle_kernels::SizeType* accepted_lengths_ptr        = get_int32_ptr(accepted_lengths);
            eagle_kernels::SizeType* best_path_ids_ptr           = get_int32_ptr(best_path_ids);
            eagle_kernels::SizeType* paths_ptr                   = get_int32_ptr(paths);

            eagle_kernels::SizeType* batch_slots_ptr = nullptr;
            if (!batch_slots_opt.is_none()) {
                batch_slots_ptr = get_int32_ptr(batch_slots_opt);
            }

            cudaStream_t stream{};
            ft::check_cuda_error(cudaStreamCreate(&stream));

            eagle_kernels::launchPackAcceptedPathsKernel(
                accepted_lengths_cumsum_ptr,
                paths_offsets_ptr,
                accepted_lengths_ptr,
                best_path_ids_ptr,
                paths_ptr,
                batch_slots_ptr,
                static_cast<eagle_kernels::SizeType>(batch_size),
                static_cast<eagle_kernels::SizeType>(max_batch_size),
                static_cast<eagle_kernels::SizeType>(num_paths),
                static_cast<eagle_kernels::SizeType>(max_path_len),
                stream);

            ft::check_cuda_error(cudaStreamSynchronize(stream));
            ft::check_cuda_error(cudaStreamDestroy(stream));
        },
        "accepted_lengths_cumsum"_a,
        "paths_offsets"_a,
        "accepted_lengths"_a,
        "best_path_ids"_a,
        "paths"_a,
        "batch_slots"_a = py::none());

    // EAGLE A12/A36: KV rewind binding for direct kernel tests – see EAGLE_TODO.md.
    m.def(
        "eagle_kv_cache_rewind",
        [](py::object rewind_lengths,
           py::object batch_slots,
           py::object block_tables,
           int num_layers,
           int block_size,
           py::object kv_cache_blocks_opt) {
            auto lengths_shape = rewind_lengths.attr("shape").cast<py::tuple>();
            auto slots_shape   = batch_slots.attr("shape").cast<py::tuple>();
            auto tables_shape  = block_tables.attr("shape").cast<py::tuple>();

            if (lengths_shape.size() != 1 || slots_shape.size() != 1 || tables_shape.size() != 2) {
                throw std::invalid_argument("eagle_kv_cache_rewind: unexpected tensor ranks");
            }

            const auto max_batch_size      = tables_shape[0].cast<int64_t>();
            const auto max_blocks_per_seq  = tables_shape[1].cast<int64_t>();
            const auto batch_size          = slots_shape[0].cast<int64_t>();

            auto device_obj  = block_tables.attr("device");
            auto device_type = device_obj.attr("type").cast<std::string>();
            if (device_type != "cuda") {
                throw std::runtime_error("eagle_kv_cache_rewind expects CUDA tensors for block_tables");
            }

            int device_index = 0;
            if (!device_obj.attr("index").is_none()) {
                device_index = device_obj.attr("index").cast<int>();
            }
            ft::CudaDeviceGuard device_guard(device_index);

            auto get_int32_ptr = [](py::object const& tensor) -> eagle_kernels::SizeType* {
                auto ptr_obj = tensor.attr("data_ptr")();
                auto ptr_val = ptr_obj.cast<uintptr_t>();
                return reinterpret_cast<eagle_kernels::SizeType*>(ptr_val);
            };

            eagle_kernels::SizeType* rewind_lengths_ptr = get_int32_ptr(rewind_lengths);
            eagle_kernels::SizeType* batch_slots_ptr    = get_int32_ptr(batch_slots);
            eagle_kernels::SizeType* block_tables_ptr   = get_int32_ptr(block_tables);

            (void)kv_cache_blocks_opt;  // kv_cache_blocks is not wired in this binding yet.
            void** kv_cache_blocks_ptr = nullptr;

            cudaStream_t stream{};
            ft::check_cuda_error(cudaStreamCreate(&stream));

            eagle_kernels::KVCacheRewindParams params{};
            params.kv_cache_blocks    = kv_cache_blocks_ptr;
            params.rewind_lengths     = rewind_lengths_ptr;
            params.batch_slots        = batch_slots_ptr;
            params.block_tables       = block_tables_ptr;
            params.batch_size         = static_cast<eagle_kernels::SizeType>(batch_size);
            params.max_batch_size     = static_cast<eagle_kernels::SizeType>(max_batch_size);
            params.num_layers         = static_cast<eagle_kernels::SizeType>(num_layers);
            params.block_size         = static_cast<eagle_kernels::SizeType>(block_size);
            params.max_blocks_per_seq = static_cast<eagle_kernels::SizeType>(max_blocks_per_seq);
            params.stream             = stream;

            eagle_kernels::invokeKVCacheRewind(params);

            ft::check_cuda_error(cudaStreamSynchronize(stream));
            ft::check_cuda_error(cudaStreamDestroy(stream));
        },
        "rewind_lengths"_a,
        "batch_slots"_a,
        "block_tables"_a,
        "num_layers"_a,
        "block_size"_a,
        "kv_cache_blocks"_a = py::none());

    // EAGLE A31/A32: prototype tree-accept binding – see EAGLE_TODO.md (🧪, GPU/CI validation pending).
    m.def(
        "eagle_tree_accept_tokens",
        [](py::object draft_ids,
           py::object target_ids,
           py::object paths,
           py::object best_path_ids,
           py::object accepted_lengths,
           py::object accepted_tokens,
           py::object batch_slots_opt) {
            auto draft_shape  = draft_ids.attr("shape").cast<py::tuple>();
            auto target_shape = target_ids.attr("shape").cast<py::tuple>();
            auto paths_shape  = paths.attr("shape").cast<py::tuple>();
            auto best_shape   = best_path_ids.attr("shape").cast<py::tuple>();
            auto lens_shape   = accepted_lengths.attr("shape").cast<py::tuple>();
            auto acc_tok_shape = accepted_tokens.attr("shape").cast<py::tuple>();

            if (draft_shape.size() != 2 || target_shape.size() != 2 || paths_shape.size() != 3
                || best_shape.size() != 1 || lens_shape.size() != 1 || acc_tok_shape.size() != 2) {
                throw std::invalid_argument("eagle_tree_accept_tokens: unexpected tensor ranks");
            }

            const auto max_batch_size   = draft_shape[0].cast<int64_t>();
            const auto max_draft_tokens = draft_shape[1].cast<int64_t>();
            const auto batch_size       = lens_shape[0].cast<int64_t>();
            const auto num_paths        = paths_shape[1].cast<int64_t>();
            const auto max_path_len     = paths_shape[2].cast<int64_t>();

            auto device_obj = draft_ids.attr("device");
            auto device_type = device_obj.attr("type").cast<std::string>();
            if (device_type != "cuda") {
                throw std::runtime_error("eagle_tree_accept_tokens expects CUDA tensors");
            }

            int device_index = 0;
            if (!device_obj.attr("index").is_none()) {
                device_index = device_obj.attr("index").cast<int>();
            }
            ft::CudaDeviceGuard device_guard(device_index);

            auto get_int32_ptr = [](py::object const& tensor) -> eagle_kernels::SizeType* {
                auto ptr_obj = tensor.attr("data_ptr")();
                auto ptr_val = ptr_obj.cast<uintptr_t>();
                return reinterpret_cast<eagle_kernels::SizeType*>(ptr_val);
            };

            auto get_token_ptr = [](py::object const& tensor) -> eagle_kernels::TokenIdType* {
                auto ptr_obj = tensor.attr("data_ptr")();
                auto ptr_val = ptr_obj.cast<uintptr_t>();
                return reinterpret_cast<eagle_kernels::TokenIdType*>(ptr_val);
            };

            eagle_kernels::TokenIdType* draft_ids_ptr   = get_token_ptr(draft_ids);
            eagle_kernels::TokenIdType* target_ids_ptr  = get_token_ptr(target_ids);
            eagle_kernels::SizeType*    paths_ptr       = get_int32_ptr(paths);
            eagle_kernels::SizeType*    best_path_ids_ptr = get_int32_ptr(best_path_ids);
            eagle_kernels::SizeType*    accepted_lens_ptr = get_int32_ptr(accepted_lengths);
            eagle_kernels::TokenIdType* accepted_tokens_ptr = get_token_ptr(accepted_tokens);

            eagle_kernels::SizeType* batch_slots_ptr = nullptr;
            if (!batch_slots_opt.is_none()) {
                batch_slots_ptr = get_int32_ptr(batch_slots_opt);
            }

            cudaStream_t stream{};
            ft::check_cuda_error(cudaStreamCreate(&stream));

            eagle_kernels::invokeTreeAcceptByIdsWithPaths(
                draft_ids_ptr,
                target_ids_ptr,
                paths_ptr,
                batch_slots_ptr,
                static_cast<eagle_kernels::SizeType>(batch_size),
                static_cast<eagle_kernels::SizeType>(max_batch_size),
                static_cast<eagle_kernels::SizeType>(num_paths),
                static_cast<eagle_kernels::SizeType>(max_path_len),
                static_cast<eagle_kernels::SizeType>(max_draft_tokens),
                best_path_ids_ptr,
                accepted_lens_ptr,
                accepted_tokens_ptr,
                stream);

            ft::check_cuda_error(cudaStreamSynchronize(stream));
            ft::check_cuda_error(cudaStreamDestroy(stream));
        },
        "draft_ids"_a,
        "target_ids"_a,
        "paths"_a,
        "best_path_ids"_a,
        "accepted_lengths"_a,
        "accepted_tokens"_a,
        "batch_slots"_a = py::none());

    // Kernel-level acceptance metrics: compute per-request acceptance rate
    // given accepted lengths and draft lengths.
    m.def(
        "compute_acceptance_stats",
    [](py::args const&, py::kwargs const&) {
        throw std::runtime_error(
            "compute_acceptance_stats is not implemented in this build; "
            "use req_metrics.spec_info / EagleMetricsSummary for EAGLE metrics.");
    },
    "Stub for legacy acceptance-stats helper; not implemented in this build.");
    // Test-only harness for EagleModule::forward.
    //
    // This helper constructs an EagleModule from a given draft-model
    // directory, runs forward twice for a small synthetic batch, and
    // reports shapes plus whether the logits / hidden-state buffers
    // were reused between calls (i.e. no per-step reallocations in
    // the speculative decode hot path).
    m.def(
        "eagle_forward_smoke",
        [](const std::string& model_dir, int batch_size) {
            if (batch_size <= 0) {
                throw std::invalid_argument("batch_size must be positive");
            }

            int device_id = 0;
            ft::CudaDeviceGuard device_guard(device_id);

            cudaStream_t stream{};
            ft::check_cuda_error(cudaStreamCreate(&stream));

            // Use small but non-trivial limits; the concrete shapes
            // are driven by the draft model config in model_dir.
            constexpr ft::EagleModule::SizeType kMaxDraftPathLen       = 16;
            constexpr ft::EagleModule::SizeType kMaxDecodingDraftTokens = 16;
            constexpr ft::EagleModule::SizeType kMaxDecodingTokens     = 16;
            constexpr ft::EagleModule::SizeType kMaxNonLeafNodes       = 32;

            ft::EagleModule module(
                kMaxDraftPathLen,
                kMaxDecodingDraftTokens,
                kMaxDecodingTokens,
                kMaxNonLeafNodes);

            module.load(model_dir, device_id, stream);

            auto const& weights = module.getWeights();
            if (!weights.embed_tokens || !weights.lm_head) {
                cudaStreamDestroy(stream);
                throw std::runtime_error(
                    "EagleModule weights not initialized; check draft model directory");
            }

            const int hidden_units = static_cast<int>(weights.embed_tokens.shape(1));
            const int vocab_size   = static_cast<int>(weights.lm_head.shape(1));

            // Allocate synthetic hidden states and input IDs on device.
            Tensor hidden_states(
                std::vector<ft::core::ssize_t>{batch_size, hidden_units},
                ft::data_type_v<ft::half_t>,
                ft::kDEVICE);
            Tensor input_ids(
                std::vector<ft::core::ssize_t>{batch_size},
                ft::data_type_v<int32_t>,
                ft::kDEVICE);

            ft::check_cuda_error(
                cudaMemsetAsync(hidden_states.raw_data(), 0, hidden_states.byte_size(), stream));
            ft::check_cuda_error(
                cudaMemsetAsync(input_ids.raw_data(), 0, input_ids.byte_size(), stream));

            ft::LlamaLinear linear(stream);

            Tensor logits_1;
            Tensor hidden_1;
            module.forward(input_ids, hidden_states, logits_1, hidden_1, linear, stream);

            ft::check_cuda_error(cudaStreamSynchronize(stream));

            void* logits_ptr_1 = logits_1.raw_data();
            void* hidden_ptr_1 = hidden_1.raw_data();

            Tensor logits_2;
            Tensor hidden_2;
            module.forward(input_ids, hidden_states, logits_2, hidden_2, linear, stream);

            ft::check_cuda_error(cudaStreamSynchronize(stream));
            ft::check_cuda_error(cudaStreamDestroy(stream));

            const bool reuse_logits = logits_ptr_1 == logits_2.raw_data();
            const bool reuse_hidden = hidden_ptr_1 == hidden_2.raw_data();

            auto logits_shape = logits_1.shape();
            auto hidden_shape = hidden_1.shape();

            return py::dict("logits_shape"_a = logits_shape,
                            "hidden_shape"_a = hidden_shape,
                            "reuse_logits_buffer"_a = reuse_logits,
                            "reuse_hidden_buffer"_a = reuse_hidden,
                            "vocab_size"_a = vocab_size,
                            "hidden_units"_a = hidden_units);
        },
        "model_dir"_a,
        "batch_size"_a = 2);

    // Microbenchmark helper for EagleModule::forward.
    //
    // Runs a configurable number of forward passes for a synthetic batch
    // and reports basic latency statistics. Intended for offline tuning
    // and regression checks, not for production inference.
    m.def(
        "eagle_forward_bench",
        [](const std::string& model_dir, int batch_size, int iters) {
            if (batch_size <= 0) {
                throw std::invalid_argument("batch_size must be positive");
            }
            if (iters <= 0) {
                throw std::invalid_argument("iters must be positive");
            }

            int device_id = 0;
            ft::CudaDeviceGuard device_guard(device_id);

            cudaStream_t stream{};
            ft::check_cuda_error(cudaStreamCreate(&stream));

            constexpr ft::EagleModule::SizeType kMaxDraftPathLen       = 16;
            constexpr ft::EagleModule::SizeType kMaxDecodingDraftTokens = 16;
            constexpr ft::EagleModule::SizeType kMaxDecodingTokens     = 16;
            constexpr ft::EagleModule::SizeType kMaxNonLeafNodes       = 32;

            ft::EagleModule module(
                kMaxDraftPathLen,
                kMaxDecodingDraftTokens,
                kMaxDecodingTokens,
                kMaxNonLeafNodes);

            module.load(model_dir, device_id, stream);

            auto const& weights = module.getWeights();
            if (!weights.embed_tokens || !weights.lm_head) {
                cudaStreamDestroy(stream);
                throw std::runtime_error(
                    "EagleModule weights not initialized; check draft model directory");
            }

            const int hidden_units = static_cast<int>(weights.embed_tokens.shape(1));
            const int vocab_size   = static_cast<int>(weights.lm_head.shape(1));

            Tensor hidden_states(
                std::vector<ft::core::ssize_t>{batch_size, hidden_units},
                ft::data_type_v<ft::half_t>,
                ft::kDEVICE);
            Tensor input_ids(
                std::vector<ft::core::ssize_t>{batch_size},
                ft::data_type_v<int32_t>,
                ft::kDEVICE);

            ft::check_cuda_error(
                cudaMemsetAsync(hidden_states.raw_data(), 0, hidden_states.byte_size(), stream));
            ft::check_cuda_error(
                cudaMemsetAsync(input_ids.raw_data(), 0, input_ids.byte_size(), stream));

            ft::LlamaLinear linear(stream);

            Tensor logits;
            Tensor hidden;

            // Warmup one run.
            module.forward(input_ids, hidden_states, logits, hidden, linear, stream);
            ft::check_cuda_error(cudaStreamSynchronize(stream));

            cudaEvent_t start, stop;
            ft::check_cuda_error(cudaEventCreate(&start));
            ft::check_cuda_error(cudaEventCreate(&stop));

            ft::check_cuda_error(cudaEventRecord(start, stream));
            for (int i = 0; i < iters; ++i) {
                module.forward(input_ids, hidden_states, logits, hidden, linear, stream);
            }
            ft::check_cuda_error(cudaEventRecord(stop, stream));
            ft::check_cuda_error(cudaEventSynchronize(stop));

            float elapsed_ms = 0.0f;
            ft::check_cuda_error(cudaEventElapsedTime(&elapsed_ms, start, stop));

            ft::check_cuda_error(cudaEventDestroy(start));
            ft::check_cuda_error(cudaEventDestroy(stop));
            ft::check_cuda_error(cudaStreamDestroy(stream));

            const double avg_ms_per_forward = static_cast<double>(elapsed_ms) / static_cast<double>(iters);
            const double tokens_per_forward = static_cast<double>(batch_size);
            const double tokens_per_second =
                (avg_ms_per_forward > 0.0)
                ? (tokens_per_forward * 1000.0 / avg_ms_per_forward)
                : 0.0;

            return py::dict("batch_size"_a = batch_size,
                            "iters"_a = iters,
                            "avg_ms_per_forward"_a = avg_ms_per_forward,
                            "tokens_per_second"_a = tokens_per_second,
                            "hidden_units"_a = hidden_units,
                            "vocab_size"_a = vocab_size);
        },
        "model_dir"_a,
        "batch_size"_a = 2,
        "iters"_a = 50);

    // DLpack bridge for pre-existing Tensor bindings.
    m.def(
        "from_dlpack",
        [](py::object obj) {
            py::capsule      cap = obj.attr("__dlpack__")();
            DLManagedTensor* dlmt =
                static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap.ptr(), kDlTensorCapsuleName));
            auto ret = DLManagedTensorToTritonTensor(dlmt);
            // take ownership of capsule's payload
            cap.set_name("used_dltensor");
            return ret;
        },
        "dl_managed_tensor"_a);

    py::bind_map<TensorMap, std::shared_ptr<TensorMap>>(m, "TensorMap");

    using ft::ModelRequest;
    py::class_<ModelRequest>(m, "ModelRequest")
        .def(
            "forward",
            [](ModelRequest*              model_request,
               std::shared_ptr<TensorMap> input_tensors,
               py::object                 session_obj,
               py::object                 gen_cfg_obj,
               bool                       stream_output,
               bool                       enable_metrics,
               std::function<void()>      cb) {
                // Build a C++ SessionParam from a lightweight Python
                // object with the same fields (id, step, start, end).
                ft::SessionParam session{};
                if (!session_obj.is_none()) {
                    try {
                        auto id_attr = session_obj.attr("id");
                        if (!id_attr.is_none()) {
                            session.id = id_attr.cast<uint64_t>();
                        }
                    }
                    catch (const py::error_already_set&) {
                        PyErr_Clear();
                    }
                    try {
                        auto step_attr = session_obj.attr("step");
                        if (!step_attr.is_none()) {
                            session.step = step_attr.cast<int>();
                        }
                    }
                    catch (const py::error_already_set&) {
                        PyErr_Clear();
                    }
                    try {
                        auto start_attr = session_obj.attr("start");
                        if (!start_attr.is_none()) {
                            session.start_flag = start_attr.cast<bool>();
                        }
                    }
                    catch (const py::error_already_set&) {
                        PyErr_Clear();
                    }
                    try {
                        auto end_attr = session_obj.attr("end");
                        if (!end_attr.is_none()) {
                            session.end_flag = end_attr.cast<bool>();
                        }
                    }
                    catch (const py::error_already_set&) {
                        PyErr_Clear();
                    }
                }

                // Build a C++ GenerationConfig from a lightweight Python
                // object (usually lmdeploy.messages.GenerationConfig or a
                // compatible namespace) without exposing the C++ type to
                // pybind11.
                ft::GenerationConfig gen_cfg{};
                if (!gen_cfg_obj.is_none()) {
                    auto get_attr_int = [&](const char* name, int default_val) {
                        try {
                            auto attr = gen_cfg_obj.attr(name);
                            if (!attr.is_none()) {
                                return attr.cast<int>();
                            }
                        }
                        catch (const py::error_already_set&) {
                            PyErr_Clear();
                        }
                        return default_val;
                    };
                    auto get_attr_double = [&](const char* name, double default_val) {
                        try {
                            auto attr = gen_cfg_obj.attr(name);
                            if (!attr.is_none()) {
                                return attr.cast<double>();
                            }
                        }
                        catch (const py::error_already_set&) {
                            PyErr_Clear();
                        }
                        return default_val;
                    };

                    gen_cfg.max_new_tokens = get_attr_int("max_new_tokens", gen_cfg.max_new_tokens);
                    gen_cfg.min_new_tokens = get_attr_int("min_new_tokens", gen_cfg.min_new_tokens);
                    gen_cfg.top_k          = get_attr_int("top_k", gen_cfg.top_k);
                    gen_cfg.output_logprobs
                        = get_attr_int("output_logprobs", gen_cfg.output_logprobs);

                    // float fields
                    gen_cfg.top_p            = get_attr_double("top_p", gen_cfg.top_p);
                    gen_cfg.min_p            = get_attr_double("min_p", gen_cfg.min_p);
                    gen_cfg.temperature      = get_attr_double("temperature", gen_cfg.temperature);
                    gen_cfg.repetition_penalty
                        = get_attr_double("repetition_penalty", gen_cfg.repetition_penalty);

                    // eos_ids, if present, are passed from Python as a
                    // flat list of ints.
                    auto try_copy_vec_int = [&](const char* name, std::vector<int>& dst) {
                        try {
                            auto attr = gen_cfg_obj.attr(name);
                            if (!attr.is_none()) {
                                dst = attr.cast<std::vector<int>>();
                            }
                        }
                        catch (const py::error_already_set&) {
                            PyErr_Clear();
                        }
                    };
                    try_copy_vec_int("eos_ids", gen_cfg.eos_ids);

                    // random_seed is optional
                    try {
                        auto attr = gen_cfg_obj.attr("random_seed");
                        if (!attr.is_none()) {
                            gen_cfg.random_seed = attr.cast<uint64_t>();
                        }
                    }
                    catch (const py::error_already_set&) {
                        PyErr_Clear();
                    }

                    // output_last_hidden_state / output_logits may be encoded
                    // as strings ("all" / "generation"); if so, map them here.
                    auto map_output_mode = [](py::object const& v) -> int {
                        if (v.is_none()) {
                            return 0;
                        }
                        if (py::isinstance<py::str>(v)) {
                            std::string s = v.cast<std::string>();
                            if (s == "all") {
                                return ft::GenerationConfig::kAll;
                            }
                            if (s == "generation") {
                                return ft::GenerationConfig::kGeneration;
                            }
                        }
                        return v.cast<int>();
                    };
                    try {
                        auto attr = gen_cfg_obj.attr("output_last_hidden_state");
                        if (!attr.is_none()) {
                            gen_cfg.output_last_hidden_state = map_output_mode(attr);
                        }
                    }
                    catch (const py::error_already_set&) {
                        PyErr_Clear();
                    }
                    try {
                        auto attr = gen_cfg_obj.attr("output_logits");
                        if (!attr.is_none()) {
                            gen_cfg.output_logits = map_output_mode(attr);
                        }
                    }
                    catch (const py::error_already_set&) {
                        PyErr_Clear();
                    }
                }

                ModelRequest::InputParam param{};
                param.tensors        = std::move(input_tensors);
                param.session        = session;
                param.gen_cfg        = gen_cfg;
                param.stream_output  = stream_output;
                param.enable_metrics = enable_metrics;

                auto ret = model_request->Forward(std::move(param), [cb = std::move(cb)]() {
                    try {
                        cb();
                    }
                    catch (const py::error_already_set& e) {
                        std::cerr << e.what() << std::endl;
                    }
                });
                // Convert C++ RequestMetrics (if present) into a plain
                // Python dict so that we do not expose the C++ type via
                // pybind11 (avoids generic_type re-registration issues).
                py::object py_metrics = py::none();
                if (ret.metrics) {
                    const ft::RequestMetrics& m = *ret.metrics;
                    py::dict d;
                    d["enque_time"]                 = m.enque_time;
                    d["scheduled_time"]             = m.scheduled_time;
                    d["eagle_total_draft_tokens"]   = m.eagle_total_draft_tokens;
                    d["eagle_total_accepted_tokens"] = m.eagle_total_accepted_tokens;
                    d["eagle_steps"]                 = m.eagle_steps;
                    d["eagle_total_rewound_tokens"]  = m.eagle_total_rewound_tokens;
                    d["eagle_rewind_steps"]          = m.eagle_rewind_steps;
                    py_metrics = std::move(d);
                }
                // Expose a callable to consume the shared state without
                // binding the underlying AtomicRequestState / RequestState
                // C++ types. The callable returns either:
                //   - None, or
                //   - a dict with keys: status, seq_len
                py::object state_consumer = py::none();
                if (ret.state) {
                    auto state_ptr = ret.state;
                    state_consumer = py::cpp_function(
                        [state_ptr]() -> py::object {
                            auto st = state_ptr->exchange(nullptr);
                            if (!st) {
                                return py::none();
                            }
                            py::dict d;
                            d["status"] = st->status;
                            d["seq_len"] = st->seq_len;
                            return py::object(std::move(d));
                        });
                }

                return std::make_tuple(std::move(ret.tensors),
                                       std::move(state_consumer),
                                       std::move(py_metrics));
            },
            py::call_guard<py::gil_scoped_release>(),
            "input_tensors"_a,
            "session"_a,
            "gen_cfg"_a,
            "stream_output"_a,
            "enable_metrics"_a,
            "cb"_a)
        .def(
            "cancel",
            [](ModelRequest* model_request) {
                model_request->Cancel();  //
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "end",
            [](ModelRequest* model_request, std::function<void(int)> cb, uint64_t session_id) {
                model_request->End(std::move(cb), session_id);  //
            },
            py::call_guard<py::gil_scoped_release>(),
            "cb"_a,
            "session_id"_a)
        .def(
            "set_grammar",
            [](ModelRequest* model_request, const xgrammar::CompiledGrammar& grammar) {
                TM_LOG_INFO("Set grammar for model_request");
                model_request->setGrammar(grammar);
            },
            py::call_guard<py::gil_scoped_release>(),
            "grammar"_a);

    // transformer model
    using ft::LlamaTritonModel;
    py::class_<LlamaTritonModel, std::shared_ptr<LlamaTritonModel>>(m, "AbstractTransformerModel")
        .def_static(
            "create_llama_model",
            [](std::string model_dir,
               std::string config,
               std::string weight_type) -> std::shared_ptr<LlamaTritonModel> {
                auto gil_factory = [] {  //
                    // erase the type
                    return std::static_pointer_cast<void>(std::make_shared<ScopedGIL>());
                };
                auto no_gil_deleter = [](LlamaTritonModel* ptr) {
                    pybind11::gil_scoped_release release;
                    delete ptr;
                };

                std::shared_ptr<LlamaTritonModel> model(new LlamaTritonModel(model_dir, config, gil_factory),
                                                        no_gil_deleter);
                return model;
            },
            "model_dir"_a,
            "config"_a      = "",
            "weight_type"_a = "half")
        .def(
            "create_model_instance",
            [](LlamaTritonModel* model, int deviceId) { return model->createModelInstance(deviceId); },
            py::call_guard<py::gil_scoped_release>(),
            "device_id"_a)
        .def("create_shared_weights",
             &LlamaTritonModel::createSharedWeights,
             py::call_guard<py::gil_scoped_release>(),
             "device_id"_a,
             "rank"_a)
        .def(
            "get_params",
            [](LlamaTritonModel* model, int deviceId, int rank) { return model->getParams(deviceId, rank); },
            py::call_guard<py::gil_scoped_release>(),
            "device_id"_a,
            "rank"_a)
        .def(
            "process_weight",
            [](LlamaTritonModel* model, int deviceId, int rank) { model->processWeights(deviceId, rank); },
            py::call_guard<py::gil_scoped_release>(),
            "device_id"_a,
            "rank"_a)
        .def(
            "create_engine",
            [](LlamaTritonModel* model, int deviceId, int rank) { model->createEngine(deviceId, rank); },
            py::call_guard<py::gil_scoped_release>(),
            "device_id"_a,
            "rank"_a)
        .def(
            "get_schedule_metrics",
            [](LlamaTritonModel* model, int deviceId, int rank) {
                ft::ScheduleMetrics m = model->getScheduleMetrics(deviceId, rank);
                py::dict            d;
                d["total_seqs"]    = m.total_seqs;
                d["active_seqs"]   = m.active_seqs;
                d["waiting_seqs"]  = m.waiting_seqs;
                d["total_blocks"]  = m.total_blocks;
                d["active_blocks"] = m.active_blocks;
                d["cached_blocks"] = m.cached_blocks;
                d["free_blocks"]   = m.free_blocks;
                return d;
            },
            py::call_guard<py::gil_scoped_release>(),
            "device_id"_a,
            "rank"_a)
        .def(
            "sleep",
            [](LlamaTritonModel* model, int deviceId, int level) { model->sleep(deviceId, level); },
            py::call_guard<py::gil_scoped_release>(),
            "device_id"_a,
            "level"_a)
        .def(
            "wakeup",
            [](LlamaTritonModel* model, int deviceId, const std::vector<std::string>& tags, int rank) {
                model->wakeup(deviceId, tags, rank);
            },
            py::call_guard<py::gil_scoped_release>(),
            "device_id"_a,
            "tags"_a,
            "rank"_a)
        .def("__str__", &LlamaTritonModel::toString)
        .def("__repr__", &LlamaTritonModel::toString)
        .def("get_tensor_para_size", &LlamaTritonModel::getTensorParaSize)
        .def("get_pipeline_para_size", &LlamaTritonModel::getPipelineParaSize);
}
