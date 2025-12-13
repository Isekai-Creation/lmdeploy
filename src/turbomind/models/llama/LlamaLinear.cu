// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/cuda_data_type.h"
#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/kernels/gemm/types.h"

#include "src/turbomind/kernels/quantization.h"

#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"

#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

using namespace gemm;

namespace {

// Simple row-major GEMM fallback: C[M,N] = A[M,K] @ B[K,N]
template<typename T>
__device__ inline float to_float_fallback(T x)
{
    return static_cast<float>(x);
}

template<>
__device__ inline float to_float_fallback<half_t>(half_t x)
{
    return __half2float(x);
}

#if ENABLE_BF16
template<>
__device__ inline float to_float_fallback<bfloat16_t>(bfloat16_t x)
{
    return __bfloat162float(x);
}
#endif

template<typename T>
__device__ inline T from_float_fallback(float x)
{
    return static_cast<T>(x);
}

template<>
__device__ inline half_t from_float_fallback<half_t>(float x)
{
    return __float2half(x);
}

#if ENABLE_BF16
template<>
__device__ inline bfloat16_t from_float_fallback<bfloat16_t>(float x)
{
    return __float2bfloat16(x);
}
#endif

template<typename T>
__global__ void NaiveRowMajorGemmKernel(const T* __restrict__ A,
                                        const T* __restrict__ B,
                                        T* __restrict__       C,
                                        int                   M,
                                        int                   K,
                                        int                   N,
                                        int                   lda,
                                        int                   ldb,
                                        int                   ldc)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }

    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        const float a = to_float_fallback<T>(A[row * lda + k]);
        const float b = to_float_fallback<T>(B[k * ldb + col]);
        acc += a * b;
    }
    C[row * ldc + col] = from_float_fallback<T>(acc);
}

template<typename T>
void launch_naive_rowmajor_gemm(const Tensor& A, const Tensor& B, Tensor& C, cudaStream_t stream)
{
    const int M   = static_cast<int>(A.shape(0));
    const int K   = static_cast<int>(A.shape(1));
    const int N   = static_cast<int>(B.shape(1));
    const int lda = static_cast<int>(A.stride(0));
    const int ldb = static_cast<int>(B.stride(0));
    const int ldc = static_cast<int>(C.stride(0));

    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    // Sanity-check that the inner dimensions line up before launching
    // the fallback kernel. When they do not (e.g. due to a higher‑level
    // geometry mismatch), we skip the GEMM and leave C cleared to
    // avoid illegal memory accesses.
    if (B.shape(0) != K) {
        TM_LOG_ERROR(
            "[LlamaLinear][fallback] Skipping naive GEMM due to incompatible shapes: "
            "A=(%d,%d) B=(%d,%d) C=(%d,%d)",
            M,
            K,
            static_cast<int>(B.shape(0)),
            static_cast<int>(B.shape(1)),
            static_cast<int>(C.shape(0)),
            static_cast<int>(C.shape(1)));
        check_cuda_error(cudaMemsetAsync(C.raw_data(), 0, C.byte_size(), stream));
        return;
    }

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    NaiveRowMajorGemmKernel<T><<<grid, block, 0, stream>>>(
        static_cast<const T*>(A.raw_data()),
        static_cast<const T*>(B.raw_data()),
        static_cast<T*>(C.raw_data()),
        M,
        K,
        N,
        lda,
        ldb,
        ldc);
}

}  // anonymous namespace

struct LlamaLinear::Impl {

    explicit Impl(cudaStream_t stream): stream_(stream)
    {
        workspace_ = {};

        workspace_.barriers_size   = gemm::Gemm::kBarriersSize;
        workspace_.partials_size   = gemm::Gemm::kPartialsSize;
        workspace_.tensormaps_size = 8192 * 128;  // maximum 4096 tensor maps

        check_cuda_error(cudaMallocAsync(&workspace_.barriers, workspace_.barriers_size, stream_));
        check_cuda_error(cudaMallocAsync(&workspace_.partials, workspace_.partials_size, stream_));
        check_cuda_error(cudaMallocAsync(&workspace_.tensormaps, workspace_.partials_size, stream_));
        check_cuda_error(cudaMemsetAsync(workspace_.barriers, 0, workspace_.barriers_size, stream_));
        check_cuda_error(cudaMalloc(&workspace_.flags, sizeof(int)));
    }

    ~Impl()
    {
        cudaFreeAsync(workspace_.barriers, stream_);
        cudaFreeAsync(workspace_.partials, stream_);
        cudaFreeAsync(workspace_.tensormaps, stream_);
        cudaFreeAsync(workspace_.flags, stream_);
        workspace_ = {};
    }

    std::tuple<Tensor, MatrixLayout, Tensor, MatrixLayout> GetOperandB(const LlamaDenseWeight& dense)
    {
        const Tensor& B      = dense.weight;
        const Tensor& V      = dense.scales;
        MatrixLayout  desc_B = dense.k_desc;
        MatrixLayout  desc_V = dense.q_desc;
        return {B, desc_B, V, desc_V};
    }

    std::tuple<Tensor, MatrixLayout, Tensor, MatrixLayout>
    GetOperandA(const LlamaDenseWeight& dense, const Tensor& input, Buffer_<int> indices, const Buffer_<int>& offsets)
    {
        Tensor A;
        Tensor U;

        const int m = indices ? indices.size() : input.shape(0);

        // Currently, FP8 only; INT8 may be added later
        if (input.dtype() != dense.input_type) {
            QuantizeSymm(A, U, input, stream_);
            sync_check_cuda_error();
        }
        else {
            A = input;
        }

        if (indices && A.dtype() == kFloat8_e4m3) {
            const auto [bsz, k] = A.shapes(0, 1);
            const int e         = indices.size() / bsz;
            Tensor    A_e       = {{m, k}, A.dtype(), kDEVICE};
            invokeMoeDispatch(A_e, A, indices.data(), e, stream_);
            sync_check_cuda_error();
            Tensor U_e;
            invokeMoeDispatchScales(U_e, U, indices.data(), e, stream_);
            sync_check_cuda_error();
            A       = A_e;
            U       = U_e;
            indices = {};  // indices already applied
        }

        MatrixLayout desc_A{A.dtype(), gemm::Order::kRowMajor, m, (int)A.shape(1), (int)A.stride(0)};
        MatrixLayout desc_U{};
        if (U) {
            desc_U = {U.dtype(), kColMajor, (int)U.shape(1), (int)U.shape(0), (int)U.stride(0)};
        }
        if (offsets) {
            desc_A.num = desc_U.num = dense.k_desc.num;
            desc_A.offsets = desc_U.offsets = const_cast<int*>(offsets.data());
        }
        if (indices) {
            desc_A.idxs = desc_U.idxs = const_cast<int*>(indices.data());
        }

        return {A, desc_A, U, desc_U};
    }

    void Forward(Tensor&                 output,
                 const Tensor&           input,  //
                 const LlamaDenseWeight& dense,
                 const Buffer_<int>&     indices,
                 const Buffer_<int>&     offsets)
    {
        using namespace gemm;

        Operation op{};
        op.dispatch  = dispatch_policy_;
        op.epilogue  = dense.epilogue;
        op.quant_a   = dense.input_quant;
        op.quant_b   = dense.weight_quant;
        op.batch_dim = 0;

        auto&& [A, desc_A, U, desc_U] = GetOperandA(dense, input, indices, offsets);
        auto&& [B, desc_B, V, desc_V] = GetOperandB(dense);

        Tensor& D = output;
        if (!D) {
            int dim = dense.epilogue == Epilogue::kGatedSilu ? dense.output_dim / 2 : dense.output_dim;
            D       = Tensor{{desc_A.rows, dim}, dense.data_type, kDEVICE};
        }

        // std::cout << "D: " << D << " " << desc_B.num << "\n";

        MatrixLayout desc_D{
            output.dtype(),
            kRowMajor,
            (int)output.shape(0),
            dense.output_dim,
            (int)output.stride(0),
        };

        if (offsets) {
            desc_D.num     = desc_B.num;
            desc_D.offsets = const_cast<int*>(offsets.data());
        }

        if (turbomind::isEnvVarEnabled("LMDEPLOY_EAGLE_GEMM_SHAPE_LOG")) {
            static int logged = 0;
            // Log only a limited number of shapes to avoid console spam.
            if (logged < 64) {
                TM_LOG_INFO(
                    "[EAGLE3][GEMM] op=MxKxN(%d,%d,%d) dtype=%d weight_dtype=%d input_dtype=%d orderA=%d "
                    "orderB=%d orderD=%d",
                    desc_A.rows,
                    desc_A.cols,
                    desc_D.cols,
                    static_cast<int>(dense.data_type),
                    static_cast<int>(dense.weight_type),
                    static_cast<int>(dense.input_type),
                    static_cast<int>(desc_A.order),
                    static_cast<int>(desc_B.order),
                    static_cast<int>(desc_D.order));
                ++logged;
            }
        }

        auto ec = gemm_.Run(op,
                            1.f,
                            A.raw_data(),
                            desc_A,
                            U.data_or((void*)nullptr),
                            desc_U,
                            B.raw_data(),
                            desc_B,
                            V.data_or((void*)nullptr),
                            desc_V,
                            0.f,
                            D.raw_data(),
                            desc_D,
                            D.raw_data(),
                            desc_D,
                            workspace_,
                            stream_);

        if (ec) {
            TM_LOG_ERROR("%s: %d", __PRETTY_FUNCTION__, ec);
            // Fallback: for non-quantized BF16/FP16 dense GEMMs that our
            // fused kernels do not cover (e.g. new SM architectures),
            // fall back to a simple row-major GEMM so that higher-level
            // code (including Eagle3) can continue to run.
            const bool no_quant = dense.input_quant.type == QuantType::kNone
                && dense.weight_quant.type == QuantType::kNone;
            if (no_quant && (dense.data_type == kBfloat16 || dense.data_type == kFloat16)
                && dense.data_type == dense.weight_type && dense.data_type == dense.input_type
                && desc_A.order == kRowMajor && desc_B.order == kRowMajor && desc_D.order == kRowMajor) {
                TM_LOG_ERROR(
                    "[LlamaLinear][fallback] GEMM failed for dtype=%d; falling back to naive row-major matmul.",
                    static_cast<int>(dense.data_type));

                if (dense.data_type == kFloat16) {
                    launch_naive_rowmajor_gemm<half_t>(A, B, D, stream_);
                }
#if ENABLE_BF16
                else if (dense.data_type == kBfloat16) {
                    launch_naive_rowmajor_gemm<bfloat16_t>(A, B, D, stream_);
                }
#endif
                sync_check_cuda_error();
            }
        }
    }

    gemm::Gemm           gemm_;
    gemm::DispatchPolicy dispatch_policy_{gemm::DispatchPolicy::kDefault};
    cudaStream_t         stream_{};

    gemm::Workspace workspace_;
};

LlamaLinear::LlamaLinear(cudaStream_t stream): impl_{std::make_shared<Impl>(stream)} {}

Tensor LlamaLinear::Forward(const Tensor&           input,  //
                            const LlamaDenseWeight& weight,
                            std::optional<Tensor>   output)
{
    return Forward(input, weight, {}, {}, output);
}

Tensor LlamaLinear::Forward(const Tensor&           input,  //
                            const LlamaDenseWeight& weight,
                            const Buffer_<int>&     indices,
                            const Buffer_<int>&     offsets,
                            std::optional<Tensor>   output)
{
    Tensor in = input.view({-1, input.shape(-1)});
    Tensor out;

    if (output) {
        out = output->view({-1, output->shape(-1)});
    }

    impl_->Forward(out, in, weight, indices, offsets);

    return out;
}

void LlamaLinear::set_measure(bool measure)
{
    impl_->dispatch_policy_ = measure ? gemm::DispatchPolicy::kMeasure : gemm::DispatchPolicy::kReuse;
}

int LlamaLinear::Export(std::ostream& os)
{
    if (os) {
        return impl_->gemm_.Export(os);
    }
    return 0;
}

int LlamaLinear::Import(std::istream& is)
{
    auto n_records = 0;
    if (is) {
        n_records = impl_->gemm_.Import(is);
    }
    if (n_records) {
        impl_->dispatch_policy_ = gemm::DispatchPolicy::kReuse;
    };
    return n_records;
}

std::vector<int> LlamaLinear::GetTuningSeq() const
{
    return impl_->gemm_.GetTuningSeq();
}

}  // namespace turbomind
