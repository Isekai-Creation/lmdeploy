// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/cuda_data_type.h"
#include "src/turbomind/kernels/gemm/context.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/dispatch_cache.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/tuner/params.h"
#include "src/turbomind/kernels/gemm/tuner/sampler.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/eagle_debug.h"
#include <cublas_v2.h>
#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

namespace turbomind::gemm {

void ExportDispatchCache(std::ostream& os, const std::vector<std::pair<GemmDesc, LaunchSpec>>& entries);

void ImportDispatchCache(std::istream&                                 is,
                         std::vector<std::pair<GemmDesc, LaunchSpec>>& entries,
                         const std::vector<std::unique_ptr<Kernel>>&   kernels);

namespace {

template<class Cmp>
std::vector<int> ArgSort(size_t size, const Cmp& cmp)
{
    std::vector<int> idxs(size);
    std::iota(idxs.begin(), idxs.end(), 0);
    std::stable_sort(idxs.begin(), idxs.end(), cmp);
    return idxs;
}

}  // namespace

namespace {

__device__ __forceinline__ float silu_f(float x)
{
    return x / (1.f + expf(-x));
}

__global__ void GatedSiluInterleavedF16Kernel(const half* __restrict__ tmp,
                                              half* __restrict__       out,
                                              int                      m,
                                              int                      n_full,
                                              int                      ld_tmp,
                                              int                      ld_out)
{
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) {
        return;
    }
    const int n_out = n_full / 2;
    if (col >= n_out) {
        return;
    }
    const int base = row * ld_tmp + (col * 2);
    const float gate = __half2float(tmp[base]);
    const float up   = __half2float(tmp[base + 1]);
    out[row * ld_out + col] = __float2half(silu_f(gate) * up);
}

#if ENABLE_BF16
__global__ void GatedSiluInterleavedBF16Kernel(const nv_bfloat16* __restrict__ tmp,
                                               nv_bfloat16* __restrict__       out,
                                               int                             m,
                                               int                             n_full,
                                               int                             ld_tmp,
                                               int                             ld_out)
{
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) {
        return;
    }
    const int n_out = n_full / 2;
    if (col >= n_out) {
        return;
    }
    const int base = row * ld_tmp + (col * 2);
    const float gate = __bfloat162float(tmp[base]);
    const float up   = __bfloat162float(tmp[base + 1]);
    out[row * ld_out + col] = __float2bfloat16(silu_f(gate) * up);
}
#endif

__global__ void GatedSiluInterleavedF32Kernel(const float* __restrict__ tmp,
                                              float* __restrict__       out,
                                              int                       m,
                                              int                       n_full,
                                              int                       ld_tmp,
                                              int                       ld_out)
{
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) {
        return;
    }
    const int n_out = n_full / 2;
    if (col >= n_out) {
        return;
    }
    const int base = row * ld_tmp + (col * 2);
    const float gate = tmp[base];
    const float up   = tmp[base + 1];
    out[row * ld_out + col] = silu_f(gate) * up;
}

int RunCublasGemmRowMajor(cublasHandle_t       handle,
                          const GemmDesc&      gdesc,
                          float                alpha,
                          const void*          A,
                          const MatrixLayout&  Adesc,
                          const void*          B,
                          const MatrixLayout&  Bdesc,
                          float                beta,
                          void*                D,
                          const MatrixLayout&  Ddesc)
{
    cublasOperation_t transa{};
    cublasOperation_t transb{};
    int               m{};
    int               n{};
    int               k{};
    const void*       A_ptr = A;
    const void*       B_ptr = B;
    int               lda{};
    int               ldb{};
    int               ldc = Ddesc.ld;

    if (gdesc.order_a == kRowMajor && gdesc.order_b == kRowMajor && gdesc.order_c == kRowMajor) {
        // Row-major GEMM: C[m,n] = A[m,k] @ B[k,n]
        // Equivalent column-major GEMM: C^T[n,m] = B^T[n,k] @ A^T[k,m]
        m     = Bdesc.cols;   // n
        n     = Adesc.rows;   // m
        k     = Adesc.cols;   // k
        A_ptr = B;
        B_ptr = A;
        lda   = Bdesc.ld;
        ldb   = Adesc.ld;

        TM_CHECK_EQ(Bdesc.rows, k);
        TM_CHECK_EQ(Ddesc.rows, Adesc.rows);
        TM_CHECK_EQ(Ddesc.cols, Bdesc.cols);

        transa = CUBLAS_OP_N;
        transb = CUBLAS_OP_N;
    }
    else {
        // Treat inputs as column-major with optional transpose
        transa = Adesc.order == kColMajor ? CUBLAS_OP_N : CUBLAS_OP_T;
        transb = Bdesc.order == kColMajor ? CUBLAS_OP_N : CUBLAS_OP_T;

        m   = Adesc.rows;
        n   = Bdesc.cols;
        k   = Adesc.cols;
        lda = Adesc.ld;
        ldb = Bdesc.ld;

        TM_CHECK_EQ(Bdesc.rows, k);
        TM_CHECK_EQ(Ddesc.rows, m);
        TM_CHECK_EQ(Ddesc.cols, n);
    }

    auto status = cublasGemmEx(handle,
                               transa,
                               transb,
                               m,
                               n,
                               k,
                               &alpha,
                               A_ptr,
                               to_cuda_dtype(Adesc.type),
                               lda,
                               B_ptr,
                               to_cuda_dtype(Bdesc.type),
                               ldb,
                               &beta,
                               D,
                               to_cuda_dtype(Ddesc.type),
                               ldc,
                               CUDA_R_32F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return status == CUBLAS_STATUS_SUCCESS ? 0 : int(status);
}

}  // namespace

struct Gemm::Impl {

    Impl():
        props_{GetCudaDeviceProps()},
        arch_{props_->major * 100 + props_->minor * 10},
        registry_{props_},
        cache_{registry_.kernels()}
    {
        if (auto str = std::getenv("TM_GEMM_TUNE")) {
            try {
                ParseTuningParams(tuning_, str);
            }
            catch (...) {
                std::cerr << "[Gemm2] Failed to parse `TM_GEMM_TUNE`, default value will be used.\n";
                tuning_ = {};
            }
        }
        if (std::getenv("TM_GEMM_WARN_CACHE_MISS")) {
            warn_cache_miss_ = true;
        }
        measurer_.emplace(CreateStoppingCriterion(tuning_.min_iter, tuning_.max_iter, tuning_.max_time));
    }

    // find launch spec in dispatch cache, dispatch by heuristic on cache miss
    LaunchSpec Dispatch(Context& ctx, DispatchPolicy policy, size_t barriers_size, size_t partials_size)
    {
        const auto& desc = ctx.desc();
        const bool  perf_mode =
            ::turbomind::isEnvVarEnabled("LMDEPLOY_EAGLE_PERF_MODE");
        const char* current_tag = ::turbomind::eagleCurrentGemmTag();
        const bool  is_eagle3_tag =
            current_tag && std::strncmp(current_tag, "EAGLE3_", 7) == 0;

        auto abort_if_eagle3_perf_generic = [&]() {
            if (perf_mode && is_eagle3_tag) {
                TM_LOG_ERROR(
                    "[Gemm2] Eagle3 GEMM dispatched without tuned cache entry in PERF_MODE "
                    "(tag=%s problem=%s); aborting.",
                    current_tag,
                    to_string(desc).c_str());
                std::abort();
            }
        };

        if (policy & DispatchPolicy::kReuse) {
            if (auto spec = cache_.LowerBound(desc)) {
                return *spec;
            }
            if (warn_cache_miss_) {
                std::cerr << "Failed to find a feasible kernel in the cache, will dispatch by heuristic: "
                          << to_string(ctx.desc()) << std::endl;
            }
            abort_if_eagle3_perf_generic();
        }

        if (auto spec = cache_.Find(desc)) {
            abort_if_eagle3_perf_generic();
            return *spec;
        }

        auto specs = Find(ctx, barriers_size, partials_size, 1);
        if (!specs.empty()) {
            cache_.Insert(desc, specs.front());
            abort_if_eagle3_perf_generic();
            return specs.front();
        }
        return {};
    }

    std::vector<LaunchSpec> Find(Context& ctx, size_t barrier_size, size_t partials_size, int top_k)
    {
        std::vector<Kernel*> feasible = ctx.Filter(registry_.kernels());

        std::vector<std::vector<LaunchSpec>> clusters;
        {
            std::vector<LaunchSpec> tmp;
            tmp.reserve(feasible.size());
            for (const auto& k : feasible) {
                LaunchSpec spec{k};
                tmp.push_back(spec);
            }
            clusters = Cluster(tmp, ClusteringParam{false, true});
        }
        std::vector<Kernel*> proxies;
        proxies.reserve(clusters.size());

        for (const auto& c : clusters) {
            proxies.push_back(c.front().kernel);
        }

        std::vector<std::pair<int, LaunchSpec>> specs;

        PopulateParam param{};
        param.max_splits    = tuning_.max_splits;
        param.max_waves     = tuning_.max_waves;
        param.swizzle       = tuning_.swizzle.at(0);
        param.barriers_size = barrier_size;
        param.partials_size = partials_size;

        for (int cluster_id = 0; cluster_id < (int)proxies.size(); ++cluster_id) {
            auto& kernel = *proxies[cluster_id];

            auto tmp = ctx.Populate(kernel, param);
            for (const auto& s : tmp) {
                specs.emplace_back(cluster_id, s);
            }
        }

        // std::cerr << "#kernel: " << kernels.size() << ", #cluster: " << clusters.size()
        //           << ", #metric: " << metrics.size() << "\n";

        int64_t mio_max = 0;
        int64_t mma_max = 0;
        for (const auto& [_, s] : specs) {
            auto& [mio, mma] = s.estimated;
            mio_max          = std::max(mio_max, mio);
            mma_max          = std::max(mma_max, mma);
        }
        std::vector<float> mio_ratio;
        std::vector<float> mma_ratio;
        std::vector<float> avg_ratio;
        for (const auto& [_, s] : specs) {
            auto& [mio, mma] = s.estimated;
            mio_ratio.push_back((float)mio / mio_max);
            mma_ratio.push_back((float)mma / mma_max);
            avg_ratio.push_back(.5 * (mio_ratio.back() + mma_ratio.back()));
        }
        auto idxs = ArgSort(specs.size(), [&](int i, int j) {  //
            return avg_ratio[i] < avg_ratio[j];
        });

        // for (const auto& i : idxs) {
        //     auto [cid, s, m] = metrics[i];
        //     std::cout << clusters[cid].front().kernel->name() << " s" << s << " " << avg_ratio[i] << " " <<
        //     mio_ratio[i]
        //               << " " << mma_ratio[i] << " " << m.mio_cost << " " << m.mma_cost << "\n";
        // }

        top_k = top_k > 0 ? std::min<int>(idxs.size(), top_k) : (int)idxs.size();
        std::vector<LaunchSpec> ret;
        ret.reserve(top_k);
        for (int i = 0; i < top_k; ++i) {
            const auto& [cluster_id, spec] = specs[idxs[i]];
            // Apply `splits` to all kernels in the cluster
            for (const auto& s : clusters[cluster_id]) {
                auto tmp   = spec;
                tmp.kernel = s.kernel;
                ret.push_back(tmp);
            }
        }

        return ret;
    }

    template<class LaunchFunc>
    int Measure(
        Context& ctx, size_t barriers_size, size_t partials_size, int top_k, LaunchFunc launch_func, cudaStream_t st)
    {
        // Early exit on exact match
        if (cache_.Find(ctx.desc())) {
            return 0;
        }
        // std::cerr << "GEMM: " << desc.m << "x" << desc.n << "x" << desc.k << "\n";

        const auto tmp = Find(ctx, barriers_size, partials_size, tuning_.top_k);

        std::vector<LaunchSpec> specs;
        for (const auto& spec : tmp) {
            // populate swizzle parameters
            const auto swis = ctx.Swizzle(spec, tuning_.swizzle);
            specs.insert(specs.end(), swis.begin(), swis.end());
        }

        specs = Sampler{*measurer_, tuning_.clusters}.Run(specs, launch_func, st);

        // for (const auto& s : specs) {
        //     std::cout << s.kernel->name()          //
        //               << " swizzle=" << s.swizzle  //
        //               << ", splits=" << s.splits   //
        //               << ", measured=" << s.measured << "ms\n";
        //     break;
        // }

        if (!specs.empty()) {
            cache_.Insert(ctx.desc(), specs.front());
        }
        else {
            std::cerr << "No valid kernel found for the problem\n";
            return -1;
        }

        return 0;
    }

    /// TODO: move to cuda utils
    static std::unique_ptr<cudaDeviceProp> GetCudaDeviceProps()
    {
        auto props     = std::make_unique<cudaDeviceProp>();
        int  device_id = -1;
        cudaGetDevice(&device_id);
        cudaGetDeviceProperties(props.get(), device_id);
        return props;
    }

    std::shared_ptr<cudaDeviceProp> props_;

    int arch_;

    Registry registry_;

    TuningParams tuning_;

    bool warn_cache_miss_{};

    std::optional<Measurer> measurer_;

    DispatchCache cache_;
};

// implementation of GEMM interfaces

Gemm::Gemm(): impl_{new Impl{}} {}

Gemm::~Gemm() = default;

int Gemm::Run(const Operation&    operation,
              float               alpha,
              const void*         A,
              const MatrixLayout& Adesc,
              const void*         U,
              const MatrixLayout& Udesc,
              const void*         B,
              const MatrixLayout& Bdesc,
              const void*         V,
              const MatrixLayout& Vdesc,
              float               beta,
              const void*         C,
              const MatrixLayout& Cdesc,
              void*               D,
              const MatrixLayout& Ddesc,
              const Workspace&    workspace,
              cudaStream_t        stream)
{

    Context context{*impl_->props_};

    const auto desc = context.Init(operation, Adesc, Udesc, Bdesc, Vdesc, Cdesc, Ddesc);

    if (!desc) {
        fprintf(stderr, "invalid argument.\n");
        TM_LOG_ERROR(
            "[Gemm] invalid argument for problem: A=(%d,%d) B=(%d,%d) C=(%d,%d) D=(%d,%d)",
            Adesc.rows,
            Adesc.cols,
            Bdesc.rows,
            Bdesc.cols,
            Cdesc.rows,
            Cdesc.cols,
            Ddesc.rows,
            Ddesc.cols);
        // Signal failure to the caller so higher-level fallbacks
        // (e.g. LlamaLinear's naive GEMM) can take over instead of
        // aborting the process.
        return 1;
    }

    const auto launch = [=](LaunchSpec spec, cudaStream_t st) {
        auto _workspace = workspace;
        return spec.kernel->Launch(operation,
                                   alpha,
                                   A,
                                   Adesc,
                                   U,
                                   Udesc,
                                   B,
                                   Bdesc,
                                   V,
                                   Vdesc,
                                   beta,
                                   C,
                                   Cdesc,
                                   D,
                                   Ddesc,
                                   spec.swizzle,
                                   spec.splits,
                                   _workspace,
                                   st);
    };

#if 0
    if (operation.reserved) {
        auto specs = impl_->Find(context, workspace.barriers_size, workspace.partials_size, 0);
        auto cases = (std::vector<std::function<LaunchSpec()>>*)operation.reserved;
        for (const auto& spec : specs) {
            cases->push_back([=] {
                launch(spec, stream);
                return spec;
            });
        }
        return -1;
    }
#endif

    LaunchSpec spec{};

    if (operation.dispatch & DispatchPolicy::kMeasure) {
        impl_->Measure(context, workspace.barriers_size, workspace.partials_size, 1, launch, stream);
    }

    spec = impl_->Dispatch(context, operation.dispatch, workspace.barriers_size, workspace.partials_size);

    if (spec.kernel) {
        // std::cout << "[Gemm] dispatch: " << spec.kernel->name()  //
        //           << " split_k=" << spec.splits                  //
        //           << " swizzle=" << spec.swizzle << std::endl;
        return launch(spec, stream);
    }

    // Fallback: when no fused kernel is available (e.g. new GPU arch or
    // unsupported dtype/layout), try a plain cuBLAS GEMM for simple
    // dense cases instead of hard‑failing. This mirrors
    // CublasKernel::is_feasible in cublas.cu.
    const auto& gdesc = context.desc();
    constexpr std::tuple flat3{Striding::kFlat, Striding::kFlat, Striding::kFlat};

    const bool gated_silu = (gdesc.epilogue == Epilogue::kGatedSilu);

    bool cublas_ok = true;
    if (std::tie(gdesc.striding_a, gdesc.striding_b, gdesc.striding_c) != flat3) {
        cublas_ok = false;
    }
    // cuBLAS fallback supports either:
    // - Epilogue::kNone (plain GEMM), or
    // - Epilogue::kGatedSilu via a 2-step GEMM + activation fallback.
    if (gdesc.epilogue != Epilogue::kNone && !gated_silu) {
        cublas_ok = false;
    }
    if (gdesc.num > 1) {
        cublas_ok = false;
    }
    if (gdesc.quant_a || gdesc.quant_b) {
        cublas_ok = false;
    }
    // Allow grouped descriptors to use the cuBLAS fallback when they
    // effectively represent a single GEMM (num == 1). For true grouped
    // problems (num > 1), keep using the fused kernels only.
    if (gdesc.group_axis >= 0 && gdesc.num > 1) {
        cublas_ok = false;
    }
    if (gdesc.type_a != kHalf && gdesc.type_a != kBfloat16 && gdesc.type_a != kFloat) {
        cublas_ok = false;
    }
    if (gdesc.type_b != gdesc.type_a) {
        cublas_ok = false;
    }
    if (gdesc.type_c != gdesc.type_a && gdesc.type_c != kFloat) {
        cublas_ok = false;
    }

    if (cublas_ok) {
        static cublasHandle_t handle = nullptr;
        static cudaStream_t   handle_stream{};
        if (!handle) {
            TM_CHECK(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS);
        }
        if (handle_stream != stream) {
            TM_CHECK(cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS);
            handle_stream = stream;
        }

        TM_CHECK(C == nullptr || C == D);

        if (!gated_silu) {
            const int status = RunCublasGemmRowMajor(handle, gdesc, alpha, A, Adesc, B, Bdesc, beta, D, Ddesc);
            if (status == 0) {
                return 0;
            }
            TM_LOG_ERROR(
                "[Gemm] cuBLAS fallback failed for problem: %s (status=%d)",
                to_string(context.desc()).c_str(),
                status);
            return 1;
        }

        // Epilogue::kGatedSilu fallback:
        // 1) Compute tmp = A @ B as a plain GEMM producing the full (2*hidden)
        //    columns, then
        // 2) Apply gated SiLU over interleaved pairs: out[j] = silu(tmp[2j]) * tmp[2j+1].
        //
        // This is slower than a fused kernel, but is vastly faster and safer than
        // falling back to naive GEMM (and preserves correctness for Eagle3 draft FFNs).
        const int m_rows  = Ddesc.rows;
        const int n_full  = Ddesc.cols;
        if ((n_full % 2) != 0) {
            TM_LOG_ERROR("[Gemm] GatedSiLU fallback requires even N, got N=%d", n_full);
            return 1;
        }
        const int n_out = n_full / 2;

        // tmp is row-major [m_rows, n_full]
        MatrixLayout tmp_desc{Ddesc.type, Ddesc.order, m_rows, n_full, n_full};
        void*        tmp_ptr = nullptr;
        const size_t tmp_bytes = static_cast<size_t>(byte_size(tmp_desc));
        check_cuda_error(cudaMallocAsync(&tmp_ptr, tmp_bytes, stream));

        const int status = RunCublasGemmRowMajor(handle, gdesc, alpha, A, Adesc, B, Bdesc, /*beta=*/0.f, tmp_ptr, tmp_desc);
        if (status != 0) {
            cudaFreeAsync(tmp_ptr, stream);
            TM_LOG_ERROR(
                "[Gemm] cuBLAS(GatedSiLU tmp) failed for problem: %s (status=%d)",
                to_string(context.desc()).c_str(),
                status);
            return 1;
        }

        // Apply gated SiLU into D (row-major [m_rows, n_out], ld = Ddesc.ld).
        const dim3 block(256);
        const dim3 grid((n_out + block.x - 1) / block.x, m_rows);
        if (Ddesc.type == kHalf) {
            GatedSiluInterleavedF16Kernel<<<grid, block, 0, stream>>>(
                static_cast<const half*>(tmp_ptr),
                static_cast<half*>(D),
                m_rows,
                n_full,
                /*ld_tmp=*/n_full,
                /*ld_out=*/Ddesc.ld);
        }
#if ENABLE_BF16
        else if (Ddesc.type == kBfloat16) {
            GatedSiluInterleavedBF16Kernel<<<grid, block, 0, stream>>>(
                static_cast<const nv_bfloat16*>(tmp_ptr),
                static_cast<nv_bfloat16*>(D),
                m_rows,
                n_full,
                /*ld_tmp=*/n_full,
                /*ld_out=*/Ddesc.ld);
        }
#endif
        else if (Ddesc.type == kFloat) {
            GatedSiluInterleavedF32Kernel<<<grid, block, 0, stream>>>(
                static_cast<const float*>(tmp_ptr),
                static_cast<float*>(D),
                m_rows,
                n_full,
                /*ld_tmp=*/n_full,
                /*ld_out=*/Ddesc.ld);
        }
        else {
            cudaFreeAsync(tmp_ptr, stream);
            TM_LOG_ERROR("[Gemm] GatedSiLU fallback unsupported dtype=%d", int(Ddesc.type));
            return 1;
        }
        sync_check_cuda_error();
        cudaFreeAsync(tmp_ptr, stream);
        return 0;
    }

    TM_LOG_ERROR(
        "[Gemm] No feasible kernel found for the problem: %s", to_string(context.desc()).c_str());
    return 1;
}

int Gemm::Export(std::ostream& os)
{
    return impl_->cache_.Export(os);
}

int Gemm::Import(std::istream& is)
{
    return impl_->cache_.Import(is);
}

std::vector<int> Gemm::GetTuningSeq() const
{
    return impl_->tuning_.seq;
}

}  // namespace turbomind::gemm
