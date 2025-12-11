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
        if (policy & DispatchPolicy::kReuse) {
            if (auto spec = cache_.LowerBound(desc)) {
                return *spec;
            }
            if (warn_cache_miss_) {
                std::cerr << "Failed to find a feasible kernel in the cache, will dispatch by heuristic: "
                          << to_string(ctx.desc()) << std::endl;
            }
        }

        if (auto spec = cache_.Find(desc)) {
            return *spec;
        }

        auto specs = Find(ctx, barriers_size, partials_size, 1);
        if (!specs.empty()) {
            cache_.Insert(desc, specs.front());
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

    bool cublas_ok = true;
    if (std::tie(gdesc.striding_a, gdesc.striding_b, gdesc.striding_c) != flat3) {
        cublas_ok = false;
    }
    if (std::tie(gdesc.pack_a, gdesc.pack_b, gdesc.pack_u, gdesc.pack_v) != std::tuple{0, 0, 0, 0}) {
        cublas_ok = false;
    }
    if (gdesc.epilogue != Epilogue::kNone) {
        cublas_ok = false;
    }
    if (gdesc.num > 1) {
        cublas_ok = false;
    }
    if (gdesc.quant_a || gdesc.quant_b) {
        cublas_ok = false;
    }
    if (gdesc.group_axis >= 0) {
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

        cublasOperation_t transa{};
        cublasOperation_t transb{};
        int               m{};
        int               n{};
        int               k{};
        const void*       A_ptr = A;
        const void*       B_ptr = B;
        int               lda{};
        int               ldb{};
        int               ldc   = Ddesc.ld;

        if (gdesc.order_a == kRowMajor && gdesc.order_b == kRowMajor && gdesc.order_c == kRowMajor) {
            // Row-major GEMM: C[m,n] (row-major) = A[m,k] (row-major) * B[k,n] (row-major)
            // is equivalent to column-major: C^T[n,m] = B^T[n,k] * A^T[k,m].
            m     = Bdesc.cols;   // n
            n     = Adesc.rows;   // m
            k     = Adesc.cols;   // k
            A_ptr = B;
            B_ptr = A;
            lda   = Bdesc.ld;     // leading dim for B^T interpreted as [n,k] col-major
            ldb   = Adesc.ld;     // leading dim for A^T interpreted as [k,m] col-major

            TM_CHECK_EQ(Bdesc.rows, k);
            TM_CHECK_EQ(Ddesc.rows, Adesc.rows);
            TM_CHECK_EQ(Ddesc.cols, Bdesc.cols);

            transa = CUBLAS_OP_N;
            transb = CUBLAS_OP_N;
        }
        else {
            // Fallback to treating A/B as column-major with optional transpose
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

        TM_CHECK(C == nullptr || C == D);

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

        if (status == CUBLAS_STATUS_SUCCESS) {
            return 0;
        }

        TM_LOG_ERROR(
            "[Gemm] cuBLAS fallback failed for problem: %s (status=%d)",
            to_string(context.desc()).c_str(),
            int(status));
        return 1;
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
