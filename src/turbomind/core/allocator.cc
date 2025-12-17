
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <sstream>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/utils/eagle_debug.h"
#include "src/turbomind/utils/progress_logger.h"

namespace turbomind::core {
namespace {

void log_allocator_oom_event(const char* tag, ssize_t size)
{
    size_t     free_bytes  = 0;
    size_t     total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        free_bytes  = 0;
        total_bytes = 0;
    }

    ProgressEvent evt{ProgressStage::kError};
    evt.pct = 100;
    std::ostringstream oss;
    oss << "cuda_oom tag=" << (tag ? tag : "unknown") << " bytes=" << size
        << " free=" << free_bytes << " total=" << total_bytes;
    evt.msg = oss.str();
    ProgressLogger::Log(evt);
}

}  // namespace

AllocatorImpl::~AllocatorImpl() = default;

Stream AllocatorImpl::stream() const noexcept
{
    return Stream{};
}

class CudaMemPoolAllocator: public AllocatorImpl {
public:
    CudaMemPoolAllocator(Stream stream, bool use_default_pool):
        pool_{}, stream_{stream}, device_{kDEVICE}, use_default_pool_{use_default_pool}
    {
        // For DriftEngine v1 and the modern TurboMind path we no longer rely
        // on device memory pools here. Instead, we fall back to the same
        // plain cudaMalloc / cudaFree behaviour as CudaAllocator to avoid
        // allocator‑pool fragmentation bugs. The stream is still tracked so
        // callers can migrate back to async allocators in the future if
        // desired.
        check_cuda_error(cudaGetDevice(&device_.id));
    }

    ~CudaMemPoolAllocator() override
    {
        pool_ = {};
    }

    void* allocate(ssize_t size) override
    {
        // Guard against callers passing negative or zero sizes. A negative
        // `ssize_t` would be converted to a huge `size_t` inside the CUDA
        // allocator APIs, which in turn looks like a multi‑TB allocation
        // request and hides the real bug. Fail fast here instead so we can
        // pinpoint the bad caller.
        if (size <= 0) {
            TM_LOG_ERROR("[CudaMemPoolAllocator] Invalid allocation size=%zd; "
                         "refusing to call device allocator with a non‑positive size.",
                         size);
            abort();
        }

        // Capture a lightweight memory snapshot before we touch the device
        // allocator so OOMs are self‑describing in logs.
        size_t     free_bytes  = 0;
        size_t     total_bytes = 0;
        cudaError_t info_err   = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (info_err == cudaSuccess) {
            TM_LOG_INFO(
                "[Allocator][CudaMemPool] about_to_alloc bytes=%zd free=%zu total=%zu",
                static_cast<ssize_t>(size),
                free_bytes,
                total_bytes);
        }
        else {
            TM_LOG_WARNING("[Allocator][CudaMemPool] cudaMemGetInfo failed before alloc: %s",
                           cudaGetErrorString(info_err));
        }

        turbomind::set_last_cuda_alloc_bytes(static_cast<size_t>(size));
        void* ptr{};
        // Use plain cudaMalloc instead of cudaMallocFromPoolAsync to avoid
        // interacting with device memory pools that can introduce hard‑to‑
        // debug fragmentation or threshold issues.
        cudaError_t alloc_err = cudaMalloc(&ptr, size);
        if (alloc_err != cudaSuccess) {
            if (alloc_err == cudaErrorMemoryAllocation) {
                log_allocator_oom_event("CudaMemPool", size);
            }
            check_cuda_error(alloc_err);
        }

        // Allocation debug logging is gated separately from general
        // EAGLE debug to avoid flooding logs during normal runs.
        if (turbomind::isEagleDebugEnabled() && turbomind::isEnvVarEnabled("LMDEPLOY_EAGLE_ALLOC_DEBUG")) {
            TM_LOG_WARNING("[EAGLE][AllocDBG:pool] size=%zd ptr=%p", (ssize_t)size, ptr);
        }
        return ptr;
    }

    void deallocate(void* p, ssize_t) override
    {
        check_cuda_error(cudaFree(p));
    }

    Device device() const noexcept override
    {
        return device_;
    }

    Stream stream() const noexcept override
    {
        return stream_;
    }

    void trim(size_t bytes_to_keep)
    {
        (void)bytes_to_keep;
        // No-op for the plain cudaMalloc/cudaFree path.
    }

private:
    cudaMemPool_t pool_;
    Stream        stream_;
    Device        device_;
    bool          use_default_pool_;
};

class CudaAllocator: public AllocatorImpl {
public:
    void* allocate(ssize_t size) override
    {
        turbomind::set_last_cuda_alloc_bytes(static_cast<size_t>(size));
        void* ptr{};
        cudaError_t alloc_err = cudaMalloc(&ptr, size);
        if (alloc_err != cudaSuccess) {
            if (alloc_err == cudaErrorMemoryAllocation) {
                log_allocator_oom_event("CudaAllocator", size);
            }
            check_cuda_error(alloc_err);
        }
        if (turbomind::isEagleDebugEnabled() && turbomind::isEnvVarEnabled("LMDEPLOY_EAGLE_ALLOC_DEBUG")) {
            TM_LOG_WARNING("[EAGLE][AllocDBG:cuda] size=%zd ptr=%p", (ssize_t)size, ptr);
        }
        return ptr;
    }

    void deallocate(void* p, ssize_t) override
    {
        check_cuda_error(cudaFree(p));
    }

    Device device() const noexcept override
    {
        return kDEVICE;
    }
};

class CudaHostAllocator: public AllocatorImpl {
public:
    void* allocate(ssize_t size) override
    {
        void* ptr{};
        check_cuda_error(cudaHostAlloc(&ptr, size, cudaHostAllocDefault));
        return ptr;
    }

    void deallocate(void* p, ssize_t) override
    {
        check_cuda_error(cudaFreeHost(p));
    }

    Device device() const noexcept override
    {
        return kCPUpinned;
    }
};

class HostAllocator: public AllocatorImpl {
public:
    void* allocate(ssize_t size) override
    {
        return ::operator new(size);
    }

    void deallocate(void* p, ssize_t) override
    {
        ::operator delete(p);
    }

    Device device() const noexcept override
    {
        return kCPU;
    }
};

Allocator::Allocator(DeviceType type)
{
    impl_ = [&]() -> shared_ptr<AllocatorImpl> {
        switch (type) {
            case kCPU:
                return std::make_shared<HostAllocator>();
            case kDEVICE:
                return std::make_shared<CudaAllocator>();
            case kCPUpinned:
                return std::make_shared<CudaHostAllocator>();
        }
        return {};
    }();
    TM_CHECK_NOTNULL(impl_);
}

Allocator::Allocator(Stream stream, bool use_default_pool)
{
    impl_ = std::make_shared<CudaMemPoolAllocator>(std::move(stream), use_default_pool);
}

}  // namespace turbomind::core
