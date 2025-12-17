// Copyright (c) OpenMMLab. All rights reserved.

#include "attention.h"
#include "block.h"
#include "decoding.h"
#include "kv_cache_utils_v2.h"
#include "fp4_kv_utils.h"
#include "src/turbomind/kernels/attention/attention_params.h"
#include "src/turbomind/kernels/attention/fp4_kv_utils.h"
#include "src/turbomind/kernels/attention/reference.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "test_utils.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>
#include <utility>

using namespace turbomind;

// Forward declaration for FP4 MXFP4 probe harness defined in fp4_kv_probe.cu.
// struct Fp4KvProbeResult;
// cudaError_t fp4_kv_probe_host(Fp4KvProbeResult& out,
//                               char**            d_blocks,
//                               char**            d_scale_blocks,
//                               int               layer_id,
//                               int               head_idx,
//                               int               head_num,
//                               int               block_len,
//                               int               head_dim,
//                               int               local_ti,
//                               cudaStream_t      stream);

// [b, h, s, d] : current -> stride_h=s, stride_s=1, stride_b=hs
// [cu_q, h, d] : qkvgemm -> stride_h=1, stride_s=h, stride_b=0
// [h, cu_s, d] : prefill -> stride_h=s, stride_s=1, stride_b=0

template<class T, class Tkv>
struct Config {
    int head_dim_;
    int head_num_;
    int block_len_;

    TM_HOST_DEVICE constexpr int t_bits() const
    {
        if constexpr (std::is_same_v<T, Tkv>) {
            return 0;
        }
        else {
            return bitsof<T>;
        }
    }

    TM_HOST_DEVICE constexpr int q_bits() const
    {
        return bitsof<Tkv>;
    }

    TM_HOST_DEVICE constexpr int head_dim() const
    {
        return head_dim_;
    }

    TM_HOST_DEVICE int head_num() const
    {
        return head_num_;
    }

    TM_HOST_DEVICE constexpr int block_len() const
    {
        return block_len_;
    }
};

// [S/S, H, S, D] <-> [S/b, H, b, D]
template<class Tkv, class T>
void TestBlocks(const thrust::universal_vector<T>& k_cache,        // [B, H, S, D]
                const thrust::universal_vector<T>& v_cache,        // [B, H, S, D]
                thrust::universal_vector<char>&    blocks,         // block data
                thrust::universal_vector<char*>&   k_ptrs,         // block ptrs
                thrust::universal_vector<int>&     cu_block_cnts,  // cumulative block counts
                const size_t                       head_num,
                const size_t                       head_dim,
                const size_t                       block_seq_len,
                const size_t                       batch_size,
                const int                          rope_dim,
                int                                quant_policy,
                AttentionParams<T>&                params,
                const thrust::universal_vector<int>& cu_kv_lens)
{
    const size_t seq_len  = k_cache.size() / (head_dim * head_num * batch_size);
    const size_t n_blocks = (seq_len + block_seq_len - 1) / block_seq_len;

    Config<T, Tkv> config{(int)head_dim, (int)head_num, (int)block_seq_len};
    block::Layout  layout{config};

    dump(layout);

    const size_t kHSD = head_num * seq_len * head_dim;

    std::cout << "batch_size = " << batch_size << ", seq_len = " << seq_len << ", block_size = " << block_seq_len
              << ", block_num = " << n_blocks << "\n";

    thrust::universal_vector<T> kv_cache(k_cache.size() * 2);  // [B, 2, H, S, D]

    {  // interleave K/V
        auto k_src = k_cache.begin();
        auto v_src = v_cache.begin();
        auto dst   = kv_cache.begin();
        for (size_t i = 0; i < batch_size; ++i) {
            dst = thrust::copy_n(k_src, kHSD, dst);
            dst = thrust::copy_n(v_src, kHSD, dst);
            k_src += kHSD;
            v_src += kHSD;
        }
    }

    // const int kHsD = head_num * block_seq_len * head_dim;

    // [B, S/s, 2, H, s, D]
    // blocks.resize(batch_size * n_blocks * 2 * kHsD);
    blocks.resize(batch_size * n_blocks * layout.block_size(1));
    thrust::fill(blocks.begin(), blocks.end(), NAN);
    k_ptrs.resize(batch_size * n_blocks + 1);  // +1 padding

    std::vector<size_t> idxs(batch_size * n_blocks);
    std::iota(idxs.begin(), idxs.end(), 0);

    std::random_device rd;
    std::mt19937       g(rd());
    std::shuffle(idxs.begin(), idxs.end(), g);

    for (size_t i = 0; i < idxs.size(); ++i) {
        // k_ptrs[i] = blocks.data().get() + idxs[i] * 2 * kHsD;
        k_ptrs[i] = blocks.data().get() + idxs[i] * layout.block_size(1);
    }

    thrust::universal_vector<int> seq_lens(batch_size);
    thrust::universal_vector<int> cu_seq_lens(batch_size + 1);
    thrust::fill(seq_lens.begin(), seq_lens.end(), seq_len);
    for (size_t i = 0; i <= batch_size; ++i) {
        cu_seq_lens[i] = i * seq_len;
    }

    std::vector<int> n_blocks_vec(batch_size + 1, n_blocks);
    cu_block_cnts.resize(batch_size + 1);
    std::exclusive_scan(n_blocks_vec.begin(), n_blocks_vec.end(), cu_block_cnts.begin(), 0);

    cudaDeviceSynchronize();

    // [B, 2H, S, D] -> [B, S/s] x [2H, s, D]
    for (int i = 0; i < 1; ++i) {
        // (B, 2, H, S, D) -> blocks
        invokeProcessKV_v2(k_ptrs.data().get(),
                           nullptr,
                           kv_cache.data().get(),
                           kv_cache.data().get() + head_num * seq_len * head_dim,
                           (T*)nullptr,
                           (T*)nullptr,
                           cu_seq_lens.data().get(),
                           cu_seq_lens.data().get(),
                           cu_block_cnts.data().get(),
                           RopeKernelParam{},
                           2 * head_num * seq_len,
                           0,
                           seq_len,
                           1,
                           block_seq_len,
                           0,  // layer_id
                           0,  // cp_rank
                           1,  // cp_size
                           seq_len,
                           head_num,
                           head_dim,
                           batch_size,
                           quant_policy,
                           getSMVersion());
    }

    thrust::universal_vector<T> kv_cache_2(kv_cache.size());

    // round trip test
    for (int i = 0; i < 1; ++i) {
        // kv_cache_2 is [B, 2, H, S, D]
        invokeFlattenKV_v2_(params, cu_kv_lens[batch_size]);
    }

    cudaDeviceSynchronize();

    if (0) {
        std::cout << ">>> Compare\n";
        Compare(
            kv_cache_2.data().get(), kv_cache.data().get(), head_dim, head_dim, batch_size * 2 * head_num * seq_len, 0);
        std::cout << "<<< Compare\n";
    }
}

double get_memory_bandwidth()  // -> GB/s
{
    int clock_rate_khz{};
    int bus_width_bits{};
    cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, 0);
    return 2. * (double)clock_rate_khz / 1e6 * (double)bus_width_bits / 8.;
}

#define KV_INT8 0

#define KV_INT4 0

#define DECODING 0

#define SINK 5

template<class T>
int test_attention()
{
    AttentionParams<T> params{};

    constexpr size_t kHeadDim    = 128;
    constexpr int    kWindowSize = 128 << 20;

#if DECODING
    // constexpr size_t kHeadNum   = 32;
    // constexpr size_t kBatchSize = 64;
    constexpr size_t kHeadNum   = 64;
    constexpr size_t KvHeadNum  = kHeadNum / 8;
    constexpr size_t kBatchSize = 256;
    constexpr size_t kInputLen  = 1;

    constexpr size_t kSequenceLen = 1000;
    // constexpr size_t kSequenceLen = 4095;
    // constexpr size_t kSequenceLen = 511;
    // constexpr size_t kSequenceLen = 2047;
    // constexpr size_t kSequenceLen = 4095;
    // constexpr size_t kSequenceLen = 8 * 1024 - 1;
    // constexpr size_t kSequenceLen = 32767;
    // constexpr size_t kSequenceLen = 65535;
    // constexpr size_t kSequenceLen = 131071;
    // constexpr size_t kSequenceLen = 200000;
    // constexpr size_t kSequenceLen = 262143;
    // constexpr size_t kSequenceLen = (1 << 20) - 1;  // 1M
    // constexpr size_t kSequenceLen = (1 << 22) - 1;  // 4M
    // constexpr size_t kSequenceLen = (1 << 24) - 1;  // 16M
    // constexpr int kSequenceLen = 2047;
    constexpr int kBlockSz   = 64;
    constexpr int kMaxSplitK = 128;
#else

    // append
    // constexpr size_t kHeadNum     = 32;
    // constexpr size_t KvHeadNum    = kHeadNum;
    // constexpr size_t kBatchSize   = 1;
    // constexpr size_t kInputLen    = 128;
    // constexpr size_t kSequenceLen = 65536;
    // constexpr int    kMaxSplitK   = 128;

    // constexpr size_t kHeadNum     = 1;
    // constexpr size_t KvHeadNum    = kHeadNum;
    // constexpr size_t kBatchSize   = 1;
    // constexpr size_t kInputLen    = 64;
    // constexpr size_t kSequenceLen = 65536;
    // constexpr int    kMaxSplitK   = 1;

    // prefill
    constexpr size_t kHeadNum     = 16;
    constexpr size_t KvHeadNum    = kHeadNum / 8;
    constexpr size_t kBatchSize   = 2;
    constexpr size_t kInputLen    = 8192;
    constexpr size_t kSequenceLen = 0;
    constexpr int    kMaxSplitK   = 1;

    constexpr int kBlockSz     = 64;

#endif

#if KV_INT8
    using Tkv                  = uint8_t;
    constexpr int kQuantPolicy = QuantPolicy::kCacheKVInt8;
#elif KV_INT4
    using Tkv                  = uint4_t;
    constexpr int kQuantPolicy = QuantPolicy::kCacheKVInt4;
#else
    using Tkv                  = T;
    constexpr int kQuantPolicy = 0;
#endif

    static_assert(KvHeadNum > 0);

    constexpr size_t kContextLen = kSequenceLen + kInputLen;
    constexpr size_t kTokenNum   = kBatchSize * kInputLen;
    constexpr int    kTestIter   = 10;

    constexpr float kRoPEBase = 10000.f;
    constexpr int   kRoPEDim  = kHeadDim / 2;
    constexpr int   kDump     = 0;

    RNG rng{};

    thrust::universal_vector<T> k_cache(kBatchSize * KvHeadNum * kContextLen * kHeadDim);
    thrust::universal_vector<T> v_cache(kBatchSize * KvHeadNum * kContextLen * kHeadDim);

    // flattened float point KV cache
    thrust::device_vector<T> kv_cache(KvHeadNum * 2 * (kBatchSize * kContextLen + MAX_CTA_S) * kHeadDim);

    thrust::universal_vector<T> qkv(kBatchSize * kInputLen * (kHeadNum + KvHeadNum * 2) * kHeadDim);
    thrust::universal_vector<T> output(kBatchSize * kInputLen * kHeadNum * kHeadDim);

    thrust::universal_vector<bool>  finished(kBatchSize);
    thrust::universal_vector<int>   sequence_length(kBatchSize);
    thrust::universal_vector<int>   input_length(kBatchSize);
    thrust::universal_vector<int>   context_length(kBatchSize);
    thrust::universal_vector<float> rope_base(kBatchSize);
    thrust::universal_vector<int>   cu_seqlens(kBatchSize + 1);
    thrust::universal_vector<int>   cu_kv_lens(kBatchSize + 1);

    thrust::device_vector<float> partial_ML(kTokenNum * kHeadNum * kMaxSplitK * 2);
    thrust::device_vector<float> partial_O(kTokenNum * kHeadNum * kMaxSplitK * kHeadDim);
    thrust::device_vector<int>   split_cnt(kTokenNum);

    thrust::universal_vector<float> qk_buf((size_t)kDump * kBatchSize * kHeadNum * kInputLen * kContextLen);
    thrust::universal_vector<T>     pr_buf((size_t)kDump * kBatchSize * kHeadNum * kInputLen * kContextLen);

    thrust::universal_vector<T> sinks(kHeadNum);

    rng.GenerateNormal(qkv.data().get(), qkv.size(), 1.f, 0.f);

    rng.GenerateNormal(k_cache.data().get(), kBatchSize * KvHeadNum * kContextLen * kHeadDim);
    rng.GenerateNormal(v_cache.data().get(), kBatchSize * KvHeadNum * kContextLen * kHeadDim);

    if (SINK) {
        rng.GenerateUniform(sinks.data().get(), sinks.size(), 2 * SINK, -SINK);
    }

    if (0) {
        // Set input range to zero
        // (BH, SD)
        cudaMemset2DAsync(k_cache.data().get() + kSequenceLen * kHeadDim,
                          sizeof(T) * kContextLen * kHeadDim,
                          0,
                          sizeof(T) * kInputLen * kHeadDim,
                          kBatchSize * KvHeadNum);
        cudaMemset2DAsync(v_cache.data().get() + kSequenceLen * kHeadDim,
                          sizeof(T) * kContextLen * kHeadDim,
                          0,
                          sizeof(T) * kInputLen * kHeadDim,
                          kBatchSize * KvHeadNum);
    }

    invokeApplyRotaryEmbedding(k_cache.data().get(), kContextLen, KvHeadNum, kHeadDim, kRoPEBase, kRoPEDim, kBatchSize);

    thrust::universal_vector<T> k_cache_ref = k_cache;
    thrust::universal_vector<T> v_cache_ref = v_cache;

    thrust::universal_vector<char>  blocks;
    thrust::universal_vector<char*> k_ptrs;
    thrust::universal_vector<int>   cu_block_cnts;

    TestBlocks<Tkv>(k_cache,
                    v_cache,
                    blocks,
                    k_ptrs,
                    cu_block_cnts,
                    KvHeadNum,
                    kHeadDim,
                    kBlockSz,
                    kBatchSize,
                    kRoPEDim,
                    kQuantPolicy,
                    params,
                    cu_kv_lens);

    thrust::universal_vector<T>     output_ref = output;
    thrust::universal_vector<void*> k_cache_ref_ptrs(kBatchSize);
    thrust::universal_vector<void*> v_cache_ref_ptrs(kBatchSize);

    thrust::universal_vector<T> bias_QKV(kHeadNum * kHeadDim + 2 * KvHeadNum * kHeadDim);

    rng.GenerateNormal(bias_QKV.data().get(), bias_QKV.size(), 0.1f, 0.f);

    cudaDeviceSynchronize();

    for (size_t i = 0; i <= kBatchSize; ++i) {
        cu_seqlens[i] = i * kInputLen;
        cu_kv_lens[i] = i * kContextLen;
    }

    for (size_t i = 0; i < kBatchSize; ++i) {
        input_length[i]     = kInputLen;
        sequence_length[i]  = kSequenceLen;
        context_length[i]   = kContextLen;
        k_cache_ref_ptrs[i] = k_cache_ref.data().get() + i * k_cache_ref.size() / kBatchSize;
        v_cache_ref_ptrs[i] = v_cache_ref.data().get() + i * v_cache_ref.size() / kBatchSize;
        rope_base[i]        = kRoPEBase;
    }

    // getchar();

    params.out = output_ref.data().get();
    params.q   = qkv.data().get();
    params.k   = params.q + kHeadNum * kHeadDim;
    params.v   = params.k + KvHeadNum * kHeadDim;

    params.q_bias = bias_QKV.data().get();
    params.k_bias = params.q_bias + kHeadNum * kHeadDim;
    params.v_bias = params.k_bias + KvHeadNum * kHeadDim;

    params.stride = (kHeadNum + 2 * KvHeadNum) * kHeadDim;

    params.token_num  = kTokenNum;
    params.batch_size = kBatchSize;
    params.max_q_len  = kInputLen;
    params.max_k_len  = kContextLen;

    params.block_iter_params = BlockIteratorParams{k_ptrs.data().get(),  //
                                                   nullptr,
                                                   cu_block_cnts.data().get(),
                                                   0,
                                                   kBlockSz};

    params.linear_iter_params = LinearIteratorParams{kv_cache.data().get(),  //
                                                     int(2 * kBatchSize * kContextLen * kHeadDim),
                                                     int(kBatchSize * kContextLen * kHeadDim)};

    params.quant_policy = kQuantPolicy;

    params.finished   = finished.data().get();
    params.rope_theta = rope_base.data().get();
    params.cu_q_len   = cu_seqlens.data().get();
    params.cu_k_len   = cu_kv_lens.data().get();

    params.num_heads     = kHeadNum;
    params.num_kv_heads  = KvHeadNum;
    params.size_per_head = kHeadDim;
    params.window_size   = kWindowSize;
    params.inv_sqrt_dh   = (float)std::log2(expf(1.)) / std::sqrt((float)params.size_per_head);

    if (SINK) {
        params.sinks       = sinks.data().get();
        params.scale_sinks = 1. / std::sqrt((float)params.size_per_head);
    }

    float scale_factor = -std::log2f(kRoPEBase) / kRoPEDim;
    params.rope_param  = RopeKernelParam{RopeType::kDefault, nullptr, kRoPEDim, scale_factor, 1.f};

    params.split_cnt  = split_cnt.data().get();
    params.partial_ML = partial_ML.data().get();
    params.partial_O  = partial_O.data().get();

    params.max_split_k = kMaxSplitK;
    params.arch        = getSMVersion();

    params.qk = qk_buf.data().get();
    params.pr = pr_buf.data().get();

    Reference<T> reference({});
    reference.Reshape(kInputLen, kContextLen, kHeadNum, kHeadDim, KvHeadNum, kBatchSize, kWindowSize);

    for (int i = 0; i < 1; ++i) {
        reference.Execute(params.out,  //
                          k_cache_ref.data().get(),
                          v_cache_ref.data().get(),
                          qkv.data().get(),
                          bias_QKV.data().get(),
                          SINK ? sinks.data().get() : nullptr,
                          kRoPEBase,
                          kRoPEDim);
    }

    cudaDeviceSynchronize();

    if constexpr (kDump) {
        for (size_t b = 0; b < kBatchSize; ++b) {
            for (size_t h = 0; h < kHeadNum; ++h) {
                for (size_t q = 0; q < kInputLen; ++q) {
                    auto qk = reference.qk() + b * kHeadNum * kInputLen * kContextLen + h * kInputLen * kContextLen
                              + q * kContextLen;
                    for (size_t k = 0; k < kContextLen; ++k) {
                        std::cout << qk[k] * params.inv_sqrt_dh << " ";
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }

    if (auto err = cudaGetLastError(); err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << "\n";
        return -1;
    }
    std::cout << "---------------------------------------------------\n";

    params.out = output.data().get();

    std::vector<thrust::universal_vector<T>> outputs;

    std::vector<cudaEvent_t> ev_start(kTestIter);
    std::vector<cudaEvent_t> ev_end(kTestIter);

    for (int i = 0; i < kTestIter; ++i) {
        cudaEventCreate(&ev_start[i]);
        cudaEventCreate(&ev_end[i]);
    }

    for (int i = 0; i < std::max(kTestIter, 1); ++i) {

#if DECODING
        cudaEventRecord(ev_start[i]);
        dispatchDecoding<T>(params);
        cudaEventRecord(ev_end[i]);
#else
        // input -> blocked
        invokeProcessKV_v2_(params);
        // blocked -> linear
        invokeFlattenKV_v2_(params, cu_kv_lens[kBatchSize]);

        cudaEventRecord(ev_start[i]);
        dispatchAttention(params);
        cudaEventRecord(ev_end[i]);
#endif

        if (auto err = cudaGetLastError(); err != cudaSuccess) {
            std::cout << cudaGetErrorString(err) << "\n";
            return -1;
        }
        if (1) {
            outputs.push_back(output);
        }
    }

    if (kDump) {
        cudaDeviceSynchronize();
        for (size_t b = 0; b < kBatchSize; ++b) {
            for (size_t h = 0; h < kHeadNum; ++h) {
                for (size_t q = 0; q < kInputLen; ++q) {
                    auto ref = reference.qk() + b * kHeadNum * kInputLen * kContextLen + h * kInputLen * kContextLen
                               + q * kContextLen;
                    auto data = qk_buf.data().get() + b * kHeadNum * kInputLen * kContextLen
                                + h * kInputLen * kContextLen + q * kContextLen;
                    for (size_t k = 0; k < kContextLen; ++k) {
                        // std::cout << std::max(0.f, std::abs(data[k] - (float)ref[k]) - 1e-5f) << " ";
                        std::cout << data[k] * params.inv_sqrt_dh << " ";
                        // std::cout << (float)data[k] << " ";
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }

    invokeFlattenKV_v2_(params, cu_kv_lens[kBatchSize]);
    cudaDeviceSynchronize();

    const size_t nbytes = blocks.size() / kContextLen * std::min(kContextLen, (size_t)params.window_size);
    const size_t ops =
        2 * kInputLen * std::min(kContextLen, (size_t)params.window_size) * kHeadDim * kHeadNum * kBatchSize;

    const float peak_bw = get_memory_bandwidth();

    std::cout << "Device peak global memory bandwidth: " << peak_bw << " GB/s\n";

    for (int i = 0; i < kTestIter; ++i) {
        float ms{};
        cudaEventElapsedTime(&ms, ev_start[i], ev_end[i]);
        const float bw      = nbytes / 1e9f / ms * 1000.f;
        const float flops   = ops / 1e12f / ms * 1000.f;
        const float percent = bw / peak_bw * 100.f;
        printf("time %.3f ms, bw %.3f GB/s, %.3f %%, tflops %.3f \n", ms, bw, percent, flops);
    }

    if (outputs.size() > 1) {
        std::cout << "Evaluating consistency..." << std::endl;
        for (size_t i = 1; i < outputs.size(); ++i) {
            Compare(outputs[i].data().get(), outputs[i - 1].data().get(), kHeadDim, kHeadDim, kHeadNum, 0, 0, 0);
        }
    }

    std::cout << "---------------------------------------------------\n";

    // [B, S, H, D]
    Compare(output.data().get(),  //
            output_ref.data().get(),
            kHeadNum * kHeadDim,
            kHeadNum * kHeadDim,
            kBatchSize * kInputLen,
            0);

    // [BH, SD]
    Compare(k_cache.data().get() + kSequenceLen * kHeadDim,
            k_cache_ref.data().get() + kSequenceLen * kHeadDim,
            kContextLen * kHeadDim,
            kInputLen * kHeadDim,
            kBatchSize * KvHeadNum,
            0);
    Compare(v_cache.data().get() + kSequenceLen * kHeadDim,
            v_cache_ref.data().get() + kSequenceLen * kHeadDim,
            kContextLen * kHeadDim,
            kInputLen * kHeadDim,
            kBatchSize * KvHeadNum);

    return 0;
}

int main(int argc, char* argv[])
{
    test_attention<half>();

#if defined(ENABLE_FP4)
    // Optional FP4 MXFP4 probe harness to validate scale pool layout and
    // payload packing. This is a lightweight debug path and is only built
    // when FP4 support is enabled.
    auto run_fp4_probe = []() {
        using T   = half;
        using Tkv = fp4_e2m1_t;

        constexpr int head_dim   = 64;
        constexpr int head_num   = 1;
        constexpr int block_len  = 64;
        constexpr int batch_size = 1;
        constexpr int layer_id   = 0;
        constexpr int head_idx   = 0;

        constexpr int q_len = block_len * 2;  // two blocks

        RNG rng{};

        // K/V tensors: [B, S, H, D] flattened as [S, D] since we force
        // stride_b/stride_c/stride_h to 0 and stride_s=1 in invokeProcessKV_v2.
        thrust::universal_vector<T> k(q_len * head_dim);
        thrust::universal_vector<T> v(q_len * head_dim);
        rng.GenerateNormal(k.data().get(), k.size());
        rng.GenerateNormal(v.data().get(), v.size());

        // Strengthen the probe: enforce a deterministic pattern on the
        // first token so that the first two 16-dim blocks along head_dim
        // clearly require different exponent scales. This lets us detect
        // any bug where FP4 MXFP4 quantization or decode accidentally
        // uses a block size != 16 or a wrong scale index.
        //
        // Token 0:
        //   dims [0..15]   -> large magnitude (8.0)
        //   dims [16..31]  -> small magnitude (0.125)
        // Remaining dims keep random values.
        {
            const int token0_offset = 0 * head_dim;
            for (int di = 0; di < 16 && di < head_dim; ++di) {
                k[token0_offset + di] = T(8.0f);
            }
            for (int di = 16; di < 32 && di < head_dim; ++di) {
                k[token0_offset + di] = T(0.125f);
            }
        }

        // Cumulative sequence lengths (single batch).
        thrust::universal_vector<int> cu_q_len(batch_size + 1);
        thrust::universal_vector<int> cu_k_len(batch_size + 1);
        cu_q_len[0] = 0;
        cu_q_len[1] = q_len;
        cu_k_len[0] = 0;
        cu_k_len[1] = q_len;

        // Block pointers for data pool.
        using BlockConfig  = block::Config<T, Tkv, head_dim>;
        block::Layout      block_layout{BlockConfig{head_num, block_len}};
        const size_t       block_bytes = block_layout.block_size(1);  // layer_num = 1
        const int          n_blocks    = (q_len + block_len - 1) / block_len;
        thrust::universal_vector<char>  blocks(block_bytes * n_blocks * batch_size);
        thrust::universal_vector<char*> block_ptrs(n_blocks * batch_size);
        for (int i = 0; i < n_blocks * batch_size; ++i) {
            block_ptrs[i] = blocks.data().get() + static_cast<size_t>(i) * block_bytes;
        }

        // Scale pool blocks for FP4 MXFP4.
        const int   scales_per_head   = head_dim / 16;
        const int   bytes_per_token   = 2 * scales_per_head;  // K scales + V scales
        const size_t scale_block_bytes =
            static_cast<size_t>(head_num) * block_len * static_cast<size_t>(bytes_per_token);  // layer_num = 1
        thrust::universal_vector<char>  scale_blocks(scale_block_bytes * n_blocks * batch_size);
        thrust::universal_vector<char*> scale_block_ptrs(n_blocks * batch_size);
        for (int i = 0; i < n_blocks * batch_size; ++i) {
            scale_block_ptrs[i] = scale_blocks.data().get() + static_cast<size_t>(i) * scale_block_bytes;
        }

        // Cumulative block counts.
        thrust::universal_vector<int> cu_block_cnts(batch_size + 1);
        cu_block_cnts[0] = 0;
        cu_block_cnts[1] = n_blocks;

        RopeKernelParam rope_param{};
        cutlass::FastDivmod cp_size(1);

        // Run ProcessKV_v2 in MXFP4 mode. We intentionally use simple
        // strides so that index = (qi * HeadDim + di).
        invokeProcessKV_v2((char**)block_ptrs.data().get(),
                           (char**)scale_block_ptrs.data().get(),
                           k.data().get(),
                           v.data().get(),
                           (T*)nullptr,
                           (T*)nullptr,
                           cu_q_len.data().get(),
                           cu_k_len.data().get(),
                           cu_block_cnts.data().get(),
                           rope_param,
                           /*stride_b*/ 0,
                           /*stride_c*/ 0,
                           /*stride_h*/ 0,
                           /*stride_s*/ 1,
                           block_len,
                           layer_id,
                           /*cp_rank*/ 0,
                           cp_size,
                           q_len,
                           head_num,
                           head_dim,
                           batch_size,
                           QuantPolicy::kCacheKVFp4,
                           getSMVersion(),
                           /*stream*/ nullptr);

        cudaDeviceSynchronize();

        // Probe two tokens: start of first block and start of second block.
        Fp4KvProbeResult res0{}, res1{};
        fp4_kv_probe_host(res0,
                          (char**)block_ptrs.data().get(),
                          (char**)scale_block_ptrs.data().get(),
                          layer_id,
                          head_idx,
                          head_num,
                          block_len,
                          head_dim,
                          /*local_ti*/ 0,
                          nullptr);
        fp4_kv_probe_host(res1,
                          (char**)block_ptrs.data().get(),
                          (char**)scale_block_ptrs.data().get(),
                          layer_id,
                          head_idx,
                          head_num,
                          block_len,
                          head_dim,
                          /*local_ti*/ block_len,
                          nullptr);

        std::cout << "[FP4Mx probe] token 0: "
                  << "k_scale0=" << int(res0.k_scale0) << ", v_scale0=" << int(res0.v_scale0)
                  << ", k_scale1=" << int(res0.k_scale1) << ", v_scale1=" << int(res0.v_scale1)
                  << ", kv_byte0=" << int(res0.kv_byte0) << "\n";
        std::cout << "[FP4Mx probe] token " << block_len << ": "
                  << "k_scale0=" << int(res1.k_scale0) << ", v_scale0=" << int(res1.v_scale0)
                  << ", k_scale1=" << int(res1.k_scale1) << ", v_scale1=" << int(res1.v_scale1)
                  << ", kv_byte0=" << int(res1.kv_byte0) << "\n";

        // Minimal di>>4 mapping assertion:
        //
        // For token 0 we constructed K so that dims [0..15] and [16..31]
        // have very different magnitudes. If MXFP4 is using a 16-element
        // block with scale index `scale_idx = di >> 4`, then the per-block
        // maxima should produce different exponents for block 0 and block 1.
        //
        // If a future change accidentally switches to a 32-element block
        // or miscomputes the scale index, k_scale0 and k_scale1 will tend
        // to collapse to the same value and this check will fail loudly.
        if (res0.k_scale0 == res0.k_scale1) {
            std::cerr << "[FP4Mx probe][ERROR] Expected different K scale bytes for blocks 0 and 1 at token 0 "
                      << "after enforcing a large magnitude difference between dims [0..15] and [16..31]. "
                      << "Check the MXFP4 block size (should be 16) and scale index computation (di >> 4)."
                      << std::endl;
            std::abort();
        }
    };

    run_fp4_probe();
#endif

    // test_attention<nv_bfloat16>();
}
