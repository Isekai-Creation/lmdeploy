// Copyright (c) OpenMMLab. All rights reserved.
// Eagle-3 attention layer. This backend is dedicated to Eagle-3
// geometry (e.g. 4096x2880 / 512x2880 / 2880x4096) and is wired
// into Eagle3DraftLayer, separate from UnifiedAttentionLayer.

#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/models/llama/Eagle3AttentionWeight.h"
#include "src/turbomind/models/llama/LlamaDecoderLayerWeight.h"

namespace turbomind {

namespace ft {



template<typename T>

__global__ void apply_rope_kernel(T*       q_ptr,

                                  T*       k_ptr,

                                  int      token_num,

                                  int      num_q_heads,

                                  int      num_kv_heads,

                                  int      head_dim,

                                  int      q_len,

                                  int      past_kv_len,

                                  const int* position_ids,

                                  float    rope_base,

                                  float    rope_scale);



template<typename T>

__global__ void sdpa_kernel(const T* __restrict__ q_ptr,

                            const T* __restrict__ k_ptr,

                            const T* __restrict__ v_ptr,

                            T* __restrict__       ctx_ptr,

                            int                   token_num,

                            int                   batch_size,

                            int                   q_len,

                            int                   kv_len,

                            int                   num_q_heads,

                            int                   num_kv_heads,

                            int                   head_dim,

                            int                   past_kv_len,

                            const int*            position_ids,

                            const int32_t*        packed_mask,

                            int                   packed_stride,

                            const int32_t*        runtime_offsets,

                            const int32_t*        tree_offsets,

                            const int32_t*        kv_lens_runtime,

                            const int32_t*        successor_offsets,

                            const int32_t*        successor_counts);



template<typename T>

__global__ void expand_kv_to_q_kernel(const T* __restrict__ kv,

                                      T* __restrict__ q,

                                      int batch,

                                      int kv_heads,

                                      int head_dim,

                                      int group_size);



template<typename T>

void launch_apply_rope_kernel(T* q_ptr,

                              T* k_ptr,

                              int token_num,

                              int num_q_heads,

                              int num_kv_heads,

                              int head_dim,

                              int q_len,

                              int past_kv_len,

                              const Tensor* position_ids,

                              float rope_base,

                              float rope_scale,

                              cudaStream_t stream);



template<typename T>

void launch_sdpa_kernel(const T* q_ptr,

                        const T* k_ptr,

                        const T* v_ptr,

                        T*       ctx_ptr,

                        int      token_num,

                        int      batch_size,

                        int      q_len,

                        int      kv_len,

                        int      num_q_heads,

                        int      num_kv_heads,

                        int      head_dim,

                        int      past_kv_len,

                        const Tensor* position_ids,

                        const Tensor* packed_mask,

                        int      packed_mask_stride,

                        const Tensor* runtime_offsets,

                        const Tensor* tree_offsets,

                        const Tensor* kv_lens_runtime,

                        const Tensor* successor_offsets,

                        const Tensor* successor_counts,

                        cudaStream_t stream);



template<typename T>
void launch_expand_kv_to_q_kernel(const Tensor& kv, Tensor& q_expanded, int kv_heads, int head_dim, int group_size, cudaStream_t stream);

// Tree-aware Eagle3 SDPA kernel that consumes precomputed per-token KV
// windows [kv_start_per_token[t], kv_start_per_token[t] +
// kv_len_per_token[t]) together with the packed mask. This is wired
// behind TM_ENABLE_EAGLE3_FMHA in Eagle3AttentionLayer and is the first
// concrete FMHA-style path for Eagle3.
template<typename T>
void launch_eagle3_fmha_kernel(const T*       q_ptr,
                               const T*       k_ptr,
                               const T*       v_ptr,
                               T*             ctx_ptr,
                               int            token_num,
                               int            batch_size,
                               int            q_len,
                               int            kv_len,
                               int            num_q_heads,
                               int            num_kv_heads,
                               int            head_dim,
                               int            past_kv_len,
                               const Tensor*  position_ids,
                               const Tensor*  packed_mask,
                               int            packed_mask_stride,
                               const int32_t* kv_start_per_token,
                               const int32_t* kv_len_per_token,
                               float*         partial_o,
                               float*         partial_m,
                               float*         partial_l,
                               int            partial_tiles,
                               cudaStream_t   stream);

// Non-templated convenience wrappers for the common dtypes so that
// call sites (e.g. Eagle3AttentionLayer) do not need to instantiate
// the FMHA template across translation units.
void launch_eagle3_fmha_kernel_fp16(const half_t*       q_ptr,
                                    const half_t*       k_ptr,
                                    const half_t*       v_ptr,
                                    half_t*             ctx_ptr,
                                    int                 token_num,
                                    int                 batch_size,
                                    int                 q_len,
                                    int                 kv_len,
                                    int                 num_q_heads,
                                    int                 num_kv_heads,
                                    int                 head_dim,
                                    int                 past_kv_len,
                                    const Tensor*       position_ids,
                                    const Tensor*       packed_mask,
                                    int                 packed_mask_stride,
                                    const int32_t*      kv_start_per_token,
                                    const int32_t*      kv_len_per_token,
                                    float*              partial_o,
                                    float*              partial_m,
                                    float*              partial_l,
                                    int                 partial_tiles,
                                    cudaStream_t        stream);

#if ENABLE_BF16
void launch_eagle3_fmha_kernel_bf16(const bfloat16_t*   q_ptr,
                                    const bfloat16_t*   k_ptr,
                                    const bfloat16_t*   v_ptr,
                                    bfloat16_t*         ctx_ptr,
                                    int                 token_num,
                                    int                 batch_size,
                                    int                 q_len,
                                    int                 kv_len,
                                    int                 num_q_heads,
                                    int                 num_kv_heads,
                                    int                 head_dim,
                                    int                 past_kv_len,
                                    const Tensor*       position_ids,
                                    const Tensor*       packed_mask,
                                    int                 packed_mask_stride,
                                    const int32_t*      kv_start_per_token,
                                    const int32_t*      kv_len_per_token,
                                    float*              partial_o,
                                    float*              partial_m,
                                    float*              partial_l,
                                    int                 partial_tiles,
                                    cudaStream_t        stream);
#endif

#if ENABLE_FP32
void launch_eagle3_fmha_kernel_fp32(const float*        q_ptr,
                                    const float*        k_ptr,
                                    const float*        v_ptr,
                                    float*              ctx_ptr,
                                    int                 token_num,
                                    int                 batch_size,
                                    int                 q_len,
                                    int                 kv_len,
                                    int                 num_q_heads,
                                    int                 num_kv_heads,
                                    int                 head_dim,
                                    int                 past_kv_len,
                                    const Tensor*       position_ids,
                                    const Tensor*       packed_mask,
                                    int                 packed_mask_stride,
                                    const int32_t*      kv_start_per_token,
                                    const int32_t*      kv_len_per_token,
                                    float*              partial_o,
                                    float*              partial_m,
                                    float*              partial_l,
                                    int                 partial_tiles,
                                    cudaStream_t        stream);
#endif

template<typename T>
void launch_eagle3_matmul_rowmajor_dispatch(const T* A,
                                            const T* B,
                                            T*       C,
                                            int      M,
                                            int      K,
                                            int      N,
                                            cudaStream_t stream);

// Build per-token KV spans for Eagle3 attention on GPU. For each token
// index t in [0, token_num), this computes a clamped KV window
// [kv_start_per_token[t], kv_start_per_token[t] + kv_len_per_token[t])
// using the same runtime/tree/successor/kv_lens logic as the current
// SDPA kernels, but without applying packed masks. The result is
// suitable for feeding into a tree-aware FMHA scheduler.
void launch_build_eagle3_kv_spans(int           token_num,
                                  int           batch_size,
                                  int           q_len,
                                  int           kv_len,
                                  const int32_t* runtime_offsets,
                                  const int32_t* tree_offsets,
                                  const int32_t* kv_lens_runtime,
                                  const int32_t* successor_offsets,
                                  const int32_t* successor_counts,
                                  int32_t*       kv_start_per_token,
                                  int32_t*       kv_len_per_token,
                                  cudaStream_t   stream);



}  // namespace ft





struct Eagle3AttentionParam {
    // Input hidden after Eagle-3 pre-FC and norm.
    // Shape: [B, q_in] (e.g. [B, 2 * draft_hidden]).
    Tensor input;

    // Output hidden after attention. Draft layer preallocates this to
    // the base-hidden width (Wo_out) so we honour the provided shape.
    Tensor output;

    // Optional fused LLaMA-style attention weights (for Wo projection)
    // used alongside the native Eagle3 projections.
    const LlamaAttentionWeight* attn_weights{nullptr};

    const Eagle3AttentionWeight* weights{nullptr};

    // Optional per-token position ids for RoPE offsetting.
    const Tensor* position_ids{nullptr};

    // Optional packed tree mask for Eagle-3 target-tree or multi-token
    // scenarios. Unused in the current single-token implementation.
    const Tensor* packed_mask{nullptr};
    int           packed_mask_stride{0};

    // Optional tree/runtime offsets and successor metadata for multi-token
    // draft attention. Runtime offsets mirror kv_lens_runtime semantics in
    // TRT (excluding extra KV tokens), while tree offsets describe the full
    // flattened tree layout. These are currently used to clamp kv_len per
    // request in the naive SDPA path.
    const Tensor* tree_offsets{nullptr};     // [batch+1]
    const Tensor* runtime_offsets{nullptr};  // [batch+1]
    const Tensor* kv_lens_runtime{nullptr};  // [batch] optional clamp excluding extra draft tokens
    const Tensor* successor_offsets{nullptr};  // [batch+1]
    const Tensor* successor_counts{nullptr};   // [total successors]

    // Token geometry for multi-position draft attention.
    int batch_size{0};
    int q_len{1};
    int kv_len{1};
    int past_kv_len{0};
    float rope_base{10000.f};
    float rope_scale{1.f};

    int layer_id{0};

    // Optional debug captures.
    Tensor* debug_qkv{nullptr};
    Tensor* debug_attn_out{nullptr};
};

class Eagle3AttentionLayer {
public:
    Eagle3AttentionLayer(const cudaDeviceProp* prop, cudaStream_t stream);
    ~Eagle3AttentionLayer() = default;

    // Forward applies a simplified Eagle-3 attention block:
    //   Q = input @ q_proj^T
    //   Y = Q @ o_proj^T
    // This uses the native Eagle-3 midlayer projections without trying
    // to squeeze them into the standard LLaMA fused QKV layout. The K
    // and V projections are reserved for future full SDPA wiring.
    void Forward(Eagle3AttentionParam& param);

private:
    cudaStream_t           stream_{};
    const cudaDeviceProp*  device_prop_{nullptr};

    // FMHA scratch buffers for the tiled multi-CTA path. These are
    // sized lazily based on (token_num * num_q_heads * num_tiles) and
    // reused across steps to avoid per-step cudaMalloc/cudaFree in the
    // hot decode loop. They are only used when TM_ENABLE_EAGLE3_FMHA
    // is active and the tiled FMHA path is selected.
    core::Tensor fmha_partial_m_;
    core::Tensor fmha_partial_l_;
    core::Tensor fmha_partial_o_;

    // Debug-only guard for concurrent FMHA usage within a single layer
    // instance. When LMDEPLOY_EAGLE_FMHA_CONCURRENCY_DEBUG is enabled,
    // Forward() will abort if it detects re-entrancy while an FMHA call
    // is in flight on this instance.
    bool fmha_in_flight_{false};
};

}  // namespace turbomind
