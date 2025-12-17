
#include "src/turbomind/models/common/model_layout.h"

namespace turbomind {

// Canonical GPT-OSS-120B (MoE) layout derived from bundled config:
// layers: 36, kv heads: 8, head_dim: 64, page_size: 128 (aligned with sliding_window),
// max_seq_len: 131072 (from max_position_embeddings), kv_dtype: fp16 by default.
ModelLayout make_gpt_oss_120b_layout() {
    ModelLayout layout{};
    layout.num_layers    = 36;
    layout.num_kv_heads  = 8;
    layout.head_dim      = 64;
    layout.page_size     = 128;
    layout.max_seq_len   = 131072;
    layout.kv_dtype      = KVDataType::kFP16;
    return layout;
}

ModelLayout make_test_layout() {
    // Tiny layout for unit tests / synthetic harnesses.
    ModelLayout layout{};
    layout.num_layers    = 2;
    layout.num_kv_heads  = 2;
    layout.head_dim      = 16;
    layout.page_size     = 32;
    layout.max_seq_len   = 2048;
    layout.kv_dtype      = KVDataType::kFP16;
    return layout;
}

} // namespace turbomind
