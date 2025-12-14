
#include "src/turbomind/models/common/model_layout.h"

namespace turbomind {

// TODO: These values should ideally come from a model configuration file or be passed dynamically.
// For now, these are placeholder values for GPT-OSS-120B.
// Reference: typical Llama-style model parameters
// Hidden size: 5120
// Intermediate size: 13824
// Number of heads: 40
// Number of KV heads: 40
// Number of layers: 48
// Vocab size: 32000
// Max sequence length: 8192
// Head Dim: 5120 / 40 = 128

ModelLayout make_gpt_oss_120b_layout() {
    ModelLayout layout;
    layout.num_layers = 48; // Placeholder
    layout.num_kv_heads = 40; // Placeholder
    layout.head_dim = 128; // Placeholder
    layout.page_size = 128; // Recommended KV page size, can be configured
    layout.max_seq_len = 8192; // Placeholder
    return layout;
}

} // namespace turbomind
