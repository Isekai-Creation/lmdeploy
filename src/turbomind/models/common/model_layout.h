
#pragma once

namespace turbomind {

struct ModelLayout {
    int num_layers;
    int num_kv_heads;
    int head_dim;
    int page_size;     // recommended KV page size
    int max_seq_len;
};

// TODO: Implement this function
ModelLayout make_gpt_oss_120b_layout();

} // namespace turbomind
