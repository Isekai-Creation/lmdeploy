
#pragma once

#include <cstdint>

namespace turbomind {

enum class KVDataType : uint8_t {
    kFP16,
    kBF16,
    kNVFP4,
};

inline int bytes_per_value_from_dtype(KVDataType dtype)
{
    switch (dtype) {
        case KVDataType::kFP16:
        case KVDataType::kBF16:
            return 2;
        case KVDataType::kNVFP4:
            // Use conservative 2 bytes until NVFP4 scale metadata is modeled; avoids underestimating capacity.
            return 2;
    }
    return 0;
}

struct ModelLayout {
    int        num_layers{0};
    int        num_kv_heads{0};
    int        head_dim{0};
    int        page_size{0};     // tokens per KV page
    int        max_seq_len{0};
    KVDataType kv_dtype{KVDataType::kFP16};
};

ModelLayout make_gpt_oss_120b_layout();
ModelLayout make_test_layout();

} // namespace turbomind
