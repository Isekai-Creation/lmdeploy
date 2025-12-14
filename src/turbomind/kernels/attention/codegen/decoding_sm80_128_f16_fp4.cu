// Copyright (c) OpenMMLab. All rights reserved.

#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

using namespace attention;

template bool invokeDecoding<Decoding<arch::Sm80, half, fp4_e2m1_t, 1, 128>>(const AttentionParams<half>& params);

}  // namespace turbomind
