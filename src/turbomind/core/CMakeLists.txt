# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.11)

add_library(core STATIC
        check.cc
        allocator.cc
        stream.cc
        context.cc
        buffer.cc
        layout.cc
        tensor.cc
        tensor.cu
        kv_cache_manager.cc
        ../models/common/model_layout.cc
        prefix_cache.cc
        module.cc)

target_link_libraries(core PUBLIC cuda_utils logger CUDA::cudart CUDA::cuda_driver)

set_property(TARGET core PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET core PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_options(core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)

if (BUILD_TEST)
    add_executable(test_core test_core.cc)
    target_link_libraries(test_core PRIVATE core logger Catch2::Catch2WithMain)
    add_executable(test_kv_cache_manager kv_cache_manager_test.cc)
    target_link_libraries(test_kv_cache_manager PRIVATE core logger Catch2::Catch2WithMain)
    add_executable(test_prefix_cache prefix_cache_test.cc)
    target_link_libraries(test_prefix_cache PRIVATE core logger Catch2::Catch2WithMain)
endif ()
