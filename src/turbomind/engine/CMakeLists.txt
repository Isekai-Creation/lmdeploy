# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.11)

add_library(engine STATIC gateway.cc request_queue.cc model_request.cc EngineScheduler.cc capacity_scheduler.cc drift_engine.cc)
target_link_libraries(engine PRIVATE core xgrammar)
target_include_directories(engine PRIVATE ${xgrammar_SOURCE_DIR})
set_property(TARGET engine PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET engine PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

if (BUILD_TEST)
    add_executable(test_engine_scheduler engine_scheduler_test.cc)
    target_link_libraries(test_engine_scheduler PRIVATE engine core logger Catch2::Catch2WithMain)
    add_executable(test_capacity_scheduler capacity_scheduler_test.cc)
    target_link_libraries(test_capacity_scheduler PRIVATE engine core logger Catch2::Catch2WithMain)
    add_executable(test_drift_engine_integration drift_engine_integration_test.cc)
    target_link_libraries(test_drift_engine_integration PRIVATE engine core logger Catch2::Catch2WithMain)
endif ()
