cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

ConfigureExecutable(bloom_filter_benchmark "${CMAKE_CURRENT_SOURCE_DIR}/src/bloom_filter_benchmark.hip.cpp" "${WARPCORE_BENCHMARKS_BINARY_DIR}")
ConfigureExecutable(bucket_list_benchmark "${CMAKE_CURRENT_SOURCE_DIR}/src/bucket_list_benchmark.hip.cpp" "${WARPCORE_BENCHMARKS_BINARY_DIR}")
ConfigureExecutable(counting_benchmark "${CMAKE_CURRENT_SOURCE_DIR}/src/counting_benchmark.hip.cpp" "${WARPCORE_BENCHMARKS_BINARY_DIR}")
ConfigureExecutable(multi_value_benchmark "${CMAKE_CURRENT_SOURCE_DIR}/src/multi_value_benchmark.hip.cpp" "${WARPCORE_BENCHMARKS_BINARY_DIR}")
ConfigureExecutable(single_value_benchmark "${CMAKE_CURRENT_SOURCE_DIR}/src/single_value_benchmark.hip.cpp" "${WARPCORE_BENCHMARKS_BINARY_DIR}")
ConfigureExecutable(multi_bucket_benchmark "${CMAKE_CURRENT_SOURCE_DIR}/src/multi_bucket_benchmark.hip.cpp" "${WARPCORE_BENCHMARKS_BINARY_DIR}")
