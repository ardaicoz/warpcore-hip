cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

# Set policy for FetchContent timestamps
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

CPMAddPackage(
  NAME Catch2
  GITHUB_REPOSITORY catchorg/Catch2
  VERSION 3.3.0
  DOWNLOAD_EXTRACT_TIMESTAMP ON
)

# Header for catch_discover_tests
if(Catch2_ADDED)
    include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)
endif()

function(ConfigureTest TEST_NAME)
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE HIP)
    add_executable(${TEST_NAME} ${ARGN})
    target_link_libraries(${TEST_NAME} PRIVATE Catch2::Catch2WithMain warpcore)
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(${TEST_NAME} PROPERTIES
                                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    target_compile_options(${TEST_NAME} PRIVATE -Wall -Wextra -Werror)
    catch_discover_tests(${TEST_NAME} EXTRA_ARGS --allow-running-no-tests)
endfunction(ConfigureTest)

ConfigureTest(tester
  src/key_value_store_test.hip.cpp
  src/status_test.hip.cpp
  src/single_value_hash_table_test.hip.cpp
  src/probing_schemes_test.hip.cpp)
