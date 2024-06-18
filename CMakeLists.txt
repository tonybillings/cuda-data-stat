cmake_minimum_required(VERSION 3.28)
project(cds LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_library(cds_lib
        src/storage.cpp
        src/service.cpp
        src/stats.cu
        include/cds/storage.h
        include/cds/service.h
        include/cds/cuda_error_check.h
        include/cds/file_metadata.h
        include/cds/data_stats.h
        include/cds/math.cuh
)

target_include_directories(cds_lib PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable(storage_test
    tests/storage_test.cpp
)

add_executable(service_test
    tests/service_test.cpp
)

target_link_libraries(storage_test PRIVATE cds_lib)

target_link_libraries(service_test PRIVATE cds_lib)

enable_testing()

add_test(NAME StorageTest COMMAND storage_test)

add_test(NAME ServiceTest COMMAND service_test)
