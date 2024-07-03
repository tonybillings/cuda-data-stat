cmake_minimum_required(VERSION 3.28)
project(libcds LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 80 86 89)  # must support atomicAdd(double*, double)

add_library(cds SHARED
        include/cds/cuda_error.h
        include/cds/data_stats.h
        include/cds/debug.h
        include/cds/service.h
        include/cds/storage.h
        include/cds/cds.h

        include/cds/math.cuh
        include/cds/stats.cuh

        src/data_stats.cpp
        src/debug.cpp
        src/service.cpp
        src/storage.cpp

        src/stats.cu
)

target_include_directories(cds PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable(storage_test
        tests/storage_test.cpp
)
add_executable(service_test
        tests/service_test.cpp
)

target_link_libraries(storage_test PRIVATE cds)
target_link_libraries(service_test PRIVATE cds)

install(TARGETS cds DESTINATION lib)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/cds DESTINATION include)

enable_testing()

add_test(NAME StorageTest COMMAND storage_test)
add_test(NAME ServiceTest COMMAND service_test)
