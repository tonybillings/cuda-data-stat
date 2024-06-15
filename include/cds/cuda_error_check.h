#ifndef CUDA_ERROR_CHECK_H
#define CUDA_ERROR_CHECK_H

#include <cuda_runtime.h>
#include <iostream>

inline void cuda_error_check(const cudaError_t value, const char* file, const int line) {
    if (value != cudaSuccess) {
        std::cerr << "Error " << cudaGetErrorString(value) << " at line " << line
            << " in file " << file << std::endl;
        exit(1);
    }
}

#define CUDA_ERROR_CHECK(value) cuda_error_check(value, __FILE__, __LINE__)

#endif // CUDA_ERROR_CHECK_H
