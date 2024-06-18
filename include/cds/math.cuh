#ifndef MATH_H
#define MATH_H

#include <cuda_runtime.h>
#include <iostream>

__device__ inline float atomicMinFloat(float* val_ptr, const float val) {
    const auto val_int_ptr = reinterpret_cast<int*>(val_ptr);
    int actual_val_int = *val_int_ptr;
    int expected_val_int;

    do {
        expected_val_int = actual_val_int;
        const float new_val = fminf(val, __int_as_float(expected_val_int));
        actual_val_int = atomicCAS(val_int_ptr, expected_val_int, __float_as_int(new_val));
    } while (expected_val_int != actual_val_int);

    return __int_as_float(actual_val_int);
}

__device__ inline float atomicMaxFloat(float* val_ptr, const float val) {
    const auto val_int_ptr = reinterpret_cast<int*>(val_ptr);
    int actual_val_int = *val_int_ptr;
    int expected_val_int;

    do {
        expected_val_int = actual_val_int;
        const float new_val = fmaxf(val, __int_as_float(expected_val_int));
        actual_val_int = atomicCAS(val_int_ptr, expected_val_int, __float_as_int(new_val));
    } while (expected_val_int != actual_val_int);

    return __int_as_float(actual_val_int);
}

#endif // MATH_H
