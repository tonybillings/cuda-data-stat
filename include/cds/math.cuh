#ifndef MATH_CUH
#define MATH_CUH

#include <cuda_runtime.h>
#include <iostream>

__device__ inline float atomicMinFloat(float* valPtr, const float val) {
    const auto valIntPtr = reinterpret_cast<int*>(valPtr);
    int actualValInt = *valIntPtr;
    int expectedValInt;

    do {
        expectedValInt = actualValInt;
        const float newVal = fminf(val, __int_as_float(expectedValInt));
        actualValInt = atomicCAS(valIntPtr, expectedValInt, __float_as_int(newVal));
    } while (expectedValInt != actualValInt);

    return __int_as_float(actualValInt);
}

__device__ inline float atomicMaxFloat(float* valPtr, const float val) {
    const auto valIntPtr = reinterpret_cast<int*>(valPtr);
    int actualValInt = *valIntPtr;
    int expectedValInt;

    do {
        expectedValInt = actualValInt;
        const float newVal = fmaxf(val, __int_as_float(expectedValInt));
        actualValInt = atomicCAS(valIntPtr, expectedValInt, __float_as_int(newVal));
    } while (expectedValInt != actualValInt);

    return __int_as_float(actualValInt);
}

#endif // MATH_CUH
