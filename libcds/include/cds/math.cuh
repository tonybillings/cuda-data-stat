#ifndef MATH_CUH
#define MATH_CUH

#include <cuda_runtime.h>

__device__ inline double atomicMin(double* address, const double val) {
    auto* valPtr = reinterpret_cast<unsigned long long*>(address);
    unsigned long long actualVal = *valPtr;
    unsigned long long expectedVal;

    do {
        expectedVal = actualVal;
        const double old = __longlong_as_double(static_cast<long long>(actualVal));
        const double newVal = min(val, old);
        actualVal = atomicCAS(valPtr, expectedVal, __double_as_longlong(newVal));
    } while (expectedVal != actualVal);

    return __longlong_as_double(static_cast<long long>(actualVal));
}

__device__ inline double atomicMax(double* address, const double val) {
    auto* valPtr = reinterpret_cast<unsigned long long*>(address);
    unsigned long long actualVal = *valPtr;
    unsigned long long expectedVal;

    do {
        expectedVal = actualVal;
        const double old = __longlong_as_double(static_cast<long long>(actualVal));
        const double newVal = max(val, old);
        actualVal = atomicCAS(valPtr, expectedVal, __double_as_longlong(newVal));
    } while (expectedVal != actualVal);

    return __longlong_as_double(static_cast<long long>(actualVal));
}

#endif // MATH_CUH
