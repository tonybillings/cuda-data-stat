/*******************************************************************************
 INCLUDES
*******************************************************************************/

#include "cds/stats.cuh"
#include "cds/math.cuh"
#include "cds/debug.h"

#include <cfloat>
#include <cstdarg>

/*******************************************************************************
 USINGS
*******************************************************************************/

using std::sqrt;

/*******************************************************************************
 KERNELS
*******************************************************************************/

__global__ void calculateMins(const float* data, const size_t recordCount, const size_t fieldCount,
    float* mins) {
    extern __shared__ float sdataMins[];

    const unsigned int globalIdxX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int globalIdxY = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int localIdx = threadIdx.x;

    sdataMins[localIdx] = FLT_MAX;
    __syncthreads();

    if (globalIdxX < recordCount && globalIdxY < fieldCount) {
        sdataMins[localIdx] = data[globalIdxX * fieldCount + globalIdxY];
    } else {
        return;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (localIdx < s) {
            sdataMins[localIdx] = fminf(sdataMins[localIdx], sdataMins[localIdx + s]);
        }
        __syncthreads();
    }

    if (localIdx == 0) {
        atomicMinFloat(&mins[globalIdxY], sdataMins[0]);
    }
}

__global__ void calculateMaxs(const float* data, const size_t recordCount, const size_t fieldCount,
    float* maxs) {
    extern __shared__ float sdataMaxs[];

    const unsigned int globalIdxX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int globalIdxY = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int localIdx = threadIdx.x;

    sdataMaxs[localIdx] = FLT_MIN;
    __syncthreads();

    if (globalIdxX < recordCount && globalIdxY < fieldCount) {
        sdataMaxs[localIdx] = data[globalIdxX * fieldCount + globalIdxY];
    } else {
        return;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (localIdx < s) {
            sdataMaxs[localIdx] = fmaxf(sdataMaxs[localIdx], sdataMaxs[localIdx + s]);
        }
        __syncthreads();
    }

    if (localIdx == 0) {
        atomicMaxFloat(&maxs[globalIdxY], sdataMaxs[0]);
    }
}

__global__ void calculateTotals(const float* data, const size_t recordCount, const size_t fieldCount,
    float* totals) {
    extern __shared__ float sdataTotals[];

    const unsigned int globalIdxX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int globalIdxY = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int localIdx = threadIdx.x;

    sdataTotals[localIdx] = 0.0f;
    __syncthreads();

    if (globalIdxX < recordCount && globalIdxY < fieldCount) {
        sdataTotals[localIdx] = data[globalIdxX * fieldCount + globalIdxY];
    } else {
        return;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (localIdx < s) {
            sdataTotals[localIdx] += sdataTotals[localIdx + s];
        }
        __syncthreads();
    }

    if (localIdx == 0) {
        atomicAdd(&totals[globalIdxY], sdataTotals[0]);
    }
}

__global__ void convertDataToDeltas(float* data, const size_t recordCount, const size_t fieldCount) {
    const unsigned int globalIdxX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int globalIdxY = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int globalIdx = globalIdxX * fieldCount + globalIdxY;

    float localValue;
    float otherValue;
    if (globalIdxX < recordCount && globalIdxY < fieldCount) {
        localValue = data[globalIdx];
        if (globalIdxX == 0)
            otherValue = data[(globalIdxX + 1) * fieldCount + globalIdxY];
        else
            otherValue = data[(globalIdxX - 1) * fieldCount + globalIdxY];
    } else {
        return;
    }
    __syncthreads();

    data[globalIdx] = abs(otherValue - localValue);
}

__global__ void convertDeltasToDeltasSquared(float* data, const size_t recordCount, const size_t fieldCount) {
    const unsigned int globalIdxX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int globalIdxY = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int globalIdx = globalIdxX * fieldCount + globalIdxY;

    if (globalIdxX < recordCount && globalIdxY < fieldCount) {
        data[globalIdx] = pow(data[globalIdx], 2);
    }
}

__global__ void calculateMeansStddevs(const float* data, const size_t recordCount, const size_t fieldCount,
    const float* totals, float* means, float* stdDevs) {
    extern __shared__ float sdataStddev[];

    const unsigned int globalIdxX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int globalIdxY = blockIdx.y;
    const unsigned int localIdx = threadIdx.x;

    sdataStddev[localIdx] = 0.0f;
    __syncthreads();

    float val = 0.0f;
    if (globalIdxX < recordCount && globalIdxY < fieldCount) {
        val = data[globalIdxX * fieldCount + globalIdxY];
    }
    __syncthreads();

    if (localIdx == 0) {
        means[globalIdxY] = totals[globalIdxY] / static_cast<float>(recordCount);
    }
    __syncthreads();

    if (globalIdxX < recordCount && globalIdxY < fieldCount) {
        const float diff = val - means[globalIdxY];
        sdataStddev[localIdx] = diff * diff;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (localIdx < s) {
            sdataStddev[localIdx] += sdataStddev[localIdx + s];
        }
        __syncthreads();
    }

    if (localIdx == 0) {
        atomicAdd(&stdDevs[globalIdxY], sdataStddev[0]);
    }
}

/*******************************************************************************
 KERNEL WRAPPERS
*******************************************************************************/

namespace {
    bool runCalculateMins(dim3 grid, dim3 block, size_t sharedMemorySize, cudaStream_t stream, const float* data,
        const size_t recordCount, const size_t fieldCount, float* mins) {
            calculateMins<<<grid, block, sharedMemorySize, stream>>>(data, recordCount, fieldCount, mins);
            if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
                ERROR("calculateMins failed: %s", cudaGetErrorString(err));
                return false;
            }
            return true;
    }

    bool runCalculateMaxs(dim3 grid, dim3 block, size_t sharedMemorySize, cudaStream_t stream, const float* data,
        const size_t recordCount, const size_t fieldCount, float* maxs) {
            calculateMaxs<<<grid, block, sharedMemorySize, stream>>>(data, recordCount, fieldCount, maxs);
            if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
                ERROR("calculateMaxs failed: %s", cudaGetErrorString(err));
                return false;
            }
            return true;
    }

    bool runCalculateTotals(dim3 grid, dim3 block, size_t sharedMemorySize, cudaStream_t stream, const float* data,
        const size_t recordCount, const size_t fieldCount, float* totals) {
            calculateTotals<<<grid, block, sharedMemorySize, stream>>>(data, recordCount, fieldCount, totals);
            if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
                ERROR("calculateTotals failed: %s", cudaGetErrorString(err));
                return false;
            }
            return true;
    }

    bool runCalculateMeansStddevs(dim3 grid, dim3 block, size_t sharedMemorySize, cudaStream_t stream,
        const float* data, const size_t recordCount, const size_t fieldCount, const float* totals,
        float* means, float* stdDevs) {
            calculateMeansStddevs<<<grid, block, sharedMemorySize, stream>>>(data, recordCount, fieldCount, totals, means, stdDevs);
            if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
                ERROR("calculateMeansStddevs failed: %s", cudaGetErrorString(err));
                return false;
            }
            return true;
    }

    bool runConvertDataToDeltas(dim3 grid, dim3 block, size_t sharedMemorySize, cudaStream_t stream,
        float *data, const size_t recordCount, const size_t fieldCount) {
        convertDataToDeltas<<<grid, block, sharedMemorySize, stream>>>(data, recordCount, fieldCount);
        if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
            ERROR("convertDataToDeltas failed: %s", cudaGetErrorString(err));
            return false;
        }
        return true;
    }
}

/*******************************************************************************
 UTILITY FUNCTIONS
*******************************************************************************/

namespace {
    bool initializeMemory(float* dPtr, const int value, const size_t size, const char* name) {
        if (const cudaError_t err = cudaMemset(dPtr, value, size); err != cudaSuccess) {
            ERROR("cudaMemset failed for %s: %s", name, cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool freeMemory(const int count, ...) {
        va_list args;
        va_start(args, count);

        for (int i = 0; i < count; i++) {
            if (float* deviceArray = va_arg(args, float*); deviceArray != nullptr) {
                if (const cudaError_t err = cudaFree(deviceArray); err != cudaSuccess) {
                    ERROR("cudaFree failed: %s", cudaGetErrorString(err));
                    return false;
                }
            }
        }

        va_end(args);
        return true;
    }

    bool copyToDevice(float* dPtr, const char* data, const size_t dataSize) {
        if (const cudaError_t err = cudaMemcpy(dPtr, data, dataSize, cudaMemcpyHostToDevice); err != cudaSuccess) {
            ERROR("copyToDevice failed for data: %s", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool copyFromDevice(float* hPtr, const float* dPtr, const size_t size, const char* name) {
        if (const cudaError_t err = cudaMemcpy(hPtr, dPtr, size, cudaMemcpyDeviceToHost); err != cudaSuccess) {
            ERROR("copyFromDevice failed for %s: %s", name, cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool allocateMemory(float*& dPtr, const size_t size, const char* name) {
        if (const cudaError_t err = cudaMalloc(&dPtr, size); err != cudaSuccess) {
            ERROR("cudaMalloc failed for %s: %s", name, cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool allocateMemory(const char* data, const size_t dataSize, const DataStats& stats, float*& dData,
        float*& dMins, float*& dMaxs, float*& dTotals, float*& dMeans, float*& dStdDevs,
        float*& dDeltaMins, float*& dDeltaMaxs, float*& dDeltaTotals, float*& dDeltaMeans, float*& dDeltaStdDevs) {
        const auto statsSize = stats.fieldCount * sizeof(float);

        if (!allocateMemory(dData, dataSize, "dData")) {
            return false;
        }
        if (!copyToDevice(dData, data, dataSize)) {
            freeMemory(1, dData);
            return false;
        }
        if (!allocateMemory(dMins, statsSize, "dMins")) {
            freeMemory(1, dData);
            return false;
        }
        if (!allocateMemory(dMaxs, statsSize, "dMaxs")) {
            freeMemory(2, dData, dMins);
            return false;
        }
        if (!allocateMemory(dTotals, statsSize, "dTotals")) {
            freeMemory(3, dData, dMins, dMaxs);
            return false;
        }
        if (!allocateMemory(dMeans, statsSize, "dMeans")) {
            freeMemory(4, dData, dMins, dMaxs, dTotals);
            return false;
        }
        if (!allocateMemory(dStdDevs, statsSize, "dStdDevs")) {
            freeMemory(5, dData, dMins, dMaxs, dTotals, dMeans);
            return false;
        }
        if (!allocateMemory(dDeltaMins, statsSize, "dDeltaMins")) {
            freeMemory(6, dData, dMins, dMaxs, dTotals, dMeans, dStdDevs);
            return false;
        }
        if (!allocateMemory(dDeltaMaxs, statsSize, "dDeltaMaxs")) {
            freeMemory(7, dData, dMins, dMaxs, dTotals, dMeans, dStdDevs,
                dDeltaMins);
            return false;
        }
        if (!allocateMemory(dDeltaTotals, statsSize, "dDeltaTotals")) {
            freeMemory(8, dData, dMins, dMaxs, dTotals, dMeans, dStdDevs,
                dDeltaMins, dDeltaMaxs);
            return false;
        }
        if (!allocateMemory(dDeltaMeans, statsSize, "dDeltaMeans")) {
            freeMemory(9, dData, dMins, dMaxs, dTotals, dMeans, dStdDevs,
                dDeltaMins, dDeltaMaxs, dDeltaTotals);
            return false;
        }
        if (!allocateMemory(dDeltaStdDevs, statsSize, "dDeltaStdDevs")) {
            freeMemory(10, dData, dMins, dMaxs, dTotals, dMeans, dStdDevs,
                dDeltaMins, dDeltaMaxs, dDeltaTotals, dDeltaMeans);
            return false;
        }

        return true;
    }

    bool synchronizeDevice() {
        if (const cudaError_t err = cudaDeviceSynchronize(); err != cudaSuccess) {
            ERROR("cudaDeviceSynchronize failed: %s", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    void finishCalculatingStdDevs(const size_t fieldCount, const size_t recordCount, float* stdDevs) {
        for (size_t i = 0; i < fieldCount; i++) {
            stdDevs[i] = sqrt(stdDevs[i] / static_cast<float>(recordCount));
        }
    }
}

/*******************************************************************************
 INTERNAL FUNCTIONS
*******************************************************************************/

bool calculateStats(const char *data, const size_t dataSize, DataStats &stats) {
    float *dData;
    float *dMins, *dMaxs, *dTotals, *dMeans, *dStdDevs;
    float *dDeltaMins, *dDeltaMaxs, *dDeltaTotals, *dDeltaMeans, *dDeltaStdDevs;

    if (!allocateMemory(data, dataSize, stats, dData, dMins, dMaxs, dTotals, dMeans, dStdDevs,
        dDeltaMins, dDeltaMaxs, dDeltaTotals, dDeltaMeans, dDeltaStdDevs)) {
        return false;
    }

    const auto statsSize = stats.fieldCount * sizeof(float);
    if (!initializeMemory(dMins, INT_MAX, statsSize, "dMins") ||
        !initializeMemory(dMaxs, INT_MIN, statsSize, "dMaxs") ||
        !initializeMemory(dTotals, 0, statsSize, "dTotals") ||
        !initializeMemory(dMeans, 0, statsSize, "dMeans") ||
        !initializeMemory(dStdDevs, 0, statsSize, "dStdDevs") ||
        !initializeMemory(dDeltaMins, INT_MAX, statsSize, "dDeltaMins") ||
        !initializeMemory(dDeltaMaxs, INT_MIN, statsSize, "dDeltaMaxs") ||
        !initializeMemory(dDeltaTotals, 0, statsSize, "dDeltaTotals") ||
        !initializeMemory(dDeltaMeans, 0, statsSize, "dDeltaMeans") ||
        !initializeMemory(dDeltaStdDevs, 0, statsSize, "dDeltaStdDevs")) {
        freeMemory(11, dData, dMins, dMaxs, dTotals, dMeans, dStdDevs,
            dDeltaMins, dDeltaMaxs, dDeltaTotals, dDeltaMeans, dDeltaStdDevs);
        return false;
    }

    // TODO: handle case where record count is greater than max threads (implement batching)
    constexpr size_t blockWidth = 32; // TODO: make configurable
    const size_t blockCount = (stats.recordCount + blockWidth - 1) / blockWidth;
    const dim3 grid(blockCount, stats.fieldCount);
    constexpr dim3 block(blockWidth, 1);
    constexpr size_t sharedMemSize = blockWidth * sizeof(float);

    cudaStream_t minStream, maxStream;
    cudaStreamCreate(&minStream);
    cudaStreamCreate(&maxStream);

    if (!runCalculateMins(grid, block, sharedMemSize, minStream, dData, stats.recordCount, stats.fieldCount, dMins) ||
        !runCalculateMaxs(grid, block, sharedMemSize, maxStream, dData, stats.recordCount, stats.fieldCount, dMaxs) ||
        !runCalculateTotals(grid, block, sharedMemSize, nullptr, dData, stats.recordCount, stats.fieldCount, dTotals) ||
        !synchronizeDevice() ||
        !runCalculateMeansStddevs(grid, block, sharedMemSize, nullptr, dData, stats.recordCount, stats.fieldCount, dTotals, dMeans, dStdDevs) ||
        !synchronizeDevice()) {
        freeMemory(11, dData, dMins, dMaxs, dTotals, dMeans, dStdDevs,
            dDeltaMins, dDeltaMaxs, dDeltaTotals, dDeltaMeans, dDeltaStdDevs);
        return false;
    }

    if (!runConvertDataToDeltas(grid, block, sharedMemSize, nullptr, dData, stats.recordCount, stats.fieldCount) ||
        !synchronizeDevice()) {
        freeMemory(11, dData, dMins, dMaxs, dTotals, dMeans, dStdDevs,
            dDeltaMins, dDeltaMaxs, dDeltaTotals, dDeltaMeans, dDeltaStdDevs);
        return false;
    }

    if (!runCalculateMins(grid, block, sharedMemSize, minStream, dData, stats.recordCount, stats.fieldCount, dDeltaMins) ||
        !runCalculateMaxs(grid, block, sharedMemSize, maxStream, dData, stats.recordCount, stats.fieldCount, dDeltaMaxs) ||
        !runCalculateTotals(grid, block, sharedMemSize, nullptr, dData, stats.recordCount, stats.fieldCount, dDeltaTotals) ||
        !synchronizeDevice() ||
        !runCalculateMeansStddevs(grid, block, sharedMemSize, nullptr, dData, stats.recordCount, stats.fieldCount, dDeltaTotals, dDeltaMeans, dDeltaStdDevs) ||
        !synchronizeDevice()) {
        freeMemory(11, dData, dMins, dMaxs, dTotals, dMeans, dStdDevs,
            dDeltaMins, dDeltaMaxs, dDeltaTotals, dDeltaMeans, dDeltaStdDevs);
        return false;
    }

    float hMins[stats.fieldCount], hMaxs[stats.fieldCount],
        hTotals[stats.fieldCount], hMeans[stats.fieldCount], hStdDevs[stats.fieldCount];

    float hDeltaMins[stats.fieldCount], hDeltaMaxs[stats.fieldCount],
            hDeltaTotals[stats.fieldCount], hDeltaMeans[stats.fieldCount], hDeltaStdDevs[stats.fieldCount];

    if (!copyFromDevice(hMins, dMins, statsSize, "hMins") ||
        !copyFromDevice(hMaxs, dMaxs, statsSize, "hMaxs") ||
        !copyFromDevice(hTotals, dTotals, statsSize, "hTotals") ||
        !copyFromDevice(hMeans, dMeans, statsSize, "hMeans") ||
        !copyFromDevice(hStdDevs, dStdDevs, statsSize, "hStdDevs")) {
        freeMemory(11, dData, dMins, dMaxs, dTotals, dMeans, dStdDevs,
            dDeltaMins, dDeltaMaxs, dDeltaTotals, dDeltaMeans, dDeltaStdDevs);
        return false;
    }

    if (!copyFromDevice(hDeltaMins, dDeltaMins, statsSize, "hDeltaMins") ||
        !copyFromDevice(hDeltaMaxs, dDeltaMaxs, statsSize, "hDeltaMaxs") ||
        !copyFromDevice(hDeltaTotals, dDeltaTotals, statsSize, "hDeltaTotals") ||
        !copyFromDevice(hDeltaMeans, dDeltaMeans, statsSize, "hDeltaMeans") ||
        !copyFromDevice(hDeltaStdDevs, dDeltaStdDevs, statsSize, "hDeltaStdDevs")) {
        freeMemory(11, dData, dMins, dMaxs, dTotals, dMeans, dStdDevs,
            dDeltaMins, dDeltaMaxs, dDeltaTotals, dDeltaMeans, dDeltaStdDevs);
        return false;
    }

    finishCalculatingStdDevs(stats.fieldCount, stats.recordCount, hStdDevs);
    finishCalculatingStdDevs(stats.fieldCount, stats.recordCount, hDeltaStdDevs);

    freeMemory(11, dData, dMins, dMaxs, dTotals, dMeans, dStdDevs,
        dDeltaMins, dDeltaMaxs, dDeltaTotals, dDeltaMeans, dDeltaStdDevs);

    stats.minimums.insert(stats.minimums.end(), hMins, hMins + stats.fieldCount);
    stats.maximums.insert(stats.maximums.end(), hMaxs, hMaxs + stats.fieldCount);
    stats.totals.insert(stats.totals.end(), hTotals, hTotals + stats.fieldCount);
    stats.means.insert(stats.means.end(), hMeans, hMeans + stats.fieldCount);
    stats.stdDevs.insert(stats.stdDevs.end(), hStdDevs, hStdDevs + stats.fieldCount);

    stats.deltaMinimums.insert(stats.deltaMinimums.end(), hDeltaMins, hDeltaMins + stats.fieldCount);
    stats.deltaMaximums.insert(stats.deltaMaximums.end(), hDeltaMaxs, hDeltaMaxs + stats.fieldCount);
    stats.deltaTotals.insert(stats.deltaTotals.end(), hDeltaTotals, hDeltaTotals + stats.fieldCount);
    stats.deltaMeans.insert(stats.deltaMeans.end(), hDeltaMeans, hDeltaMeans + stats.fieldCount);
    stats.deltaStdDevs.insert(stats.deltaStdDevs.end(), hDeltaStdDevs, hDeltaStdDevs + stats.fieldCount);

    return true;
}
