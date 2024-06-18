#include "cds/math.cuh"
#include "cds/data_stats.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <c++/13/cfloat>
#include <c++/13/cstdarg>

using namespace std;

__global__ void calculate_mins(const float* data, const size_t record_count, const size_t field_count,
    float* mins) {
    extern __shared__ float sdata_mins[];

    const unsigned int global_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int global_idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int local_idx = threadIdx.x;

    sdata_mins[local_idx] = FLT_MAX;
    __syncthreads();

    if (global_idx_x < record_count && global_idx_y < field_count) {
        sdata_mins[local_idx] = data[global_idx_x * field_count + global_idx_y];
    } else {
        return;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_idx < s) {
            sdata_mins[local_idx] = fminf(sdata_mins[local_idx], sdata_mins[local_idx + s]);
        }
        __syncthreads();
    }

    if (local_idx == 0) {
        atomicMinFloat(&mins[global_idx_y], sdata_mins[0]);
    }
}

__global__ void calculate_maxs(const float* data, const size_t record_count, const size_t field_count,
    float* maxs) {
    extern __shared__ float sdata_maxs[];

    const unsigned int global_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int global_idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int local_idx = threadIdx.x;

    sdata_maxs[local_idx] = FLT_MIN;
    __syncthreads();

    if (global_idx_x < record_count && global_idx_y < field_count) {
        sdata_maxs[local_idx] = data[global_idx_x * field_count + global_idx_y];
    } else {
        return;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_idx < s) {
            sdata_maxs[local_idx] = fmaxf(sdata_maxs[local_idx], sdata_maxs[local_idx + s]);
        }
        __syncthreads();
    }

    if (local_idx == 0) {
        atomicMaxFloat(&maxs[global_idx_y], sdata_maxs[0]);
    }
}

__global__ void calculate_totals(const float* data, const size_t record_count, const size_t field_count,
    float* totals) {
    extern __shared__ float sdata_totals[];

    const unsigned int global_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int global_idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int local_idx = threadIdx.x;

    sdata_totals[local_idx] = 0.0f;
    __syncthreads();

    if (global_idx_x < record_count && global_idx_y < field_count) {
        sdata_totals[local_idx] = data[global_idx_x * field_count + global_idx_y];
    } else {
        return;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_idx < s) {
            sdata_totals[local_idx] += sdata_totals[local_idx + s];
        }
        __syncthreads();
    }

    if (local_idx == 0) {
        atomicAdd(&totals[global_idx_y], sdata_totals[0]);
    }
}

__global__ void calculate_means_stddevs(const float* data, const size_t record_count, const size_t field_count,
    const float* totals, float* means, float* std_devs) {
    extern __shared__ float sdata_stddev[];

    const unsigned int global_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int global_idx_y = blockIdx.y;
    const unsigned int local_idx = threadIdx.x;

    sdata_stddev[local_idx] = 0.0f;
    __syncthreads();

    float val = 0.0f;
    if (global_idx_x < record_count && global_idx_y < field_count) {
        val = data[global_idx_x * field_count + global_idx_y];
    }
    __syncthreads();

    if (local_idx == 0) {
        means[global_idx_y] = totals[global_idx_y] / static_cast<float>(record_count);
    }
    __syncthreads();

    if (global_idx_x < record_count && global_idx_y < field_count) {
        const float diff = val - means[global_idx_y];
        sdata_stddev[local_idx] = diff * diff;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_idx < s) {
            sdata_stddev[local_idx] += sdata_stddev[local_idx + s];
        }
        __syncthreads();
    }

    if (local_idx == 0) {
        atomicAdd(&std_devs[global_idx_y], sdata_stddev[0]);
    }
}

static void cleanup(const int count, ...) {
    va_list args;
    va_start(args, count);

    for (int i = 0; i < count; i++) {
        if (float* device_array = va_arg(args, float*); device_array != nullptr) {
            cudaFree(device_array);
        }
    }

    va_end(args);
}

// TODO: cleanup
bool calculate_stats(const vector<char>& data, const size_t field_count, const size_t record_count, DataStats& stats) {
    float *d_data, *d_mins, *d_maxs, *d_totals, *d_means, *d_std_devs;

    if (cudaMalloc(&d_data, data.size()) != cudaSuccess) {
        cerr << "Error: cudaMalloc failed for d_data" << endl;
        return false;
    }
    if (cudaMemcpy(d_data, data.data(), data.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
        cerr << "Error: cudaMemcpy failed for d_data" << endl;
        cleanup(1, d_data);
        return false;
    }
    if (cudaMalloc(&d_mins, field_count * sizeof(float)) != cudaSuccess) {
        cerr << "Error: cudaMalloc failed for d_mins" << endl;
        cleanup(1, d_data);
        return false;
    }
    if (cudaMalloc(&d_maxs, field_count * sizeof(float)) != cudaSuccess) {
        cerr << "Error: cudaMalloc failed for d_maxs" << endl;
        cleanup(2, d_data, d_mins);
        return false;
    }
    if (cudaMalloc(&d_totals, field_count * sizeof(float)) != cudaSuccess) {
        cerr << "Error: cudaMalloc failed for d_totals" << endl;
        cleanup(3, d_data, d_mins, d_maxs);
        return false;
    }
    if (cudaMalloc(&d_means, field_count * sizeof(float)) != cudaSuccess) {
        cerr << "Error: cudaMalloc failed for d_means" << endl;
        cleanup(4, d_data, d_mins, d_maxs, d_totals);
        return false;
    }
    if (cudaMalloc(&d_std_devs, field_count * sizeof(float)) != cudaSuccess) {
        cerr << "Error: cudaMalloc failed for d_std_devs" << endl;
        cleanup(5, d_data, d_mins, d_maxs, d_totals, d_means);
        return false;
    }

    cudaError_t err = cudaMemset(d_mins, FLT_MAX, field_count * sizeof(float));
    if (err != cudaSuccess) {
        cerr << "Error: cudaMemset failed for d_mins: " << cudaGetErrorString(err) << endl;
        cleanup(6, d_data, d_mins, d_maxs, d_totals, d_means, d_std_devs);
        return false;
    }
    err = cudaMemset(d_maxs, FLT_MIN, field_count * sizeof(float));
    if (err != cudaSuccess) {
        cerr << "Error: cudaMemset failed for d_maxs: " << cudaGetErrorString(err) << endl;
        cleanup(6, d_data, d_mins, d_maxs, d_totals, d_means, d_std_devs);
        return false;
    }
    err = cudaMemset(d_totals, 0, field_count * sizeof(float));
    if (err != cudaSuccess) {
        cerr << "Error: cudaMemset failed for d_totals: " << cudaGetErrorString(err) << endl;
        cleanup(6, d_data, d_mins, d_maxs, d_totals, d_means, d_std_devs);
        return false;
    }
    err = cudaMemset(d_means, 0, field_count * sizeof(float));
    if (err != cudaSuccess) {
        cerr << "Error: cudaMemset failed for d_means: " << cudaGetErrorString(err) << endl;
        cleanup(6, d_data, d_mins, d_maxs, d_totals, d_means, d_std_devs);
        return false;
    }
    err = cudaMemset(d_std_devs, 0, field_count * sizeof(float));
    if (err != cudaSuccess) {
        cerr << "Error: cudaMemset failed for d_std_devs: " << cudaGetErrorString(err) << endl;
        cleanup(6, d_data, d_mins, d_maxs, d_totals, d_means, d_std_devs);
        return false;
    }

    // TODO: handle case where record count is greater than max threads (implement batching)
    constexpr size_t block_width = 32; // TODO: make configurable
    const size_t block_count = (record_count + block_width - 1) / block_width;
    dim3 grid(block_count, field_count);
    dim3 block(block_width, 1);
    size_t shared_mem_size = block_width * sizeof(float);

    cudaStream_t stats_stream;
    cudaStreamCreate(&stats_stream);

    calculate_mins<<<grid, block, shared_mem_size, stats_stream>>>(d_data, record_count, field_count, d_mins);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "Error: calculate_mins failed: " << cudaGetErrorString(err) << endl;
        cleanup(6, d_data, d_mins, d_maxs, d_totals, d_means, d_std_devs);
        return false;
    }

    calculate_maxs<<<grid, block, shared_mem_size, stats_stream>>>(d_data, record_count, field_count, d_maxs);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "Error: calculate_maxs failed: " << cudaGetErrorString(err) << endl;
        cleanup(6, d_data, d_mins, d_maxs, d_totals, d_means, d_std_devs);
        return false;
    }

    calculate_totals<<<grid, block, shared_mem_size, stats_stream>>>(d_data, record_count, field_count, d_totals);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "Error: calculate_totals failed: " << cudaGetErrorString(err) << endl;
        cleanup(6, d_data, d_mins, d_maxs, d_totals, d_means, d_std_devs);
        return false;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = cudaGetLastError();
        cerr << "Error: cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << endl;
        cleanup(6, d_data, d_mins, d_maxs, d_totals, d_means, d_std_devs);
        return false;
    }

    calculate_means_stddevs<<<grid, block, shared_mem_size>>>(d_data, record_count, field_count, d_totals, d_means, d_std_devs);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "Error: calculate_means_stddevs failed: " << cudaGetErrorString(err) << endl;
        cleanup(6, d_data, d_mins, d_maxs, d_totals, d_means, d_std_devs);
        return false;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = cudaGetLastError();
        cerr << "Error: cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << endl;
        cleanup(6, d_data, d_mins, d_maxs, d_totals, d_means, d_std_devs);
        return false;
    }

    float h_mins[field_count];
    if (cudaMemcpy(h_mins, d_mins, field_count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cerr << "Error: cudaMemcpy failed for h_mins" << endl;
        cleanup(6, d_data, d_mins, d_maxs, d_totals, d_means, d_std_devs);
        return false;
    }

    float h_maxs[field_count];
    if (cudaMemcpy(h_maxs, d_maxs, field_count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cerr << "Error: cudaMemcpy failed for h_maxs" << endl;
        cleanup(6, d_data, d_mins, d_maxs, d_totals, d_means, d_std_devs);
        return false;
    }

    float h_totals[field_count];
    if (cudaMemcpy(h_totals, d_totals, field_count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cerr << "Error: cudaMemcpy failed for h_totals" << endl;
        cleanup(6, d_data, d_mins, d_maxs, d_totals, d_means, d_std_devs);
        return false;
    }

    float h_means[field_count];
    if (cudaMemcpy(h_means, d_means, field_count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cerr << "Error: cudaMemcpy failed for h_means" << endl;
        cleanup(6, d_data, d_mins, d_maxs, d_totals, d_means, d_std_devs);
        return false;
    }

    float h_std_devs[field_count];
    if (cudaMemcpy(h_std_devs, d_std_devs, field_count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cerr << "Error: cudaMemcpy failed for h_std_devs" << endl;
        cleanup(6, d_data, d_mins, d_maxs, d_totals, d_means, d_std_devs);
        return false;
    }

    for (size_t i = 0; i < field_count; i++)
        h_std_devs[i] = sqrt(h_std_devs[i] / static_cast<float>(record_count));

    cleanup(6, d_data, d_mins, d_maxs, d_totals, d_means, d_std_devs);

    stats.minimums.insert(stats.minimums.end(), h_mins, h_mins + field_count);
    stats.maximums.insert(stats.maximums.end(), h_maxs, h_maxs + field_count);
    stats.totals.insert(stats.totals.end(), h_totals, h_totals + field_count);
    stats.means.insert(stats.means.end(), h_means, h_means + field_count);
    stats.std_devs.insert(stats.std_devs.end(), h_std_devs, h_std_devs + field_count);

    return true;
}
