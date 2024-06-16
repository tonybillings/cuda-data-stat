#include "cds/data_stats.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>

using namespace std;

__global__ void calculate_totals(const float* data, const size_t record_count, const size_t field_count, float* totals) {
    extern __shared__ float sdata[];

    const unsigned int global_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int global_idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int local_idx = threadIdx.x;

    sdata[threadIdx.x] = 0.0f;
    __syncthreads();

    float val = FP_NAN;
    if (global_idx_x < record_count && global_idx_y < field_count) {
        val = data[global_idx_x * field_count + global_idx_y];
        sdata[local_idx] = val;
    } else {
        return;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_idx < s) {
            sdata[local_idx] += sdata[local_idx + s];
        }
        __syncthreads();
    }

    if (local_idx == 0) {
        atomicAdd(&totals[global_idx_y], sdata[0]);
    }
}

__global__ void calculate_means_stddevs(const float* data, const size_t record_count, const size_t field_count,
    const float* totals, float* means, float* std_devs) {
    extern __shared__ float sdata[];

    const unsigned int global_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int global_idx_y = blockIdx.y;
    const unsigned int local_idx = threadIdx.x;

    sdata[local_idx] = 0.0f;
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
        sdata[local_idx] = diff * diff;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_idx < s) {
            sdata[local_idx] += sdata[local_idx + s];
        }
        __syncthreads();
    }

    if (local_idx == 0) {
        atomicAdd(&std_devs[global_idx_y], sdata[0]);
    }
}

static void cleanup(float* data, float* totals, float* means, float* std_devs) {
    if (data != nullptr)
        cudaFree(data);

    if (totals != nullptr)
        cudaFree(totals);

    if (means != nullptr)
        cudaFree(means);

    if (std_devs != nullptr)
        cudaFree(std_devs);
}

// TODO: cleanup
bool calculate_stats(const vector<char>& data, const size_t field_count, const size_t record_count, DataStats& stats) {
    float *d_data, *d_totals, *d_means, *d_std_devs;

    if (cudaMalloc(&d_data, data.size()) != cudaSuccess) {
        cerr << "Error: cudaMalloc failed for d_data" << endl;
        return false;
    }
    if (cudaMemcpy(d_data, data.data(), data.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
        cerr << "Error: cudaMemcpy failed for d_data" << endl;
        cleanup(d_data, d_totals, d_means, d_std_devs);
        return false;
    }
    if (cudaMalloc(&d_totals, field_count * sizeof(float)) != cudaSuccess) {
        cerr << "Error: cudaMalloc failed for d_totals" << endl;
        cleanup(d_data, d_totals, d_means, d_std_devs);
        return false;
    }
    if (cudaMalloc(&d_means, field_count * sizeof(float)) != cudaSuccess) {
        cerr << "Error: cudaMalloc failed for d_means" << endl;
        cleanup(d_data, d_totals, d_means, d_std_devs);
        return false;
    }
    if (cudaMalloc(&d_std_devs, field_count * sizeof(float)) != cudaSuccess) {
        cerr << "Error: cudaMalloc failed for d_std_devs" << endl;
        cleanup(d_data, d_totals, d_means, d_std_devs);
        return false;
    }

    cudaError_t err = cudaMemset(d_totals, 0, field_count * sizeof(float));
    if (err != cudaSuccess) {
        cerr << "Error: cudaMemset failed for d_totals: " << cudaGetErrorString(err) << endl;
        cleanup(d_data, d_totals, d_means, d_std_devs);
        return false;
    }
    err = cudaMemset(d_means, 0, field_count * sizeof(float));
    if (err != cudaSuccess) {
        cerr << "Error: cudaMemset failed for d_means: " << cudaGetErrorString(err) << endl;
        return false;
    }
    err = cudaMemset(d_std_devs, 0, field_count * sizeof(float));
    if (err != cudaSuccess) {
        cerr << "Error: cudaMemset failed for d_std_devs: " << cudaGetErrorString(err) << endl;
        return false;
    }

    // TODO: handle case where record count is greater than max threads (implement batching)
    constexpr size_t block_width = 512; // TODO: make configurable
    const size_t block_count = (record_count + block_width - 1) / block_width;
    dim3 grid(block_count, field_count);
    dim3 block(block_width, 1);
    size_t shared_mem_size = block_width * sizeof(float);

    calculate_totals<<<grid, block, shared_mem_size>>>(d_data, record_count, field_count, d_totals);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "Error: calculate_totals failed: " << cudaGetErrorString(err) << endl;
        cleanup(d_data, d_totals, d_means, d_std_devs);
        return false;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = cudaGetLastError();
        cerr << "Error: cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << endl;
        cleanup(d_data, d_totals, d_means, d_std_devs);
        return false;
    }

    calculate_means_stddevs<<<grid, block, shared_mem_size>>>(d_data, record_count, field_count, d_totals, d_means, d_std_devs);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "Error: calculate_means_stddevs failed: " << cudaGetErrorString(err) << endl;
        cleanup(d_data, d_totals, d_means, d_std_devs);
        return false;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = cudaGetLastError();
        cerr << "Error: cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << endl;
        cleanup(d_data, d_totals, d_means, d_std_devs);
        return false;
    }

    float h_totals[field_count];
    if (cudaMemcpy(h_totals, d_totals, field_count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cerr << "Error: cudaMemcpy failed for h_totals" << endl;
        cleanup(d_data, d_totals, d_means, d_std_devs);
        return false;
    }

    float h_means[field_count];
    if (cudaMemcpy(h_means, d_means, field_count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cerr << "Error: cudaMemcpy failed for h_means" << endl;
        cleanup(d_data, d_totals, d_means, d_std_devs);
        return false;
    }

    float h_std_devs[field_count];
    if (cudaMemcpy(h_std_devs, d_std_devs, field_count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cerr << "Error: cudaMemcpy failed for h_std_devs" << endl;
        cleanup(d_data, d_totals, d_means, d_std_devs);
        return false;
    }

    for (size_t i = 0; i < field_count; i++)
        h_std_devs[i] = sqrt(h_std_devs[i] / static_cast<float>(record_count));

    cleanup(d_data, d_totals, d_means, d_std_devs);

    stats.totals.insert(stats.totals.end(), h_totals, h_totals + field_count);
    stats.means.insert(stats.means.end(), h_means, h_means + field_count);
    stats.std_devs.insert(stats.std_devs.end(), h_std_devs, h_std_devs + field_count);

    return true;
}

