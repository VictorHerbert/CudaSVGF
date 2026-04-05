#pragma once
#ifndef UTILS_H
#define UTILS_h

#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

#define KERNEL __global__
#define CUDA_FUNC __device__
#define CUDA_CPU_FUNC __device__ __host__
#define LAUNCHER

//#define MEM_DEBUG

#define CUDA_ERROR_CHECK(result) \
do { \
    cudaError_t err = (result); \
    if (err != cudaSuccess) { \
        const char* errStr = cudaGetErrorString(err); \
        throw std::runtime_error( \
            std::string("CUDA error: ") + errStr + \
            " at " + __FILE__ + ":" + std::to_string(__LINE__) \
        ); \
    } \
} while(0)

#define CUDA_KERNEL_ERROR_CHECK(kernel) \
do { \
    (kernel) \
    CUDA_ERROR_CHECK(cudaGetLastError()); \
} while(0)

void printGPUProperties();


#endif