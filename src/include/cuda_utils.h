#pragma once
#ifndef UTILS_H
#define UTILS_h

/// @file cuda_utils.h
/// @brief CUDA utility macros and helper functions.

#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

#define KERNEL __global__
#define CUDA_FUNC __device__
#define CUDA_CPU_FUNC __device__ __host__
#define INLINE __inline__

/// @def CUDA_ERROR_CHECK
/// @brief Checks CUDA API call result and throws runtime error on failure.
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

/// @def CUDA_KERNEL_ERROR_CHECK
/// @brief Launches a kernel and checks for runtime launch errors.
#define CUDA_KERNEL_ERROR_CHECK(kernel) CUDA_ERROR_CHECK(cudaGetLastError())

/// @brief Prints information about available GPU devices.
/// Queries CUDA runtime for device properties and prints them to stdout.
void printGPUProperties();

#endif